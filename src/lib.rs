//! Runtime poser for Blinc skeletons.
//!
//! Consumes [`blinc_gltf`] output (`GltfSkeleton`, `GltfAnimation`),
//! evaluates animation channels each frame, composes per-joint world
//! transforms, and emits GPU-ready skinning matrices that feed into
//! `blinc_core::draw::SkinningData`.
//!
//! # Workflow
//!
//! ```ignore
//! use blinc_gltf::load_glb;
//! use blinc_skeleton::{Pose, Player};
//!
//! let scene = load_glb(&bytes)?;
//! let skin = &scene.skeletons[0];
//! let clip = &scene.animations[0];
//!
//! let mut player = Player::new(skin, clip);
//! player.set_looping(true);
//!
//! loop {
//!     let dt = /* seconds */;
//!     player.tick(dt);
//!     let skinning = player.skinning_matrices();
//!     // feed skinning into MeshData::skin on each primitive that
//!     // references this skin
//! }
//! # Ok::<_, Box<dyn std::error::Error>>(())
//! ```
//!
//! For lower-level use (custom playback, blending two clips, or
//! procedural poses), drop into [`Pose`] directly:
//!
//! ```ignore
//! let mut pose = Pose::rest(&skin.skeleton);
//! pose.evaluate(&clip, time_seconds, skin);
//! let world = pose.world_matrices(&skin.skeleton);
//! let skinning = pose.skinning_matrices(&skin.skeleton);
//! ```

use blinc_core::draw::{Skeleton, SkinningData};
use blinc_core::Mat4;
use blinc_gltf::{AnimatedProperty, AnimationSampler, GltfAnimation, GltfSkeleton};

mod sample;

pub use sample::{normalize4, quat_slerp, Sampled};

// ─────────────────────────────────────────────────────────────────────────────
// Per-joint transform
// ─────────────────────────────────────────────────────────────────────────────

/// Per-joint local transform as TRS, so animation channels can write
/// translation / rotation / scale independently without having to
/// decompose a composed 4×4 each frame.
#[derive(Debug, Clone, Copy)]
pub struct JointTransform {
    pub translation: [f32; 3],
    /// Quaternion `[x, y, z, w]`.
    pub rotation: [f32; 4],
    pub scale: [f32; 3],
}

impl JointTransform {
    pub const IDENTITY: Self = Self {
        translation: [0.0; 3],
        rotation: [0.0, 0.0, 0.0, 1.0],
        scale: [1.0; 3],
    };

    /// Compose into a column-major 4×4. Uses the same `T · R · S`
    /// convention as `blinc_gltf::NodeTransform::to_mat4`.
    pub fn to_mat4(&self) -> Mat4 {
        let r = quat_to_mat4(self.rotation);
        let s = Mat4 {
            cols: [
                [self.scale[0], 0.0, 0.0, 0.0],
                [0.0, self.scale[1], 0.0, 0.0],
                [0.0, 0.0, self.scale[2], 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        };
        let t = Mat4::translation(self.translation[0], self.translation[1], self.translation[2]);
        t.mul(&r).mul(&s)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Pose
// ─────────────────────────────────────────────────────────────────────────────

/// A full skeleton pose — one `JointTransform` per bone.
///
/// Poses are built from the rest pose via [`Pose::rest`], modified by
/// sampling animations via [`Pose::evaluate`], and finalized into
/// world / skinning matrices when the renderer needs them.
#[derive(Debug, Clone)]
pub struct Pose {
    pub joints: Vec<JointTransform>,
}

impl Pose {
    /// Rest pose — every joint starts at identity. glTF's rest pose is
    /// effectively encoded in the node TRS triples we *don't* read
    /// here; animation channels typically write every joint each
    /// frame, so starting at identity is a reasonable default. Call
    /// [`Pose::from_node_transforms`] if you need the glTF-authored
    /// rest pose.
    pub fn rest(skeleton: &Skeleton) -> Self {
        Self {
            joints: vec![JointTransform::IDENTITY; skeleton.bones.len()],
        }
    }

    /// Build a pose from the per-joint node transforms already parsed
    /// by `blinc_gltf`. `joint_nodes` maps bone index → glTF node
    /// index, and `node_trs` is the parallel list of parsed node
    /// transforms from `scene.nodes[*].transform`.
    ///
    /// Use this when a skin has a meaningful rest pose baked into the
    /// node tree (common in Blender exports), or as a starting
    /// scaffold that animations overwrite channel-by-channel.
    pub fn from_node_transforms(skeleton: &Skeleton, joints: &[JointTransform]) -> Self {
        let mut out = Self::rest(skeleton);
        for (i, j) in joints.iter().take(skeleton.bones.len()).enumerate() {
            out.joints[i] = *j;
        }
        out
    }

    /// Sample every applicable channel from `anim` at time `t` and
    /// write the results into this pose.
    ///
    /// Channels targeting nodes that aren't in `skin`'s joint list are
    /// silently skipped — in real glTFs this happens when one
    /// animation clip drives multiple skins plus unskinned props.
    pub fn evaluate(&mut self, anim: &GltfAnimation, t: f32, skin: &GltfSkeleton) {
        let bone_by_node = build_lookup(&skin.joint_nodes);
        for channel in &anim.channels {
            let Some(&bone_idx) = bone_by_node.get(&channel.target.node) else {
                continue;
            };
            let Some(joint) = self.joints.get_mut(bone_idx) else {
                continue;
            };
            apply_sample(&channel.sampler, t, channel.target.property, joint);
        }
    }

    /// Compose each joint's local transform with its ancestor chain
    /// to produce per-joint world-space 4×4 matrices.
    pub fn world_matrices(&self, skeleton: &Skeleton) -> Vec<Mat4> {
        let n = self.joints.len().min(skeleton.bones.len());
        let mut world = vec![Mat4::IDENTITY; n];
        for i in 0..n {
            let local = self.joints[i].to_mat4();
            world[i] = if let Some(p) = skeleton.bones[i].parent {
                // Parent MUST precede the child in bone order for this
                // to produce correct composition in a single pass.
                // glTF writers respect that ordering; if a custom
                // importer violates it, use a DFS variant instead.
                if p < i {
                    world[p].mul(&local)
                } else {
                    local
                }
            } else {
                local
            };
        }
        world
    }

    /// GPU-ready skinning matrices: `world_i * inverse_bind_i`. These
    /// go straight into `SkinningData::joint_matrices`.
    pub fn skinning_matrices(&self, skeleton: &Skeleton) -> Vec<Mat4> {
        let world = self.world_matrices(skeleton);
        world
            .into_iter()
            .zip(&skeleton.bones)
            .map(|(w, b)| {
                let ibm = mat4_from_array(b.inverse_bind_matrix);
                w.mul(&ibm)
            })
            .collect()
    }

    /// Convenience wrapper: build a `SkinningData` suitable for dropping
    /// into `MeshData::skin`. Matrices are flattened to `[f32; 16]` in
    /// the column-major layout the GPU expects.
    pub fn skinning_data(&self, skeleton: &Skeleton) -> SkinningData {
        let matrices = self
            .skinning_matrices(skeleton)
            .into_iter()
            .map(flatten_mat4)
            .collect();
        SkinningData {
            joint_matrices: matrices,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Player
// ─────────────────────────────────────────────────────────────────────────────

/// A stateful animation player. Wraps a `Pose`, tracks time,
/// handles play/pause/seek/looping, and emits skinning matrices on
/// demand.
///
/// Clone is cheap — the referenced `GltfSkeleton` / `GltfAnimation`
/// aren't copied (they're read-only data the caller owns elsewhere).
pub struct Player<'a> {
    pose: Pose,
    skin: &'a GltfSkeleton,
    animation: &'a GltfAnimation,
    time: f32,
    speed: f32,
    looping: bool,
    playing: bool,
    duration: f32,
}

impl<'a> Player<'a> {
    /// Build a player that drives `skin` with `animation`.
    pub fn new(skin: &'a GltfSkeleton, animation: &'a GltfAnimation) -> Self {
        Self {
            pose: Pose::rest(&skin.skeleton),
            skin,
            animation,
            time: 0.0,
            speed: 1.0,
            looping: false,
            playing: true,
            duration: clip_duration(animation),
        }
    }

    /// Advance playback by `dt` seconds, then re-evaluate the pose.
    /// No-op when paused.
    pub fn tick(&mut self, dt: f32) {
        if !self.playing || self.duration <= 0.0 {
            // Still re-evaluate so a freshly-paused player renders its
            // frozen pose correctly.
            self.pose.evaluate(self.animation, self.time, self.skin);
            return;
        }
        self.time += dt * self.speed;
        if self.time > self.duration {
            if self.looping {
                self.time = self.time.rem_euclid(self.duration);
            } else {
                self.time = self.duration;
                self.playing = false;
            }
        } else if self.time < 0.0 {
            if self.looping {
                self.time = self.time.rem_euclid(self.duration);
            } else {
                self.time = 0.0;
                self.playing = false;
            }
        }
        self.pose.evaluate(self.animation, self.time, self.skin);
    }

    /// Jump to an absolute time in the clip, clamped to `[0, duration]`.
    pub fn seek(&mut self, t: f32) {
        self.time = t.clamp(0.0, self.duration);
        self.pose.evaluate(self.animation, self.time, self.skin);
    }

    pub fn set_playing(&mut self, playing: bool) {
        self.playing = playing;
    }

    pub fn set_looping(&mut self, looping: bool) {
        self.looping = looping;
    }

    /// Playback speed multiplier. `1.0` = normal, `-1.0` = reverse.
    pub fn set_speed(&mut self, speed: f32) {
        self.speed = speed;
    }

    pub fn time(&self) -> f32 {
        self.time
    }

    pub fn duration(&self) -> f32 {
        self.duration
    }

    pub fn is_playing(&self) -> bool {
        self.playing
    }

    pub fn pose(&self) -> &Pose {
        &self.pose
    }

    /// GPU-ready skinning matrices for the current pose.
    pub fn skinning_matrices(&self) -> Vec<Mat4> {
        self.pose.skinning_matrices(&self.skin.skeleton)
    }

    /// Build a `SkinningData` for drop-in use with
    /// `MeshData::skin = Some(...)`.
    pub fn skinning_data(&self) -> SkinningData {
        self.pose.skinning_data(&self.skin.skeleton)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internals
// ─────────────────────────────────────────────────────────────────────────────

fn build_lookup(joint_nodes: &[usize]) -> std::collections::HashMap<usize, usize> {
    joint_nodes
        .iter()
        .copied()
        .enumerate()
        .map(|(bone_idx, node_idx)| (node_idx, bone_idx))
        .collect()
}

fn apply_sample(
    sampler: &AnimationSampler,
    t: f32,
    property: AnimatedProperty,
    joint: &mut JointTransform,
) {
    let Some(sampled) = sample::sample(sampler, t) else {
        return;
    };
    match (property, sampled) {
        (AnimatedProperty::Translation, Sampled::Vec3(v)) => joint.translation = v,
        (AnimatedProperty::Scale, Sampled::Vec3(v)) => joint.scale = v,
        (AnimatedProperty::Rotation, Sampled::Vec4(q)) => joint.rotation = normalize4(q),
        // Morph weights don't live on JointTransform — they're a
        // mesh-level channel. A future revision will expose a
        // MorphWeights sink on `Pose`; for now, silently skip.
        (AnimatedProperty::MorphWeights, Sampled::Scalars(_)) => {}
        // Type mismatches (e.g. translation channel with Vec4 values)
        // are invalid glTF; skip defensively rather than panic.
        _ => {}
    }
}

fn clip_duration(anim: &GltfAnimation) -> f32 {
    // Duration is the latest keyframe time across all channels.
    anim.channels
        .iter()
        .filter_map(|ch| ch.sampler.times.last().copied())
        .fold(0.0f32, |acc, t| acc.max(t))
}

// ─────────────────────────────────────────────────────────────────────────────
// Matrix helpers (duplicated small pieces rather than adding a blinc_core dep)
// ─────────────────────────────────────────────────────────────────────────────

fn mat4_from_array(m: [f32; 16]) -> Mat4 {
    Mat4 {
        cols: [
            [m[0], m[1], m[2], m[3]],
            [m[4], m[5], m[6], m[7]],
            [m[8], m[9], m[10], m[11]],
            [m[12], m[13], m[14], m[15]],
        ],
    }
}

fn flatten_mat4(m: Mat4) -> [f32; 16] {
    let c = m.cols;
    [
        c[0][0], c[0][1], c[0][2], c[0][3], //
        c[1][0], c[1][1], c[1][2], c[1][3], //
        c[2][0], c[2][1], c[2][2], c[2][3], //
        c[3][0], c[3][1], c[3][2], c[3][3], //
    ]
}

fn quat_to_mat4(q: [f32; 4]) -> Mat4 {
    let [x, y, z, w] = q;
    let xx = x * x;
    let yy = y * y;
    let zz = z * z;
    let xy = x * y;
    let xz = x * z;
    let yz = y * z;
    let wx = w * x;
    let wy = w * y;
    let wz = w * z;
    // Same column-major layout as blinc_gltf::node — kept here to
    // avoid pulling a matrix-ops dep for one function.
    Mat4 {
        cols: [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy + wz), 2.0 * (xz - wy), 0.0],
            [2.0 * (xy - wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz + wx), 0.0],
            [2.0 * (xz + wy), 2.0 * (yz - wx), 1.0 - 2.0 * (xx + yy), 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    }
}

// Re-export types that downstream code commonly holds alongside a Pose.
pub use blinc_core::draw::{Bone, Skeleton as CoreSkeleton, SkinningData as CoreSkinningData};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_joint_produces_identity_matrix() {
        let m = JointTransform::IDENTITY.to_mat4();
        for c in 0..4 {
            for r in 0..4 {
                let expected = if c == r { 1.0 } else { 0.0 };
                assert!((m.cols[c][r] - expected).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn rest_pose_shape_matches_skeleton() {
        let skel = Skeleton {
            bones: vec![
                Bone {
                    name: "a".into(),
                    parent: None,
                    inverse_bind_matrix: identity16(),
                },
                Bone {
                    name: "b".into(),
                    parent: Some(0),
                    inverse_bind_matrix: identity16(),
                },
            ],
        };
        let pose = Pose::rest(&skel);
        assert_eq!(pose.joints.len(), 2);
    }

    #[test]
    fn world_matrix_composes_parent_then_child() {
        let skel = Skeleton {
            bones: vec![
                Bone {
                    name: "root".into(),
                    parent: None,
                    inverse_bind_matrix: identity16(),
                },
                Bone {
                    name: "child".into(),
                    parent: Some(0),
                    inverse_bind_matrix: identity16(),
                },
            ],
        };
        let mut pose = Pose::rest(&skel);
        // Root translated by (10, 0, 0); child translated by (0, 5, 0) locally.
        pose.joints[0].translation = [10.0, 0.0, 0.0];
        pose.joints[1].translation = [0.0, 5.0, 0.0];
        let world = pose.world_matrices(&skel);
        // Child's world = Tx10 · Ty5 → translation component should
        // be (10, 5, 0).
        assert!((world[1].cols[3][0] - 10.0).abs() < 1e-5);
        assert!((world[1].cols[3][1] - 5.0).abs() < 1e-5);
    }

    fn identity16() -> [f32; 16] {
        [
            1.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0, //
            0.0, 0.0, 0.0, 1.0, //
        ]
    }
}
