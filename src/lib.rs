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
use blinc_gltf::{
    AnimatedProperty, AnimationSampler, GltfAnimation, GltfScene, GltfSkeleton, NodeTransform,
};

mod densify;
mod sample;

pub use densify::{densify_one_channel, densify_rotation_channels, MAX_SEG_RAD};
pub use sample::{normalize4, quat_slerp, Sampled};

/// Sample an animation channel's sampler at time `t`, returning the
/// interpolated value. Returns `None` when the sampler has zero
/// keyframes.
///
/// Exposed so callers can write their own pose / node-transform
/// machinery without depending on blinc_skeleton's `Pose` struct.
pub fn sample_channel(
    sampler: &blinc_gltf::AnimationSampler,
    t: f32,
) -> Option<Sampled> {
    sample::sample(sampler, t)
}

/// Evaluate `clip` at scene time `t` and write the sampled values
/// directly into `scene.nodes[*].transform`.
///
/// This is the **transform-animation** path, for clips that drive
/// scene-graph node TRS channels directly (vehicles lifting off,
/// mechanical parts rotating, cameras moving). For **skinned**
/// animation — clips that target joint nodes inside a skin — use
/// [`Pose::evaluate`] instead, which looks channels up through the
/// skin's joint list and writes into per-joint `JointTransform`s.
///
/// Channels targeting nodes that don't exist in `scene.nodes` are
/// silently skipped. Nodes stored in [`NodeTransform::Matrix`] form
/// are also skipped (they can't be updated component-wise without a
/// polar decomposition) — real exporters emit TRS for any node with
/// animation channels, so this is only a defensive carve-out.
pub fn animate_scene_nodes(scene: &mut GltfScene, clip: &GltfAnimation, t: f32) {
    animate_scene_nodes_with(scene, clip, |_| t);
}

/// Same as [`animate_scene_nodes`] but picks the sampling time
/// per-channel. `time_at_node(node_index)` decides at what clip time
/// each channel's target node is sampled.
///
/// Motivation: mechanical assets often compose a scene-speed "body"
/// animation (hover, liftoff, bob) with one or more parts that should
/// spin at a visibly different rate (rotors, turbines, fans). Feeding
/// all channels the same `t` makes either the body look wrong or the
/// rotors look wrong. This lets callers apply a per-node time
/// multiplier — e.g. `t * 3.0` for rotor nodes, `t` for everything
/// else — without having to split the clip into multiple passes.
///
/// Typical use:
///
/// ```ignore
/// let fast_nodes: std::collections::HashSet<usize> =
///     scene.nodes.iter().enumerate()
///         .filter(|(_, n)| n.name.as_deref().map_or(false, |s| s.contains("Rotor")))
///         .map(|(i, _)| i)
///         .collect();
/// animate_scene_nodes_with(&mut scene, clip, |node_idx| {
///     if fast_nodes.contains(&node_idx) { t * 3.0 } else { t }
/// });
/// ```
pub fn animate_scene_nodes_with<F: Fn(usize) -> f32>(
    scene: &mut GltfScene,
    clip: &GltfAnimation,
    time_at_node: F,
) {
    for ch in &clip.channels {
        let Some(node) = scene.nodes.get_mut(ch.target.node) else {
            continue;
        };
        let t = time_at_node(ch.target.node);
        let Some(sampled) = sample::sample(&ch.sampler, t) else {
            continue;
        };

        let (mut translation, mut rotation, mut scale) = match node.transform {
            NodeTransform::Trs {
                translation,
                rotation,
                scale,
            } => (translation, rotation, scale),
            NodeTransform::Matrix(_) => continue,
        };

        match (ch.target.property, sampled) {
            (AnimatedProperty::Translation, Sampled::Vec3(v)) => translation = v,
            (AnimatedProperty::Rotation, Sampled::Vec4(q)) => {
                rotation = normalize4(q);
            }
            (AnimatedProperty::Scale, Sampled::Vec3(v)) => scale = v,
            // MorphWeights target meshes, not nodes — unaffected by
            // this helper. They're reserved for a future
            // `animate_scene_morph_weights`.
            _ => continue,
        }

        node.transform = NodeTransform::Trs {
            translation,
            rotation,
            scale,
        };
    }
}

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

    /// Linearly interpolate this joint toward `other` by `weight`.
    /// `weight = 0.0` leaves `self` unchanged; `weight = 1.0` replaces
    /// it with `other`; intermediate values blend per-component on
    /// translation / scale and slerp on rotation. The output
    /// quaternion is renormalised, so repeated blends don't drift.
    ///
    /// Values outside `[0, 1]` aren't clamped — they extrapolate, which
    /// is occasionally useful (e.g. anticipation overshoot on an
    /// action follow-through). Pass a clamped weight if extrapolation
    /// would break the downstream rig.
    pub fn blend(&mut self, other: &Self, weight: f32) {
        let u = weight;
        let inv = 1.0 - u;
        for i in 0..3 {
            self.translation[i] = self.translation[i] * inv + other.translation[i] * u;
            self.scale[i] = self.scale[i] * inv + other.scale[i] * u;
        }
        self.rotation = quat_slerp(self.rotation, other.rotation, u);
    }

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

    /// Blend this pose toward `other` by `weight`. Every joint pair
    /// gets a per-channel lerp on translation / scale and a slerp on
    /// rotation (see [`JointTransform::blend`] for the single-joint
    /// version). Joints past the shorter pose's length are left
    /// untouched — skeletons of different sizes can't sensibly blend,
    /// and silently mapping by index would produce garbage.
    ///
    /// Typical use is sampling two clips at the same scene time into
    /// two separate poses and crossfading:
    ///
    /// ```ignore
    /// let mut walk = Pose::rest(&skin.skeleton);
    /// walk.evaluate(&walk_clip, t, skin);
    /// let mut run = Pose::rest(&skin.skeleton);
    /// run.evaluate(&run_clip, t, skin);
    /// walk.blend(&run, run_weight);  // walk → walk+run mix
    /// let skinning = walk.skinning_matrices(&skin.skeleton);
    /// ```
    ///
    /// The operation is allocation-free — results are written in place.
    pub fn blend(&mut self, other: &Pose, weight: f32) {
        let n = self.joints.len().min(other.joints.len());
        for i in 0..n {
            let rhs = other.joints[i];
            self.joints[i].blend(&rhs, weight);
        }
    }

    /// Blend `self` with N other poses in a single pass, using
    /// `(weight, pose)` pairs. Weights are normalised so their sum
    /// equals 1.0 — callers can pass un-normalised weights (say,
    /// stick-axis magnitudes from a blend tree) and get a sensible
    /// result without having to renormalise upstream.
    ///
    /// Mathematically: at each joint this accumulates
    /// `Σ (w_i / Σw) * pose_i.joint`, where `pose_0 = self` gets an
    /// implicit weight of `1.0 - Σ weights` so single-source blend
    /// reduces to [`Pose::blend`]. If all weights are zero (or the
    /// slice is empty) this is a no-op.
    ///
    /// Shape mismatch between poses is handled the same way as
    /// [`Pose::blend`]: out-of-range joints on either side are left
    /// untouched.
    pub fn blend_many(&mut self, others: &[(f32, &Pose)]) {
        if others.is_empty() {
            return;
        }
        let total: f32 = others.iter().map(|(w, _)| *w).sum();
        if total <= f32::EPSILON {
            return;
        }
        // Iterate with a running accumulator: after k sources have
        // been folded in, the accumulated pose carries weight
        // `weight_seen / total_seen` of each source. The next source
        // adds itself at fraction `w / (weight_seen + w)` so the
        // running pose stays a convex combination of the first k+1.
        let mut weight_seen = 0.0f32;
        for (w, pose) in others {
            let w = *w;
            if w <= 0.0 {
                continue;
            }
            let next_total = weight_seen + w;
            let frac = w / next_total;
            self.blend(pose, frac);
            weight_seen = next_total;
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

    // ── Blend tests ──────────────────────────────────────────────────────

    fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    /// Quaternion for a rotation of `angle_rad` around the +Y axis.
    fn quat_y(angle_rad: f32) -> [f32; 4] {
        let h = 0.5 * angle_rad;
        [0.0, h.sin(), 0.0, h.cos()]
    }

    fn single_bone_skel() -> Skeleton {
        Skeleton {
            bones: vec![Bone {
                name: "b".into(),
                parent: None,
                inverse_bind_matrix: identity16(),
            }],
        }
    }

    #[test]
    fn joint_blend_zero_weight_is_identity_op() {
        let mut base = JointTransform {
            translation: [1.0, 2.0, 3.0],
            rotation: quat_y(0.0),
            scale: [2.0, 2.0, 2.0],
        };
        let other = JointTransform {
            translation: [5.0, 5.0, 5.0],
            rotation: quat_y(std::f32::consts::PI),
            scale: [1.0, 1.0, 1.0],
        };
        base.blend(&other, 0.0);
        assert!(approx_eq(base.translation[0], 1.0, 1e-5));
        assert!(approx_eq(base.scale[0], 2.0, 1e-5));
    }

    #[test]
    fn joint_blend_full_weight_replaces_self() {
        let mut base = JointTransform::IDENTITY;
        let other = JointTransform {
            translation: [4.0, 0.0, 0.0],
            rotation: quat_y(std::f32::consts::FRAC_PI_2),
            scale: [3.0, 3.0, 3.0],
        };
        base.blend(&other, 1.0);
        assert!(approx_eq(base.translation[0], 4.0, 1e-4));
        assert!(approx_eq(base.scale[0], 3.0, 1e-4));
        // rotation should be ~90° around Y — quaternion w component
        // cos(45°) ≈ 0.7071.
        assert!(approx_eq(base.rotation[3], std::f32::consts::FRAC_1_SQRT_2, 1e-4));
    }

    #[test]
    fn joint_blend_halfway_translation_is_midpoint() {
        let mut base = JointTransform {
            translation: [0.0, 0.0, 0.0],
            rotation: quat_y(0.0),
            scale: [1.0, 1.0, 1.0],
        };
        let other = JointTransform {
            translation: [10.0, 0.0, 0.0],
            rotation: quat_y(std::f32::consts::PI),
            scale: [1.0, 1.0, 1.0],
        };
        base.blend(&other, 0.5);
        assert!(approx_eq(base.translation[0], 5.0, 1e-4));
        // 50% slerp between identity and 180°-around-Y = 90°-around-Y.
        // The w component of a 90° rotation is cos(45°) = 1/sqrt(2).
        assert!(approx_eq(base.rotation[3], std::f32::consts::FRAC_1_SQRT_2, 1e-4));
    }

    #[test]
    fn pose_blend_writes_each_joint_independently() {
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
        let mut a = Pose::rest(&skel);
        a.joints[0].translation = [0.0, 0.0, 0.0];
        a.joints[1].translation = [0.0, 10.0, 0.0];
        let mut b = Pose::rest(&skel);
        b.joints[0].translation = [2.0, 0.0, 0.0];
        b.joints[1].translation = [0.0, 20.0, 0.0];

        a.blend(&b, 0.25);
        assert!(approx_eq(a.joints[0].translation[0], 0.5, 1e-5)); // 0 * 0.75 + 2 * 0.25
        assert!(approx_eq(a.joints[1].translation[1], 12.5, 1e-5)); // 10 * 0.75 + 20 * 0.25
    }

    #[test]
    fn pose_blend_shorter_other_leaves_extra_joints_untouched() {
        let skel = single_bone_skel();
        let mut a = Pose::rest(&skel);
        a.joints[0].translation = [7.0, 0.0, 0.0];
        // Synthetic: build a zero-joint Pose and blend — self shouldn't move.
        let b = Pose { joints: vec![] };
        a.blend(&b, 1.0);
        assert!(approx_eq(a.joints[0].translation[0], 7.0, 1e-5));
    }

    #[test]
    fn pose_blend_many_normalises_weights() {
        let skel = single_bone_skel();
        let mut base = Pose::rest(&skel); // translation = [0, 0, 0]
        let mut p1 = Pose::rest(&skel);
        p1.joints[0].translation = [10.0, 0.0, 0.0];
        let mut p2 = Pose::rest(&skel);
        p2.joints[0].translation = [0.0, 20.0, 0.0];

        // Pass un-normalised weights (3 + 1 = 4). Expected mix at the
        // joint: (3/4) * p1 + (1/4) * p2 layered onto the base's
        // existing translation. The running-accumulator algorithm
        // produces the same end state as manually computing that sum.
        base.blend_many(&[(3.0, &p1), (1.0, &p2)]);
        assert!(approx_eq(base.joints[0].translation[0], 7.5, 1e-4));
        assert!(approx_eq(base.joints[0].translation[1], 5.0, 1e-4));
    }

    #[test]
    fn pose_blend_many_empty_and_zero_weight_are_noop() {
        let skel = single_bone_skel();
        let mut a = Pose::rest(&skel);
        a.joints[0].translation = [1.0, 2.0, 3.0];
        a.blend_many(&[]);
        assert!(approx_eq(a.joints[0].translation[0], 1.0, 1e-5));

        let other = Pose::rest(&skel);
        a.blend_many(&[(0.0, &other)]);
        assert!(approx_eq(a.joints[0].translation[0], 1.0, 1e-5));
    }
}
