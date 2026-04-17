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
mod fsm;
mod ik;
mod sample;

pub use densify::{densify_one_channel, densify_rotation_channels, MAX_SEG_RAD};
pub use fsm::{ClipState, Condition, Parameters, StateIndex, StateMachine, Transition};
pub use ik::{rotation_from_to, solve_fabrik, solve_two_bone, TwoBoneSolution};
pub use sample::{normalize4, quat_slerp, Sampled};

use ik::{v3_length, v3_normalise_or, v3_sub};

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
            // this helper. Callers feed them to the renderer via
            // [`animate_scene_morph_weights`].
            _ => continue,
        }

        node.transform = NodeTransform::Trs {
            translation,
            rotation,
            scale,
        };
    }
}

/// Sample every `MorphWeights` channel in `clip` at time `t` and
/// return a `node_index → weights` map. One entry per mesh-bearing
/// node the clip animates; nodes without a morph channel are absent.
///
/// This is the morph-target counterpart to [`animate_scene_nodes`],
/// which handles TRS channels only. Run both per frame, then pass the
/// sampled weights into each draw via `MeshData::morph_weights` (the
/// map key is the scene node index, so look up by the same node you
/// used to fetch the world transform).
///
/// ```ignore
/// // Per-frame:
/// animate_scene_nodes(&mut scene, &clip, t);
/// let morphs = animate_scene_morph_weights(&clip, t);
/// let world = scene.compute_world_transforms();
/// for (node_i, node) in scene.nodes.iter().enumerate() {
///     let Some(mesh_i) = node.mesh else { continue };
///     for prim in &arc_meshes[mesh_i] {
///         let mut per_draw = (**prim).clone();
///         if let Some(w) = morphs.get(&node_i) {
///             per_draw.morph_weights = w.clone();
///         }
///         ctx.draw_mesh_data(Arc::new(per_draw), world[node_i]);
///     }
/// }
/// ```
pub fn animate_scene_morph_weights(
    clip: &GltfAnimation,
    t: f32,
) -> std::collections::HashMap<usize, Vec<f32>> {
    let mut out = std::collections::HashMap::new();
    for ch in &clip.channels {
        if ch.target.property != AnimatedProperty::MorphWeights {
            continue;
        }
        if let Some(w) = sample_morph_weights(&ch.sampler, t) {
            out.insert(ch.target.node, w);
        }
    }
    out
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

    /// Compute the per-joint *delta* that takes `rest` to `target`.
    /// The result isn't a full pose — it's a transform that means
    /// "translate by Δt, scale by Δs, rotate by Δq", to be layered
    /// on top of some other base pose via [`Self::apply_delta`].
    ///
    /// Translation is subtractive (`target.t − rest.t`); scale is
    /// multiplicative (`target.s / rest.s`, component-wise) with a
    /// 1e-6 denominator floor so a degenerate rest scale of 0 doesn't
    /// blow up; rotation is `conj(rest) · target`, the quaternion
    /// `q` such that `rest · q = target`.
    ///
    /// Typical use: sample the author-provided "aim rest pose" and
    /// "aim full-right pose" at the same time, call `delta(rest, full)`
    /// to get the additive overlay, then `apply_delta(&delta, stick_x)`
    /// on the base locomotion each frame.
    pub fn delta(rest: &Self, target: &Self) -> Self {
        let mut s = [1.0; 3];
        for i in 0..3 {
            // Floor avoids div-by-zero on pathological rest scales.
            let d = if rest.scale[i].abs() < 1e-6 {
                rest.scale[i].signum().max(1e-6)
            } else {
                rest.scale[i]
            };
            s[i] = target.scale[i] / d;
        }
        let mut t = [0.0; 3];
        for i in 0..3 {
            t[i] = target.translation[i] - rest.translation[i];
        }
        Self {
            translation: t,
            rotation: quat_mul(quat_conj(rest.rotation), target.rotation),
            scale: s,
        }
    }

    /// Apply an additive `delta` (from [`Self::delta`]) on top of
    /// `self`, scaled by `weight`. `weight = 0.0` is a no-op;
    /// `weight = 1.0` fully layers the delta on.
    ///
    /// Math, per-channel:
    /// - translation: `self.t += delta.t * weight`
    /// - scale:       `self.s *= lerp(1.0, delta.s, weight)` (so
    ///   weight 0 multiplies by 1.0 and weight 1 multiplies by the
    ///   full delta factor)
    /// - rotation:    `self.r = self.r · slerp(identity, delta.r, weight)`
    ///   (local post-multiply — the layer rotates in the bone's
    ///   own frame, then the base places it in the skeleton)
    ///
    /// The quaternion is renormalised after the multiply so repeated
    /// layered calls don't drift.
    pub fn apply_delta(&mut self, delta: &Self, weight: f32) {
        for i in 0..3 {
            self.translation[i] += delta.translation[i] * weight;
            // lerp(1, delta.s, w) = 1 + w * (delta.s - 1).
            let factor = 1.0 + weight * (delta.scale[i] - 1.0);
            self.scale[i] *= factor;
        }
        let scaled_delta = quat_slerp([0.0, 0.0, 0.0, 1.0], delta.rotation, weight);
        self.rotation = normalize4(quat_mul(self.rotation, scaled_delta));
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

/// A full skeleton pose — one `JointTransform` per bone, plus a
/// side-table of per-node morph weights.
///
/// Poses are built from the rest pose via [`Pose::rest`], modified by
/// sampling animations via [`Pose::evaluate`], and finalized into
/// world / skinning matrices when the renderer needs them.
///
/// Morph weights are keyed by *scene node index* (not bone index) —
/// a node carries a `mesh`, and a mesh owns the morph targets. The
/// map is dynamically sized by `evaluate` as `MorphWeights` channels
/// are encountered; meshes without morph data produce an empty map.
#[derive(Debug, Clone)]
pub struct Pose {
    pub joints: Vec<JointTransform>,
    pub morph_weights: std::collections::HashMap<usize, Vec<f32>>,
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
            morph_weights: std::collections::HashMap::new(),
        }
    }

    /// Read the weights written for `node` by the last
    /// [`Self::evaluate`] call. Returns `None` if the clip didn't
    /// animate morph weights on that node.
    pub fn morph_weights_for_node(&self, node: usize) -> Option<&[f32]> {
        self.morph_weights.get(&node).map(Vec::as_slice)
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

    /// Build a pose by reading the current TRS of each joint's glTF
    /// node directly from `scene`. This is the right starting point
    /// for skinned assets whose bind pose is non-trivial — call it
    /// *after* [`animate_scene_nodes`] to carry the clip's animated
    /// transforms into the pose, and bones the clip doesn't touch
    /// keep their authored rest TRS instead of collapsing to identity
    /// (which is what [`Pose::rest`] would do).
    ///
    /// `NodeTransform::Matrix(_)` joints fall back to identity — real
    /// skeletons export as TRS; this is a defensive carve-out.
    ///
    /// ```ignore
    /// animate_scene_nodes(&mut scene, &clip, t);
    /// let mut pose = Pose::from_scene(&scene, skin);
    /// pose.evaluate(&clip, t, skin); // populates pose.morph_weights
    /// let skinning = pose.skinning_data(&skin.skeleton);
    /// ```
    pub fn from_scene(scene: &GltfScene, skin: &GltfSkeleton) -> Self {
        let mut out = Self::rest(&skin.skeleton);
        for (bone_idx, &node_idx) in skin.joint_nodes.iter().enumerate() {
            if bone_idx >= out.joints.len() {
                break;
            }
            if let Some(node) = scene.nodes.get(node_idx) {
                if let NodeTransform::Trs {
                    translation,
                    rotation,
                    scale,
                } = node.transform
                {
                    out.joints[bone_idx] = JointTransform {
                        translation,
                        rotation,
                        scale,
                    };
                }
            }
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
            // Morph-weights channels target mesh-bearing nodes — they
            // don't live on the skeleton's joint list, so we route
            // them to the pose's per-node morph-weight sink instead
            // of the joint-TRS sink below.
            if channel.target.property == AnimatedProperty::MorphWeights {
                if let Some(weights) = sample_morph_weights(&channel.sampler, t) {
                    self.morph_weights.insert(channel.target.node, weights);
                }
                continue;
            }
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

    /// Build a *delta pose* whose per-joint transforms take `rest` to
    /// `target` via [`JointTransform::delta`]. Feed the result to
    /// [`Pose::apply_delta`] to layer it on top of a base pose.
    ///
    /// The two inputs must share a skeleton — same joint ordering,
    /// same count. Anything past the shorter list is dropped.
    ///
    /// Typical wiring for an "aim" overlay:
    ///
    /// ```ignore
    /// // Authored once at load time from two aim keyframes:
    /// let aim_layer = Pose::delta(&aim_rest_pose, &aim_full_pose);
    ///
    /// // Every frame:
    /// let mut base = Pose::rest(&skin.skeleton);
    /// base.evaluate(&locomotion_clip, t, skin);
    /// base.apply_delta(&aim_layer, aim_strength); // stick_x ∈ [−1, 1]
    /// let skinning = base.skinning_matrices(&skin.skeleton);
    /// ```
    pub fn delta(rest: &Pose, target: &Pose) -> Pose {
        let n = rest.joints.len().min(target.joints.len());
        let mut joints = Vec::with_capacity(n);
        for i in 0..n {
            joints.push(JointTransform::delta(&rest.joints[i], &target.joints[i]));
        }
        Pose {
            joints,
            morph_weights: std::collections::HashMap::new(),
        }
    }

    /// Layer `delta` (from [`Pose::delta`] or composed elsewhere) on
    /// top of this pose at `weight`. Per-joint math matches
    /// [`JointTransform::apply_delta`]; joints past the shorter pose's
    /// length are left untouched.
    ///
    /// Allocation-free — the result is written into `self` in place.
    pub fn apply_delta(&mut self, delta: &Pose, weight: f32) {
        let n = self.joints.len().min(delta.joints.len());
        for i in 0..n {
            let d = delta.joints[i];
            self.joints[i].apply_delta(&d, weight);
        }
    }

    /// Compose each joint's local rotation with its ancestor chain
    /// to produce per-joint world-space quaternions.
    ///
    /// Cheaper than [`Self::world_matrices`] when callers only need
    /// rotations (IK solvers, constraint math) — stays in quaternion
    /// space instead of building full 4×4 matrices and extracting the
    /// rotation component.
    pub fn world_rotations(&self, skeleton: &Skeleton) -> Vec<[f32; 4]> {
        let n = self.joints.len().min(skeleton.bones.len());
        let mut world = vec![[0.0, 0.0, 0.0, 1.0]; n];
        for i in 0..n {
            let local = self.joints[i].rotation;
            world[i] = if let Some(p) = skeleton.bones[i].parent {
                if p < i {
                    normalize4(quat_mul(world[p], local))
                } else {
                    local
                }
            } else {
                local
            };
        }
        world
    }

    /// Solve a two-bone IK chain (root → middle → end) and write the
    /// resulting local rotations back into this pose.
    ///
    /// Positions and the pole are interpreted in world space — same
    /// coordinate system as the pose's `world_matrices`. Bone lengths
    /// are extracted from the *current* pose (not the rest pose), so
    /// IK respects any translation animation already in place.
    ///
    /// Silently no-ops if any bone index is out of range. End's
    /// rotation is left untouched (only root and middle are
    /// reoriented to get end to `target`).
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Foot-placement: anchor the foot to a ground-trace point
    /// // while the leg bends naturally.
    /// let ground = raycast_down(body_position);
    /// let knee_pole = body_position + body_forward * 0.3; // knees forward
    /// pose.solve_two_bone_ik(
    ///     &skin.skeleton,
    ///     hip_bone,
    ///     knee_bone,
    ///     foot_bone,
    ///     ground,
    ///     knee_pole,
    /// );
    /// ```
    pub fn solve_two_bone_ik(
        &mut self,
        skeleton: &Skeleton,
        root: usize,
        middle: usize,
        end: usize,
        target: [f32; 3],
        pole: [f32; 3],
    ) {
        let joint_count = self.joints.len().min(skeleton.bones.len());
        if root >= joint_count || middle >= joint_count || end >= joint_count {
            return;
        }

        let world_mats = self.world_matrices(skeleton);
        let world_rots = self.world_rotations(skeleton);

        // Extract world-space positions from the translation column
        // of each joint's world matrix.
        let world_pos = |i: usize| -> [f32; 3] {
            let c = world_mats[i].cols[3];
            [c[0], c[1], c[2]]
        };
        let root_pos = world_pos(root);
        let middle_pos = world_pos(middle);
        let end_pos = world_pos(end);

        let upper_dir = v3_sub(middle_pos, root_pos);
        let lower_dir = v3_sub(end_pos, middle_pos);
        let l_upper = v3_length(upper_dir);
        let l_lower = v3_length(lower_dir);

        if l_upper < 1e-5 || l_lower < 1e-5 {
            return;
        }

        let sol = ik::solve_two_bone(root_pos, l_upper, l_lower, target, pole);

        // World-space rotation delta for the root joint: rotate the
        // upper-bone direction from its current orientation onto the
        // solve's desired direction.
        let current_upper_dir = v3_normalise_or(upper_dir, [0.0, 1.0, 0.0]);
        let desired_upper_dir = v3_normalise_or(v3_sub(sol.middle, root_pos), [0.0, 1.0, 0.0]);
        let root_delta_world = ik::rotation_from_to(current_upper_dir, desired_upper_dir);

        // After root rotates, the lower bone's world direction picks
        // up that rotation. The middle's own delta is whatever gets
        // the already-rotated lower onto the solve's desired direction.
        let current_lower_dir = v3_normalise_or(lower_dir, [0.0, 1.0, 0.0]);
        let lower_after_root = quat_rotate_vec(root_delta_world, current_lower_dir);
        let desired_lower_dir = v3_normalise_or(v3_sub(sol.end, sol.middle), [0.0, 1.0, 0.0]);
        let middle_delta_world = ik::rotation_from_to(lower_after_root, desired_lower_dir);

        // Convert world-space deltas back to joint-local rotations
        // via the skeleton hierarchy. The root's parent world rotation
        // is identity when the root bone has no parent.
        let parent_world_rot = skeleton.bones[root]
            .parent
            .and_then(|p| world_rots.get(p).copied())
            .unwrap_or([0.0, 0.0, 0.0, 1.0]);
        let current_world_root_rot = world_rots[root];
        let new_world_root_rot =
            normalize4(quat_mul(root_delta_world, current_world_root_rot));
        let new_local_root_rot =
            normalize4(quat_mul(quat_conj(parent_world_rot), new_world_root_rot));

        let current_world_middle_rot = world_rots[middle];
        // Middle's world rotation after the root has applied its delta,
        // but before middle's own delta, is `root_delta_world · current_middle`.
        let middle_after_root = quat_mul(root_delta_world, current_world_middle_rot);
        let new_world_middle_rot = normalize4(quat_mul(middle_delta_world, middle_after_root));
        let new_local_middle_rot =
            normalize4(quat_mul(quat_conj(new_world_root_rot), new_world_middle_rot));

        self.joints[root].rotation = new_local_root_rot;
        self.joints[middle].rotation = new_local_middle_rot;
    }

    /// Aim a single bone's `local_forward` axis at a world-space
    /// target, writing the resulting local rotation into the pose.
    ///
    /// `local_forward` is the bone's "look direction" in its own
    /// local frame — e.g. `[0.0, 0.0, 1.0]` for an eye that faces +Z
    /// when un-rotated, or `[1.0, 0.0, 0.0]` for a head where the
    /// forward axis points down +X at rest. Pick whichever axis the
    /// rig authored as forward.
    ///
    /// This is a "shortest-arc" look-at: no up-vector constraint, so
    /// the bone may roll around the forward axis as the target moves.
    /// For an eye that's fine; for a head you usually want an
    /// additional up-constraint, which layers on top via
    /// [`JointTransform::apply_delta`].
    ///
    /// Silently no-ops if `bone` is out of range.
    pub fn look_at_bone(
        &mut self,
        skeleton: &Skeleton,
        bone: usize,
        target: [f32; 3],
        local_forward: [f32; 3],
    ) {
        let joint_count = self.joints.len().min(skeleton.bones.len());
        if bone >= joint_count {
            return;
        }
        let world_mats = self.world_matrices(skeleton);
        let world_rots = self.world_rotations(skeleton);

        let bone_pos = {
            let c = world_mats[bone].cols[3];
            [c[0], c[1], c[2]]
        };
        // Current world forward = bone's world rotation applied to local_forward.
        let current_forward_world = quat_rotate_vec(world_rots[bone], local_forward);
        let desired_forward_world = v3_sub(target, bone_pos);
        let delta_world = ik::rotation_from_to(current_forward_world, desired_forward_world);

        let parent_world_rot = skeleton.bones[bone]
            .parent
            .and_then(|p| world_rots.get(p).copied())
            .unwrap_or([0.0, 0.0, 0.0, 1.0]);
        let new_world_rot = normalize4(quat_mul(delta_world, world_rots[bone]));
        let new_local_rot =
            normalize4(quat_mul(quat_conj(parent_world_rot), new_world_rot));
        self.joints[bone].rotation = new_local_rot;
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

/// Build `SkinningData` by reading each joint's world transform
/// straight from `scene.compute_world_transforms()`, then multiplying
/// by the bone's inverse-bind matrix.
///
/// This is the preferred path for real glTF assets. [`Pose::skinning_data`]
/// walks `Bone::parent`, which `blinc_gltf::skin::parse_skin` only sets
/// when a joint's *direct* glTF parent is also a joint — rigs that
/// thread non-joint glue nodes between joints (Armature wrappers,
/// offset / pivot nodes, mesh root placeholders, etc.) lose those
/// intermediate transforms and the character renders at origin,
/// at wrong scale, or entirely off-camera.
///
/// Walking `compute_world_transforms` uses the full scene node graph
/// so those glue transforms are folded in correctly.
///
/// Typical usage:
///
/// ```ignore
/// animate_scene_nodes(&mut scene, &clip, t);
/// // Optional: morph weights (or call Pose::evaluate for both).
/// let morphs = animate_scene_morph_weights(&clip, t);
/// let skinning = scene_skinning_data(&scene, skin);
/// // Pass identity as the model matrix for skinned draws — skinning
/// // matrices already produce world-space positions.
/// ```
pub fn scene_skinning_data(scene: &GltfScene, skin: &GltfSkeleton) -> SkinningData {
    let world_per_node = scene.compute_world_transforms();
    let matrices: Vec<[f32; 16]> = skin
        .joint_nodes
        .iter()
        .zip(skin.skeleton.bones.iter())
        .map(|(&node_idx, bone)| {
            let w = world_per_node
                .get(node_idx)
                .copied()
                .unwrap_or(Mat4::IDENTITY);
            let ibm = mat4_from_array(bone.inverse_bind_matrix);
            flatten_mat4(w.mul(&ibm))
        })
        .collect();
    SkinningData {
        joint_matrices: matrices,
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
        // MorphWeights channels are intercepted by `Pose::evaluate`
        // and routed to the per-node morph-weights sink — they never
        // reach this helper. Same for type mismatches (e.g. a
        // translation channel carrying Vec4 values, which is invalid
        // glTF) — skip defensively rather than panic.
        _ => {}
    }
}

/// Sample a morph-weights channel and return the interpolated weight
/// vector at time `t`. The sampler's values are scalars laid out as
/// `times.len() * weight_count` contiguous floats; the returned slice
/// is exactly one weight-block wide (`weight_count` entries).
fn sample_morph_weights(sampler: &AnimationSampler, t: f32) -> Option<Vec<f32>> {
    match sample::sample(sampler, t)? {
        Sampled::Scalars(v) => Some(v),
        _ => None,
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

/// Hamilton product. `quat_mul(a, b)` applies `b` first, then `a`
/// (i.e. `(ab)v = a(bv)`). Used by the additive-blend path to
/// compose the local delta with the base rotation.
pub(crate) fn quat_mul(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    let [ax, ay, az, aw] = a;
    let [bx, by, bz, bw] = b;
    [
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ]
}

/// Conjugate (inverse for unit quaternions). Negates the vector part.
pub(crate) fn quat_conj(q: [f32; 4]) -> [f32; 4] {
    [-q[0], -q[1], -q[2], q[3]]
}

/// Rotate a vector by a unit quaternion via the classic
/// `v' = v + 2·qxyz × (qxyz × v + qw·v)` formula. Used by the IK
/// Pose wrappers to compute "what direction does a child bone face
/// after the parent rotates".
pub(crate) fn quat_rotate_vec(q: [f32; 4], v: [f32; 3]) -> [f32; 3] {
    let qxyz = [q[0], q[1], q[2]];
    let inner = [
        qxyz[1] * v[2] - qxyz[2] * v[1] + q[3] * v[0],
        qxyz[2] * v[0] - qxyz[0] * v[2] + q[3] * v[1],
        qxyz[0] * v[1] - qxyz[1] * v[0] + q[3] * v[2],
    ];
    let outer = [
        qxyz[1] * inner[2] - qxyz[2] * inner[1],
        qxyz[2] * inner[0] - qxyz[0] * inner[2],
        qxyz[0] * inner[1] - qxyz[1] * inner[0],
    ];
    [v[0] + 2.0 * outer[0], v[1] + 2.0 * outer[1], v[2] + 2.0 * outer[2]]
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
        let b = Pose {
            joints: vec![],
            morph_weights: std::collections::HashMap::new(),
        };
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

    // ── Additive-blend tests ─────────────────────────────────────────────

    #[test]
    fn joint_delta_rest_to_target_is_subtractive() {
        let rest = JointTransform {
            translation: [1.0, 2.0, 3.0],
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [2.0, 2.0, 2.0],
        };
        let target = JointTransform {
            translation: [4.0, 5.0, 7.0],
            rotation: quat_y(std::f32::consts::FRAC_PI_2),
            scale: [4.0, 4.0, 4.0],
        };
        let d = JointTransform::delta(&rest, &target);
        assert!(approx_eq(d.translation[0], 3.0, 1e-5));
        assert!(approx_eq(d.translation[1], 3.0, 1e-5));
        assert!(approx_eq(d.translation[2], 4.0, 1e-5));
        // 4 / 2 = 2.
        assert!(approx_eq(d.scale[0], 2.0, 1e-5));
        // Rotation delta: conj(identity) · quat_y(90°) = quat_y(90°).
        assert!(approx_eq(d.rotation[3], std::f32::consts::FRAC_1_SQRT_2, 1e-4));
    }

    #[test]
    fn apply_delta_zero_weight_is_identity_op() {
        let mut base = JointTransform {
            translation: [10.0, 20.0, 30.0],
            rotation: quat_y(0.3),
            scale: [1.5, 1.5, 1.5],
        };
        let saved = base;
        let delta = JointTransform {
            translation: [1.0, 1.0, 1.0],
            rotation: quat_y(1.0),
            scale: [2.0, 2.0, 2.0],
        };
        base.apply_delta(&delta, 0.0);
        for i in 0..3 {
            assert!(approx_eq(base.translation[i], saved.translation[i], 1e-5));
            assert!(approx_eq(base.scale[i], saved.scale[i], 1e-5));
        }
        for i in 0..4 {
            assert!(approx_eq(base.rotation[i], saved.rotation[i], 1e-5));
        }
    }

    #[test]
    fn apply_delta_full_weight_lands_on_target_when_base_is_rest() {
        // Canonical round-trip: extract (rest → target), then apply
        // that delta to `rest` at weight 1 — you should land on target.
        let rest = JointTransform {
            translation: [0.0, 0.0, 0.0],
            rotation: quat_y(0.0),
            scale: [1.0, 1.0, 1.0],
        };
        let target = JointTransform {
            translation: [3.0, 4.0, 5.0],
            rotation: quat_y(std::f32::consts::FRAC_PI_3),
            scale: [2.0, 2.0, 2.0],
        };
        let delta = JointTransform::delta(&rest, &target);
        let mut reconstructed = rest;
        reconstructed.apply_delta(&delta, 1.0);
        assert!(approx_eq(reconstructed.translation[0], target.translation[0], 1e-4));
        assert!(approx_eq(reconstructed.translation[1], target.translation[1], 1e-4));
        assert!(approx_eq(reconstructed.translation[2], target.translation[2], 1e-4));
        assert!(approx_eq(reconstructed.scale[0], target.scale[0], 1e-4));
        // Rotation: reconstructed should match target up to slight
        // accumulated floating-point drift in the slerp/mul path.
        assert!(approx_eq(reconstructed.rotation[3], target.rotation[3], 1e-4));
    }

    #[test]
    fn apply_delta_half_weight_is_halfway_on_translation() {
        let mut base = JointTransform::IDENTITY;
        let delta = JointTransform {
            translation: [10.0, 0.0, 0.0],
            rotation: quat_y(0.0),
            scale: [1.0, 1.0, 1.0],
        };
        base.apply_delta(&delta, 0.5);
        assert!(approx_eq(base.translation[0], 5.0, 1e-5));
    }

    // ── IK Pose wrapper tests ────────────────────────────────────────────

    fn three_bone_skel_chain() -> Skeleton {
        // Root at origin, middle at (2, 0, 0), end at (4, 0, 0) —
        // chain along +X with bone lengths 2 and 2. Identity IBMs
        // mean inverse_bind just places each joint at its rest pos.
        // The pose itself carries the rest translations.
        Skeleton {
            bones: vec![
                Bone {
                    name: "root".into(),
                    parent: None,
                    inverse_bind_matrix: identity16(),
                },
                Bone {
                    name: "middle".into(),
                    parent: Some(0),
                    inverse_bind_matrix: identity16(),
                },
                Bone {
                    name: "end".into(),
                    parent: Some(1),
                    inverse_bind_matrix: identity16(),
                },
            ],
        }
    }

    fn chain_rest_pose(skel: &Skeleton) -> Pose {
        // Joint 0 at origin with no local translation.
        // Joint 1 translated (2, 0, 0) in parent frame.
        // Joint 2 translated (2, 0, 0) in joint 1's frame.
        let mut pose = Pose::rest(skel);
        pose.joints[1].translation = [2.0, 0.0, 0.0];
        pose.joints[2].translation = [2.0, 0.0, 0.0];
        pose
    }

    #[test]
    fn pose_solve_two_bone_ik_reaches_target() {
        let skel = three_bone_skel_chain();
        let mut pose = chain_rest_pose(&skel);

        // Target at (2, 2, 0) — in reach (|target| = 2.83, max reach 4).
        pose.solve_two_bone_ik(&skel, 0, 1, 2, [2.0, 2.0, 0.0], [0.0, 5.0, 0.0]);

        // After IK, end joint's world position should be near target.
        let world = pose.world_matrices(&skel);
        let end_pos = [world[2].cols[3][0], world[2].cols[3][1], world[2].cols[3][2]];
        assert!(
            approx_eq(end_pos[0], 2.0, 1e-3),
            "end x = {} (expected ~2.0)",
            end_pos[0]
        );
        assert!(
            approx_eq(end_pos[1], 2.0, 1e-3),
            "end y = {} (expected ~2.0)",
            end_pos[1]
        );
    }

    #[test]
    fn pose_solve_two_bone_ik_out_of_range_is_noop_structure() {
        let skel = three_bone_skel_chain();
        let mut pose = chain_rest_pose(&skel);
        // Bad indices → no-op.
        pose.solve_two_bone_ik(&skel, 0, 1, 9, [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        let rot0 = pose.joints[0].rotation;
        assert!(approx_eq(rot0[3], 1.0, 1e-5)); // still identity
    }

    #[test]
    fn pose_look_at_bone_points_forward_at_target() {
        let skel = three_bone_skel_chain();
        let mut pose = chain_rest_pose(&skel);
        // Aim the root's +X axis at a point to the +Z side — should
        // produce a ~90° rotation around -Y.
        pose.look_at_bone(&skel, 0, [0.0, 0.0, 5.0], [1.0, 0.0, 0.0]);
        // After rotation, applying the joint's rotation to [1, 0, 0]
        // should produce a vector pointing toward +Z.
        let rotated = quat_rotate_vec(pose.joints[0].rotation, [1.0, 0.0, 0.0]);
        assert!(approx_eq(rotated[0], 0.0, 1e-3));
        assert!(approx_eq(rotated[2], 1.0, 1e-3));
    }

    #[test]
    fn pose_world_rotations_composes_parents() {
        let skel = three_bone_skel_chain();
        let mut pose = chain_rest_pose(&skel);
        // Rotate root by 90° around +Y. Middle's world rotation should
        // pick up that rotation even though its local is identity.
        pose.joints[0].rotation = quat_y(std::f32::consts::FRAC_PI_2);
        let world_rots = pose.world_rotations(&skel);
        // Both root and middle should carry the same world rotation
        // (middle is a direct child with identity local).
        for i in 0..4 {
            assert!(approx_eq(world_rots[0][i], world_rots[1][i], 1e-5));
        }
    }

    #[test]
    fn pose_evaluate_sinks_morph_weights_by_node() {
        use blinc_gltf::{
            AnimationChannel, AnimationSampler, AnimationTarget, Interpolation, KeyframeValues,
        };

        // Single-joint skin; clip drives a morph-weights channel on
        // a non-joint node. Tests that the pose routes the scalar
        // samples into the per-node sink rather than trying to cram
        // them into a joint-TRS slot.
        let skel = GltfSkeleton {
            name: None,
            skeleton: Skeleton {
                bones: vec![Bone {
                    name: "bone".into(),
                    parent: None,
                    inverse_bind_matrix: identity16(),
                }],
            },
            joint_nodes: vec![0],
        };
        // 3 morph targets; two keyframes — at t=0 all zero, at t=1
        // the three weights are [1, 0.5, 0]. Linear interp gives
        // [0.5, 0.25, 0] at t=0.5.
        let clip = blinc_gltf::GltfAnimation {
            name: None,
            channels: vec![AnimationChannel {
                target: AnimationTarget {
                    node: 42, // not a joint — goes to morph sink
                    property: AnimatedProperty::MorphWeights,
                },
                sampler: AnimationSampler {
                    times: vec![0.0, 1.0],
                    values: KeyframeValues::Scalars(vec![
                        0.0, 0.0, 0.0, // t=0
                        1.0, 0.5, 0.0, // t=1
                    ]),
                    interpolation: Interpolation::Linear,
                },
            }],
        };

        let mut pose = Pose::rest(&skel.skeleton);
        pose.evaluate(&clip, 0.5, &skel);

        let weights = pose.morph_weights_for_node(42).expect("weights present");
        assert_eq!(weights.len(), 3);
        assert!(approx_eq(weights[0], 0.5, 1e-4));
        assert!(approx_eq(weights[1], 0.25, 1e-4));
        assert!(approx_eq(weights[2], 0.0, 1e-4));

        // Non-MorphWeights node shouldn't have an entry.
        assert!(pose.morph_weights_for_node(0).is_none());
    }

    #[test]
    fn pose_apply_delta_layers_onto_animating_base() {
        // The motivating use-case: base pose is the locomotion
        // sample; delta is an "aim overlay"; after apply_delta the
        // translation reflects base + weight * delta.
        let skel = single_bone_skel();
        let mut base = Pose::rest(&skel);
        base.joints[0].translation = [0.0, 5.0, 0.0]; // pretend-walking, bone is 5 up

        let rest = Pose::rest(&skel);
        let mut full = Pose::rest(&skel);
        full.joints[0].translation = [0.0, 0.0, 2.0]; // aim pushes forward
        let aim_delta = Pose::delta(&rest, &full);

        base.apply_delta(&aim_delta, 0.75);
        // Translation ends at: base + 0.75 * (full − rest)
        //                    = [0, 5, 0] + 0.75 * [0, 0, 2]
        //                    = [0, 5, 1.5]
        assert!(approx_eq(base.joints[0].translation[0], 0.0, 1e-5));
        assert!(approx_eq(base.joints[0].translation[1], 5.0, 1e-5));
        assert!(approx_eq(base.joints[0].translation[2], 1.5, 1e-5));
    }
}
