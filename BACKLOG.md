# blinc_skeleton — Backlog

Outstanding work, ordered by impact on a typical skeletal-animation
pipeline.

---

## Blending + layering

- [x] **Weighted blend of two poses** — `Pose::blend` +
  `JointTransform::blend`, per-joint lerp / slerp, tested
  identity → 90°-Y at 50% weight.

- [x] **Blend tree / N-way blend** — `Pose::blend_many` folds
  `(weight, &Pose)` pairs with un-normalised weights via a running
  accumulator (convex combination maintained through the fold).

- [x] **Additive blending** — `Pose::delta(rest, target)` extracts
  the per-joint offset; `Pose::apply_delta(&delta, weight)` layers
  it on top of the base pose (translation additive, scale
  multiplicative via `lerp(1, δ_s, w)`, rotation via
  `slerp(I, δ_r, w)` local post-multiply). Round-trip tested:
  applying `delta(rest, target)` to `rest` at weight 1 lands on
  `target` within fp tolerance.

---

## Morph-target support

- [x] **Morph-weight channel sink on `Pose` (data flow).**
  `Pose::morph_weights: HashMap<usize, Vec<f32>>` keyed by scene node
  index (not bone index — morphs belong to meshes, which live on
  nodes). `Pose::evaluate` routes `AnimatedProperty::MorphWeights`
  channels here via the `sample_morph_weights` helper. Companion
  work in `blinc_core` (`MorphTarget` struct + `MeshData::morph_targets`
  field) and `blinc_gltf` (parse primitive morph targets into those
  fields) shipped together.

- [ ] **Morph-target GPU rendering (Phase 2).** Mesh shader samples
  `MeshData::morph_targets` per vertex and computes
  `final = base + Σ weights[t]·delta[t]`. Needs blinc_gpu uniform +
  per-vertex sampling of a morph-delta storage buffer (or a
  data-texture path for the WebGL2 fallback). Not started —
  visually-verifiable via Khronos's `AnimatedMorphCube` sample once
  the shader lands.

---

## Kinematics

- [x] **Two-bone IK — position solver.** `solve_two_bone(root,
  l_upper, l_lower, target, pole) -> TwoBoneSolution { middle, end,
  reached }` in `ik.rs`. Law-of-cosines analytic, handles
  over-extension (full-extend along target direction, `reached =
  false`) and under-extension (clamp to minimum bone-length
  difference). `rotation_from_to(from, to) -> quaternion` helper
  exposed for callers converting position deltas to joint rotations.

- [x] **Two-bone IK — orientation-aware Pose wrapper.**
  `Pose::solve_two_bone_ik(&skeleton, root, middle, end, target,
  pole)` writes new local rotations into the pose. Bone lengths
  extracted from the current pose's world positions (respects any
  translation animation in place). World→local conversion via
  `Pose::world_rotations` (added alongside — cheaper than going
  through `world_matrices` when only rotations are needed).

- [x] **FABRIK.** `solve_fabrik(joints, bone_lengths, target,
  iterations, tolerance) -> bool` in `ik.rs`. Iterative two-pass
  (forward / backward) position solver for N-segment chains;
  returns `true` on convergence, `false` on out-of-reach (chain
  fully extends toward target) or iteration-exhaust. Bone lengths
  preserved to float tolerance. Anchors the root — backward pass
  resets root each iteration.

- [x] **Look-at.** `Pose::look_at_bone(&skeleton, bone, target,
  local_forward)` writes the shortest-arc rotation that aims the
  bone's `local_forward` axis at the world-space target. No
  up-vector constraint by design (shortest-arc = minimum twist);
  layer an up-constraint via `JointTransform::apply_delta` if
  needed.

---

## Higher-level state

- [x] **Animation state machine.** `StateMachine` in `fsm.rs` with
  `ClipState` nodes (looping, speed-scalable, one-shot / held).
  `Transition` edges carry a `Condition` predicate + crossfade
  duration. Conditions: `Always`, `Bool`, `FloatGreaterThan`,
  `FloatLessThan`, `StateFinished`, `StateDuration`, plus `All` /
  `Any` combinators. Parameters held on the machine
  (`set_bool` / `set_float`). Crossfades use `Pose::blend` internally
  with a pre-allocated scratch pose. Blend-tree nodes (directional
  locomotion fed by a stick axis) left as a `ClipState` enum
  extension for a future PR.

- [ ] **Event markers on animation clips**
  - **Why:** "Play footstep sound at frame 12 of the run cycle."
  - **How:** `AnimationClip::markers: Vec<Marker { name, time }>`,
    fired from `Player::tick` as playback crosses them (same
    emission model as `blinc_lottie`). Needs `blinc_gltf` to also
    parse them — glTF itself doesn't standardize markers, so this
    probably reads from `extras`.

---

## Performance

- [ ] **Pose scratch arena**
  - Avoid per-frame `Vec<Mat4>` allocations in `world_matrices` and
    `skinning_matrices`. Pre-allocate scratch buffers on `Player`
    and reuse.

- [x] **Keyframe bracket: binary search.** `bracket()` in `sample.rs`
  now uses `slice::partition_point` for O(log n) per call on the
  guaranteed-monotonic `times` array. Covers the worst case with
  zero API change. For a 1000-keyframe channel that's ~10
  comparisons per sample instead of ~500 under the old linear scan.

- [ ] **Keyframe cursor (further optimization).** Amortized O(1) in
  the common case of monotonically-advancing clip time. Needs
  per-channel cursor state plumbed through either `Pose` or a
  dedicated `Cursor` handle — meaningful extra ergonomic surface
  compared to the zero-API-change binary search above. Worth
  doing if profiling ever shows the log-n bracket path as a hot
  spot (unlikely for typical ~100-keyframe channels).

- [ ] **SIMD quaternion slerp / matrix multiply**
  - After the cursor optimization lands, the next hot path is the
    quat / matrix math. `std::simd` or `wide` for the inner loops.

---

## Retargeting

- [ ] **Skeleton → skeleton clip retargeting**
  - **Why:** Reuse animations across characters with different
    proportions.
  - **How:** Bone-name-based mapping between skeletons + per-bone
    pose adaptation (usually translation scale + rotation
    reorientation). Large enough to belong in its own crate
    eventually; track here until scope forces a split.

---

## Non-goals

- **Physics-driven motion** (ragdolls, springs, cloth, hair). Those
  belong behind a physics integration — users drop in Rapier or any
  other solver.
- **Cinematic scene graphs** (Maya-style animation layers, time warp,
  curve editing). Authoring tooling, not runtime.
