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

- [ ] **Morph-weight channel sink on `Pose`**
  - **Why:** Facial expressions, blend shapes. Already parsed by
    blinc_gltf; just needs a home to write to.
  - **How:** Add `morph_weights: Vec<f32>` on `Pose` (sized by the
    mesh's morph-target count). Wire the `AnimatedProperty::MorphWeights`
    match arm to write into it. Renderer-side: `MeshData` grows a
    `morph_weights` field, shader applies delta positions.

---

## Kinematics

- [x] **Two-bone IK — position solver.** `solve_two_bone(root,
  l_upper, l_lower, target, pole) -> TwoBoneSolution { middle, end,
  reached }` in `ik.rs`. Law-of-cosines analytic, handles
  over-extension (full-extend along target direction, `reached =
  false`) and under-extension (clamp to minimum bone-length
  difference). `rotation_from_to(from, to) -> quaternion` helper
  exposed for callers converting position deltas to joint rotations.
- [ ] **Two-bone IK — orientation-aware wrapper on `Pose`.** Follow-
  up that takes `(skeleton, root_bone, middle_bone, end_bone,
  target, pole)` and writes new local rotations into the pose,
  handling the world ↔ local conversion via the joint hierarchy.
  Depends on a per-pose world-rotation cache to avoid recomputing
  the ancestor chain.

- [ ] **FABRIK (Forward-And-Backward Reaching IK)**
  - **Why:** Multi-segment chains (spines, tentacles, tails).
  - **How:** Iterative forward pass (end → root, pulling each
    segment's length intact) then backward pass (root → end). 2-4
    iterations typically suffice.

- [ ] **Look-at constraint**
  - **Why:** Eye tracking, head follow targets.
  - **How:** Compute the rotation that aligns a bone's local axis
    with a world-space direction, apply as a constraint on top of
    the sampled pose.

---

## Higher-level state

- [ ] **Animation state machine**
  - **Why:** Game controllers want "idle → walk → run → attack" as a
    graph, not manual clip juggling.
  - **How:** Nodes are clips / blend trees; edges have transition
    predicates (`finished`, `duration`, `condition(f32)`) and a
    crossfade duration. Player becomes a `StateMachine` variant.

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

- [ ] **Keyframe cursor**
  - Linear scan in `bracket` is O(n) per sample. Cache the last-hit
    keyframe index per channel and bias the search from there.
    Usually the cursor advances by 1 keyframe per frame; this makes
    the sample path O(1) in the common case.

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
