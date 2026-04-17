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

- [ ] **Additive blending**
  - **Why:** Layer an "aiming" clip or "breathing" idle on top of a
    locomotion base without replacing it.
  - **How:** Store the *delta* of each joint from a rest reference;
    apply via `self.translation += delta * weight`,
    `self.rotation = delta_quat * self.rotation` (weighted).
    Needs a helper to compute the delta between two poses.

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

- [ ] **Two-bone IK**
  - **Why:** Foot placement, hand-reaches-target constraints,
    procedural head-look.
  - **How:** Analytic solver — given root + middle + end joint plus
    a target + pole, compute the two rotations that put the end at
    the target. Write back into the pose before world composition.

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
