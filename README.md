# blinc_skeleton

Runtime poser for [Blinc](https://github.com/project-blinc/Blinc).

Consumes `blinc_gltf` output (`GltfSkeleton`, `GltfAnimation`),
evaluates animation channels each frame, composes per-joint world
transforms, and emits GPU-ready skinning matrices that feed into
`blinc_core::draw::SkinningData`.

```rust
use blinc_gltf::load_glb;
use blinc_skeleton::Player;

let scene = load_glb(&bytes)?;
let skin = &scene.skeletons[0];
let clip = &scene.animations[0];

let mut player = Player::new(skin, clip);
player.set_looping(true);

// Each frame:
player.tick(dt);
let skinning = player.skinning_data();   // drop into MeshData::skin
```

For lower-level use (custom blending, procedural IK, multiple clips
simultaneously), bypass `Player` and drive [`Pose`](src/lib.rs) directly:

```rust
use blinc_skeleton::Pose;

let mut pose = Pose::rest(&skin.skeleton);
pose.evaluate(&clip, time_seconds, skin);
let skinning = pose.skinning_data(&skin.skeleton);
```

## What's implemented

- **Pose**: per-joint `JointTransform` (translation + quaternion +
  scale), world-matrix composition respecting parent links,
  `world * inverse_bind` skinning matrix computation, flatten to
  `SkinningData { joint_matrices: Vec<[f32; 16]> }`.
- **Sampler**: step / linear / cubic-Hermite interpolation for Vec3,
  Vec4 (quaternion slerp), and morph-weight scalar arrays. Cubic
  spline uses glTF's `(in_tangent, value, out_tangent)` layout.
- **Player**: play / pause / seek / loop / speed, per-tick pose
  re-evaluation, ready-to-render `SkinningData`.

See [BACKLOG.md](./BACKLOG.md) for blending, additive layering, IK,
state machines, and morph-weight channel support.

## License

Apache-2.0.
