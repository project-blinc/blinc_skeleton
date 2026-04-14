//! Rotation channel densification — fixes the slerp shortest-arc trap.
//!
//! glTF 2.0 stores rotations as quaternions and asks runtimes to slerp
//! between consecutive keyframes. Slerp always picks the **shortest
//! arc** between its endpoints, which is the right behavior for arcs
//! up to 180° but fails badly for anything larger:
//!
//! - A rotor that the author intended to spin 260° between two
//!   keyframes is encoded as a quaternion whose `w` component is
//!   negative. Slerp sees `dot(q[i], q[i+1]) < 0`, flips one of the
//!   signs to preserve shortest-arc, and interpolates the *other*
//!   100° — the wrong direction.
//! - A rotor authored to spin 360° between keys is encoded as
//!   `(0, 0, 0, ±1)`, indistinguishable from "no rotation". Slerp
//!   produces no motion at all.
//!
//! In assets like Sketchfab's `buster_drone`, the rotor channels are
//! authored at low keyframe density (~30 fps for a 5+ rev/s spin), so
//! every other segment is ambiguous. Looping the clip turns the
//! rotation into visible jitter as the slerp direction flips
//! frame-to-frame.
//!
//! [`densify_rotation_channels`] preprocesses an animation in-place,
//! inserting intermediate keyframes wherever a segment's *true* arc
//! (read from the delta quaternion's `w`, before any sign games)
//! exceeds [`MAX_SEG_RAD`]. After densification, every consecutive
//! pair represents a rotation < 60° — well inside the unambiguous
//! slerp range — so the runtime sampler reproduces the authored
//! motion regardless of speed or keyframe spacing.
//!
//! The pass is idempotent: a second call sees segments that are
//! already small enough and returns 0 insertions.
//!
//! ```ignore
//! let mut anim = blinc_gltf::load_glb(bytes)?.animations.remove(0);
//! let n_inserted = blinc_skeleton::densify_rotation_channels(&mut anim);
//! tracing::info!("rotation densification added {} keyframes", n_inserted);
//! ```

use blinc_gltf::{
    AnimatedProperty, GltfAnimation, Interpolation, KeyframeValues,
};

/// Maximum allowed slerp arc per segment after densification, in
/// radians. 60° is comfortably below the 180° slerp ambiguity limit
/// while keeping the inserted-keyframe count low — a 360° spin between
/// two original keys densifies to 6 sub-segments of 60° each.
pub const MAX_SEG_RAD: f32 = std::f32::consts::FRAC_PI_3;

/// Densify every `Linear`-interpolation rotation channel in `anim` so
/// that no consecutive pair of keyframes represents a slerp arc larger
/// than [`MAX_SEG_RAD`]. Returns the total number of keyframes
/// inserted across all channels.
///
/// `Step` and `CubicSpline` channels are passed through untouched —
/// step interpolation has no slerp ambiguity to fix, and cubic-spline
/// keyframes carry their own per-key tangents that would need a
/// different densification strategy.
///
/// Translation, scale, and morph-weight channels are also left alone;
/// only quaternion rotation suffers from the shortest-arc problem.
pub fn densify_rotation_channels(anim: &mut GltfAnimation) -> usize {
    let mut total = 0usize;
    for ch in anim.channels.iter_mut() {
        if ch.target.property != AnimatedProperty::Rotation {
            continue;
        }
        if ch.sampler.interpolation != Interpolation::Linear {
            continue;
        }
        let KeyframeValues::Vec4(values) = &mut ch.sampler.values else {
            continue;
        };
        total += densify_one_channel(&mut ch.sampler.times, values);
    }
    total
}

/// Densify a single rotation channel. Returns the number of keyframes
/// inserted. The `times` and `values` buffers are replaced with the
/// densified versions.
///
/// Exposed for direct use on raw channel buffers — callers driving
/// channels outside the `GltfAnimation` shape can still benefit from
/// the same fix.
pub fn densify_one_channel(times: &mut Vec<f32>, values: &mut Vec<[f32; 4]>) -> usize {
    if times.len() < 2 || values.len() < 2 {
        return 0;
    }
    debug_assert_eq!(times.len(), values.len());

    let mut new_times = Vec::with_capacity(times.len() * 2);
    let mut new_values = Vec::with_capacity(values.len() * 2);
    new_times.push(times[0]);
    new_values.push(values[0]);

    for i in 1..times.len() {
        let q0 = values[i - 1];
        let q1 = values[i];
        let t0 = times[i - 1];
        let t1 = times[i];
        let dt = t1 - t0;

        // Compute the delta rotation that takes us from q0 to q1, in
        // the *authored* sign — no shortest-arc rewrite.
        //
        //   delta = q1 * conj(q0)
        //
        // delta.w encodes the FULL arc: w = cos(angle/2) ranges over
        // [-1, 1] as angle ranges over [0, 2π]. A negative w means
        // the authored arc is in (π, 2π) — the long-way case that
        // standard slerp gets wrong.
        let delta = delta_quat(q0, q1);
        let w = delta[3].clamp(-1.0, 1.0);
        let angle = 2.0 * w.acos();

        if angle <= MAX_SEG_RAD {
            // Slerp between q0 and q1 is unambiguous (angle < 60°
            // means dot(q0, q1) ≥ 0.866 — far from the sign flip).
            new_times.push(t1);
            new_values.push(q1);
            continue;
        }

        // Extract the rotation axis. When sin(angle/2) ≈ 0 the axis
        // is undefined — that happens at angle = 0 (handled by the
        // early-return above) or angle = 2π (full revolution authored
        // as q1 = ±q0). The 2π case is genuinely ambiguous: we can't
        // tell "rotated full circle" from "did nothing". Skip
        // densification for it; the runtime will see zero rotation
        // either way, which is what slerp would produce anyhow.
        let s = (1.0 - w * w).sqrt();
        if s < 1e-4 {
            new_times.push(t1);
            new_values.push(q1);
            continue;
        }
        let axis = [delta[0] / s, delta[1] / s, delta[2] / s];

        // Number of sub-segments needed so each is ≤ MAX_SEG_RAD.
        // Always ≥ 2 because we entered this branch with angle > MAX_SEG_RAD.
        let n_subs = (angle / MAX_SEG_RAD).ceil() as usize;

        // Insert (n_subs - 1) intermediate keyframes. Each keyframe
        // is computed by composing q0 with a partial rotation of the
        // delta — `(axis, angle * u)` — so the runtime's slerp
        // between adjacent inserted keys reproduces the authored
        // axis-angle motion exactly.
        for k in 1..n_subs {
            let u = k as f32 / n_subs as f32;
            let sub_delta = quat_from_axis_angle(axis, angle * u);
            let q_mid = quat_mul(sub_delta, q0);
            new_times.push(t0 + dt * u);
            new_values.push(q_mid);
        }

        new_times.push(t1);
        new_values.push(q1);
    }

    let inserted = new_times.len() - times.len();
    *times = new_times;
    *values = new_values;
    inserted
}

// ─────────────────────────────────────────────────────────────────────────────
// Quaternion helpers (xyzw order, matching glTF and `blinc_skeleton::sample`)
// ─────────────────────────────────────────────────────────────────────────────

/// Hamilton product. `quat_mul(a, b)` applies `b` first, then `a`.
fn quat_mul(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
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
fn quat_conj(q: [f32; 4]) -> [f32; 4] {
    [-q[0], -q[1], -q[2], q[3]]
}

/// Rotation that takes `q0` to `q1`: `delta = q1 · conj(q0)`. Sign
/// is preserved verbatim — callers reading `delta.w` to recover the
/// authored arc length depend on this.
fn delta_quat(q0: [f32; 4], q1: [f32; 4]) -> [f32; 4] {
    quat_mul(q1, quat_conj(q0))
}

/// Build a unit quaternion from `axis` (assumed already normalized)
/// and an `angle` in radians. Angle range is unrestricted.
fn quat_from_axis_angle(axis: [f32; 3], angle: f32) -> [f32; 4] {
    let half = angle * 0.5;
    let s = half.sin();
    [axis[0] * s, axis[1] * s, axis[2] * s, half.cos()]
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build a rotation channel as a `(times, values)` pair
    /// representing `n_keys` keyframes of a constant-velocity Y-axis
    /// rotation at `omega` rad/sec, sampled `dt` apart.
    fn rotor_channel(omega: f32, dt: f32, n_keys: usize) -> (Vec<f32>, Vec<[f32; 4]>) {
        let mut times = Vec::with_capacity(n_keys);
        let mut values = Vec::with_capacity(n_keys);
        for i in 0..n_keys {
            let t = i as f32 * dt;
            let half = omega * t * 0.5;
            // Quaternion for rotation around +Y by `omega * t` rad.
            // Note we DO NOT canonicalize sign here — let the encoded
            // quaternion follow the natural cosine/sine, which means
            // `w` flips from positive to negative as the angle
            // crosses π.
            times.push(t);
            values.push([0.0, half.sin(), 0.0, half.cos()]);
        }
        (times, values)
    }

    #[test]
    fn densify_is_noop_on_dense_channel() {
        // 10°-per-frame channel: every segment is well under 60°.
        let (mut t, mut v) = rotor_channel(0.5, 1.0 / 30.0, 30); // 0.5 rad/s × 1/30s ≈ 0.95°
        let inserted = densify_one_channel(&mut t, &mut v);
        assert_eq!(inserted, 0, "small-arc channel should not be densified");
    }

    #[test]
    fn densify_subdivides_a_270_degree_segment() {
        // Two keys with 270° (= 4.71 rad) between them. Should split
        // into ceil(4.71 / (π/3)) = 5 sub-segments → 4 inserted keys.
        let q0 = [0.0, 0.0, 0.0, 1.0];
        let half = (270.0_f32.to_radians()) * 0.5;
        let q1 = [0.0, half.sin(), 0.0, half.cos()];
        let mut times = vec![0.0, 1.0];
        let mut values = vec![q0, q1];

        let inserted = densify_one_channel(&mut times, &mut values);
        assert_eq!(inserted, 4);
        assert_eq!(times.len(), 6);

        // Inserted timestamps should be evenly spaced.
        for (i, t) in times.iter().enumerate() {
            let expected = i as f32 / 5.0;
            assert!((t - expected).abs() < 1e-5, "time[{i}] = {t}, expected {expected}");
        }

        // Each consecutive pair must now have an unambiguous slerp arc
        // (angle < 60°).
        for w in values.windows(2) {
            let d = delta_quat(w[0], w[1]);
            let angle = 2.0 * d[3].clamp(-1.0, 1.0).acos();
            assert!(angle <= MAX_SEG_RAD + 1e-4, "post-densify segment angle = {angle} > MAX_SEG_RAD");
        }
    }

    #[test]
    fn densify_handles_negative_w_authored_long_way() {
        // 200° rotation. Quaternion has negative w. Slerp without
        // densification would interpret this as -160° (shortest arc).
        // Densified, it should be a sequence of small forward steps
        // covering the full 200°.
        let q0 = [0.0, 0.0, 0.0, 1.0];
        let half = (200.0_f32.to_radians()) * 0.5;
        let q1 = [0.0, half.sin(), 0.0, half.cos()];
        assert!(q1[3] < 0.0, "test setup: q1.w should be negative for >180° rotation");

        let mut times = vec![0.0, 1.0];
        let mut values = vec![q0, q1];

        let inserted = densify_one_channel(&mut times, &mut values);
        assert!(inserted >= 3, "200° should split into at least 4 sub-segments");

        // Walk the densified keys and verify the cumulative axis-angle
        // matches +200° around +Y, not -160°.
        let mut total = 0.0_f32;
        for w in values.windows(2) {
            let d = delta_quat(w[0], w[1]);
            let angle = 2.0 * d[3].clamp(-1.0, 1.0).acos();
            // After densification each delta should be a forward
            // rotation around +Y. Verify by checking the y component
            // sign matches the rotation direction.
            assert!(d[1] >= -1e-4, "densified deltas must rotate around +Y, got d.y = {}", d[1]);
            total += angle;
        }
        let expected = 200.0_f32.to_radians();
        assert!(
            (total - expected).abs() < 0.05,
            "cumulative densified angle = {total} rad, expected {expected} rad"
        );
    }

    #[test]
    fn densify_is_idempotent() {
        let q0 = [0.0, 0.0, 0.0, 1.0];
        let half = (240.0_f32.to_radians()) * 0.5;
        let q1 = [0.0, half.sin(), 0.0, half.cos()];
        let mut times = vec![0.0, 1.0];
        let mut values = vec![q0, q1];

        let first = densify_one_channel(&mut times, &mut values);
        let second = densify_one_channel(&mut times, &mut values);
        assert!(first > 0);
        assert_eq!(second, 0, "densify should be idempotent");
    }

    #[test]
    fn densify_skips_full_revolution_ambiguity() {
        // 360° = q1 represents identity (or -identity). We can't
        // distinguish "no rotation" from "full revolution" — the pass
        // should leave it alone rather than inserting bogus keys
        // around an undefined axis.
        let q0 = [0.0, 0.0, 0.0, 1.0];
        let q1 = [0.0, 0.0, 0.0, -1.0]; // exact -identity
        let mut times = vec![0.0, 1.0];
        let mut values = vec![q0, q1];

        let inserted = densify_one_channel(&mut times, &mut values);
        assert_eq!(inserted, 0, "ambiguous 360° rotation should be left alone");
    }

    #[test]
    fn densify_preserves_short_arcs_alongside_long_ones() {
        // Mix of small (10°) and large (200°) segments in one channel.
        // Only the large segment should be subdivided.
        let q0 = [0.0, 0.0, 0.0, 1.0];
        let h_small = (10.0_f32.to_radians()) * 0.5;
        let q1 = [0.0, h_small.sin(), 0.0, h_small.cos()];
        let h_big = h_small + (200.0_f32.to_radians()) * 0.5;
        let q2 = [0.0, h_big.sin(), 0.0, h_big.cos()];

        let mut times = vec![0.0, 0.1, 0.2];
        let mut values = vec![q0, q1, q2];
        let pre_len = times.len();
        let inserted = densify_one_channel(&mut times, &mut values);
        assert!(inserted >= 3);
        assert_eq!(times.len(), pre_len + inserted);

        // First original segment (q0 → q1) is small — shouldn't gain
        // keys before t = 0.1.
        let small_segment_keys = times.iter().filter(|&&t| t < 0.1 - 1e-6).count();
        assert_eq!(small_segment_keys, 1, "small segment should have only the original starting key");
    }
}
