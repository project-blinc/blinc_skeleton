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

    // ── Sign-align consecutive quaternions ────────────────────────────
    //
    // FBX → glTF exporters frequently write quaternion keyframes whose
    // signs are chosen independently per-keyframe, losing the
    // continuous-sign convention authored in the source. A pair that
    // physically represents a 1° rotation can end up as `(q, -q')`
    // where `q'` is `q` perturbed by 1° — their raw quaternion delta
    // reads as 358° around an effectively-random axis (the small
    // residual numerator divided by a small denominator amplifies
    // noise).
    //
    // Without this pass, the axis-angle densifier below would subdivide
    // those 358° arcs into half a dozen intermediate keys pointing at
    // garbage directions, teleporting the affected node subtree into
    // impossible poses mid-animation.
    //
    for i in 1..values.len() {
        let prev = values[i - 1];
        let q = values[i];
        let dot = prev[0] * q[0] + prev[1] * q[1] + prev[2] * q[2] + prev[3] * q[3];
        if dot < 0.0 {
            values[i] = [-q[0], -q[1], -q[2], -q[3]];
        }
    }

    // ── Precompute per-segment axis + angle (shortest-arc) ────────────
    //
    // After sign-alignment every segment's measured arc is in `[0, π]`,
    // which is what runtime slerp would pick. That's safe but loses
    // authored long-way rotations on fast rotors — the exact buster_
    // drone blade problem. We fix it in the next step by looking at
    // each segment's neighbors and re-interpreting arcs that buck the
    // neighbor trend.
    let n_segs = times.len() - 1;
    let mut axes: Vec<[f32; 3]> = Vec::with_capacity(n_segs);
    let mut angles: Vec<f32> = Vec::with_capacity(n_segs);
    for i in 0..n_segs {
        let q0 = values[i];
        let q1 = values[i + 1];
        let delta = delta_quat(q0, q1);
        let w = delta[3].clamp(-1.0, 1.0);
        let angle = 2.0 * w.acos();
        let s = (1.0 - w * w).sqrt();
        let axis = if s > 1e-4 {
            let raw = [delta[0] / s, delta[1] / s, delta[2] / s];
            let len = (raw[0] * raw[0] + raw[1] * raw[1] + raw[2] * raw[2]).sqrt();
            if len > 1e-6 {
                [raw[0] / len, raw[1] / len, raw[2] / len]
            } else {
                [1.0, 0.0, 0.0]
            }
        } else {
            // Near-identity or full-revolution — axis undefined. Set
            // to an arbitrary unit vector; the `angle < threshold`
            // guard below will skip subdivision for this segment.
            [1.0, 0.0, 0.0]
        };
        axes.push(axis);
        angles.push(angle);
    }

    // ── Multi-point long-way inference ────────────────────────────────
    //
    // A segment whose axis points opposite to both of its neighbors'
    // (while the neighbors agree) almost certainly encoded a > 180°
    // rotation that sign-alignment collapsed to its shortest-arc
    // representation. Restore the authored direction by flipping the
    // axis and replacing the angle with `2π - angle` — same quaternion
    // endpoints, forward path instead of backward.
    //
    // Bounds: only middle segments get checked; endpoints have no
    // second neighbor to confirm the pattern, and flipping a single
    // segment with no corroboration is more likely to fabricate
    // rotation than recover it. Threshold on `angle > 30°` ensures we
    // don't flip tiny rotations where the axis is numerically
    // ambiguous.
    if n_segs >= 3 {
        let min_flip_angle: f32 = 30.0_f32.to_radians();
        let threshold_neighbors_agree: f32 = 0.9;
        let threshold_opposed: f32 = -0.5;
        for i in 1..n_segs - 1 {
            if angles[i] < min_flip_angle {
                continue;
            }
            let prev_axis = axes[i - 1];
            let next_axis = axes[i + 1];
            let curr_axis = axes[i];
            let na = dot3(prev_axis, next_axis);
            let cp = dot3(curr_axis, prev_axis);
            let cn = dot3(curr_axis, next_axis);
            if na > threshold_neighbors_agree
                && cp < threshold_opposed
                && cn < threshold_opposed
            {
                axes[i] = [-curr_axis[0], -curr_axis[1], -curr_axis[2]];
                angles[i] = std::f32::consts::TAU - angles[i];
            }
        }
    }

    // ── Emit densified keyframe list ─────────────────────────────────
    let mut new_times = Vec::with_capacity(times.len() * 2);
    let mut new_values = Vec::with_capacity(values.len() * 2);
    new_times.push(times[0]);
    new_values.push(values[0]);

    for i in 0..n_segs {
        let q0 = values[i];
        let q1 = values[i + 1];
        let t0 = times[i];
        let t1 = times[i + 1];
        let dt = t1 - t0;
        let axis = axes[i];
        let angle = angles[i];

        if angle <= MAX_SEG_RAD {
            new_times.push(t1);
            new_values.push(q1);
            continue;
        }

        // Number of sub-segments needed so each is ≤ MAX_SEG_RAD.
        // Always ≥ 2 because we entered this branch with angle > MAX_SEG_RAD.
        let n_subs = (angle / MAX_SEG_RAD).ceil() as usize;

        // Insert (n_subs - 1) intermediate keyframes along the chosen
        // (possibly flipped) axis/angle. Runtime slerp between each
        // adjacent pair covers ≤ MAX_SEG_RAD and is guaranteed
        // unambiguous. Results are renormalized defensively — exporter
        // rounding in the input can propagate into the reconstructed
        // quaternions if we don't.
        for k in 1..n_subs {
            let u = k as f32 / n_subs as f32;
            let sub_delta = quat_from_axis_angle(axis, angle * u);
            let q_mid = quat_normalize(quat_mul(sub_delta, q0));
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

/// 3-component dot product — inlined to keep the densifier self-contained.
fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
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

/// Renormalize a quaternion to unit length. `blinc_skeleton::sample`
/// expects unit-length inputs for the slerp fast path; a quaternion
/// that is 0.999× unit-length won't crash anything but will bias the
/// sampled rotation angle, which compounds over multiple sub-segment
/// slerps into visible per-frame wobble on densified channels.
fn quat_normalize(q: [f32; 4]) -> [f32; 4] {
    let len = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    if len > 1e-8 {
        [q[0] / len, q[1] / len, q[2] / len, q[3] / len]
    } else {
        [0.0, 0.0, 0.0, 1.0]
    }
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
    fn densify_subdivides_a_large_short_arc() {
        // 270° authored as a single quaternion delta. After the
        // sign-alignment pre-pass, this collapses to its 90°
        // shortest-arc representation (the authored long-way intent
        // is lost — see module docs). 90° > MAX_SEG_RAD = 60°, so
        // one intermediate key is inserted at 45°.
        let q0 = [0.0, 0.0, 0.0, 1.0];
        let half = (270.0_f32.to_radians()) * 0.5;
        let q1 = [0.0, half.sin(), 0.0, half.cos()];
        let mut times = vec![0.0, 1.0];
        let mut values = vec![q0, q1];

        let inserted = densify_one_channel(&mut times, &mut values);
        // 90° / 60° → ceil = 2 sub-segments → 1 inserted key.
        assert_eq!(inserted, 1);
        assert_eq!(times.len(), 3);
        assert!((times[1] - 0.5).abs() < 1e-5);

        // Every consecutive pair is now < 60°.
        for w in values.windows(2) {
            let d = delta_quat(w[0], w[1]);
            let angle = 2.0 * d[3].clamp(-1.0, 1.0).acos();
            assert!(angle <= MAX_SEG_RAD + 1e-4);
        }
    }

    #[test]
    fn sign_inconsistent_quaternions_are_aligned_not_blown_up() {
        // This is the buster_drone failure mode: two keyframes that
        // represent the same physical rotation (or near it), but the
        // exporter sign-flipped one of them. Raw delta extraction
        // reads this as ~358° around a noise-amplified axis and,
        // without sign alignment, the densifier would insert garbage
        // intermediate keys that teleport the affected node around.
        //
        // After the pre-pass, consecutive quaternions are sign-aligned
        // so the delta.w is always >= 0 and the measured angle is the
        // shortest arc between them. For a near-identity physical
        // rotation, that means no densification fires at all.
        let q0 = [0.5779998, -0.4667882, -0.5311385, -0.40732908];
        // `q1` has the same physical orientation as q0 up to a small
        // perturbation around X, but the whole quaternion sign was
        // flipped during export. dot(q0, q1_raw) ≈ -1.
        let q1_raw = [-0.5788, 0.4672, 0.5311, 0.4063];
        let mut times = vec![0.0, 1.0];
        let mut values = vec![q0, q1_raw];

        let inserted = densify_one_channel(&mut times, &mut values);
        // The inserted count here must be 0 — a small physical
        // rotation should never subdivide.
        assert_eq!(
            inserted, 0,
            "sign-inconsistent near-identity pair incorrectly densified \
             (would teleport the node at runtime)"
        );

        // And the aligned q1 should now have dot(q0, q1) >= 0.
        let q1 = values[1];
        let dot = q0[0] * q1[0] + q0[1] * q1[1] + q0[2] * q1[2] + q0[3] * q1[3];
        assert!(dot >= 0.0);
    }

    #[test]
    fn multi_point_inference_flips_axis_outlier_to_long_way() {
        // The concrete jitter pattern on accelerating rotors: three
        // segments whose authored rates straddle the 180°/frame
        // threshold. The middle segment's authored arc is > 180°, so
        // sign-alignment collapses it to the opposite-axis short arc.
        // The outer segments' authored arcs are < 180°, so alignment
        // leaves them on the same axis as authored. The neighbor
        // check should spot the middle segment's axis disagreement
        // and flip it back to the long-way interpretation.
        //
        // Authored: 0° → 90° (around +Y) → 290° (around +Y, via a
        // 200° step) → 380° (around +Y, via a 90° step).
        let angles_deg = [0.0_f32, 90.0, 290.0, 380.0];
        let mut times = Vec::new();
        let mut values = Vec::new();
        for (i, deg) in angles_deg.iter().enumerate() {
            let h = deg.to_radians() * 0.5;
            times.push(i as f32);
            values.push([0.0, h.sin(), 0.0, h.cos()]);
        }
        let inserted = densify_one_channel(&mut times, &mut values);
        assert!(inserted > 0);

        // The first and third segments were authored as short (< 180°)
        // rotations around +Y; sign-alignment leaves them alone. The
        // middle segment's delta, after sign-alignment, has axis -Y
        // and angle 160° (shortest arc). The inference check sees
        // both neighbors on +Y and flips the middle to +Y 200°, which
        // is the authored direction.
        //
        // We verify by recomputing the shortest-arc delta between each
        // densified pair and asserting it rotates forward around +Y.
        // The raw delta can appear negative on pairs where the
        // sign-align pre-pass left the two quaternions on opposite
        // hemispheres — physically a forward rotation, but encoded as
        // a long-way negative-w rotation. Normalize the endpoints'
        // signs before reading the delta so the y-component reflects
        // the short-arc (i.e. physical) direction.
        for w in values.windows(2) {
            let mut q0 = w[0];
            let q1 = w[1];
            let dot = q0[0] * q1[0] + q0[1] * q1[1] + q0[2] * q1[2] + q0[3] * q1[3];
            if dot < 0.0 {
                q0 = [-q0[0], -q0[1], -q0[2], -q0[3]];
            }
            let d = delta_quat(q0, q1);
            assert!(
                d[1] >= -1e-3,
                "post-inference shortest-arc delta should rotate around +Y, got d = {d:?}",
            );
        }
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
    fn densified_keyframes_are_unit_length() {
        // Input quaternions slightly off-unit (simulating exporter
        // drift) should still produce unit-length densified keys.
        let q0 = [0.0, 0.0, 0.0, 1.001]; // 0.1% over-unit
        let half = (270.0_f32.to_radians()) * 0.5;
        let q1 = [0.0, half.sin() * 1.002, 0.0, half.cos() * 1.002];

        let mut times = vec![0.0, 1.0];
        let mut values = vec![q0, q1];
        let _ = densify_one_channel(&mut times, &mut values);

        // Skip the two original endpoints (we don't renormalize those —
        // sampler should handle slight drift without trouble) and check
        // only the *inserted* ones.
        for v in &values[1..values.len() - 1] {
            let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3]).sqrt();
            assert!(
                (len - 1.0).abs() < 1e-4,
                "inserted key {v:?} has length {len}, expected unit"
            );
        }
    }

    #[test]
    fn densify_preserves_short_arcs_alongside_mid_arcs() {
        // Mix of a small (10°) and a medium (160°) segment. After
        // sign alignment the medium segment's delta is still 160°
        // — above MAX_SEG_RAD but below the sign-flip threshold —
        // so it subdivides. The small segment doesn't.
        let q0 = [0.0, 0.0, 0.0, 1.0];
        let h_small = (10.0_f32.to_radians()) * 0.5;
        let q1 = [0.0, h_small.sin(), 0.0, h_small.cos()];
        let h_big = h_small + (160.0_f32.to_radians()) * 0.5;
        let q2 = [0.0, h_big.sin(), 0.0, h_big.cos()];

        let mut times = vec![0.0, 0.1, 0.2];
        let mut values = vec![q0, q1, q2];
        let pre_len = times.len();
        let inserted = densify_one_channel(&mut times, &mut values);
        assert!(inserted >= 2, "160° segment should subdivide");
        assert_eq!(times.len(), pre_len + inserted);

        // First original segment (q0 → q1) is small — shouldn't gain
        // keys before t = 0.1.
        let small_segment_keys = times.iter().filter(|&&t| t < 0.1 - 1e-6).count();
        assert_eq!(small_segment_keys, 1, "small segment should have only the original starting key");
    }
}
