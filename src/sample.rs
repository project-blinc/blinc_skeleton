//! Keyframe sampling + interpolation.
//!
//! Given a `(times, values, interpolation)` triple, evaluate the
//! channel at any time `t` and return the interpolated value. Handles
//! the three glTF interpolation modes (Step / Linear / CubicSpline)
//! and clamps to the first / last keyframe outside the clip range —
//! looping is a concern for the player, not the sampler.

use blinc_gltf::{AnimationSampler, Interpolation, KeyframeValues};

/// Result of sampling an animation channel.
#[derive(Debug, Clone)]
pub enum Sampled {
    Vec3([f32; 3]),
    Vec4([f32; 4]),
    /// Morph weights — length matches the source's weights-per-keyframe
    /// stride.
    Scalars(Vec<f32>),
}

/// Sample `sampler` at time `t`. Returns `None` when the channel is
/// empty (no keyframes).
pub fn sample(sampler: &AnimationSampler, t: f32) -> Option<Sampled> {
    if sampler.times.is_empty() {
        return None;
    }
    match (&sampler.values, sampler.interpolation) {
        (KeyframeValues::Vec3(v), Interpolation::Step) => Some(Sampled::Vec3(step3(sampler.times.as_slice(), v, t))),
        (KeyframeValues::Vec3(v), Interpolation::Linear) => Some(Sampled::Vec3(lerp3(sampler.times.as_slice(), v, t))),
        (KeyframeValues::Vec3(v), Interpolation::CubicSpline) => {
            Some(Sampled::Vec3(cubic3(sampler.times.as_slice(), v, t)))
        }

        (KeyframeValues::Vec4(v), Interpolation::Step) => Some(Sampled::Vec4(step4(sampler.times.as_slice(), v, t))),
        (KeyframeValues::Vec4(v), Interpolation::Linear) => {
            // Rotation channels use quaternions — slerp rather than
            // per-component lerp so the shortest-arc interpolation is
            // preserved. If the caller passes a non-quaternion Vec4
            // through here it still gets a normalized 4-vec, which is
            // close to what they'd want.
            Some(Sampled::Vec4(slerp4(sampler.times.as_slice(), v, t)))
        }
        (KeyframeValues::Vec4(v), Interpolation::CubicSpline) => {
            let q = cubic4(sampler.times.as_slice(), v, t);
            Some(Sampled::Vec4(normalize4(q)))
        }

        (KeyframeValues::Scalars(v), interp) => {
            // Scalars are used for morph weights — a contiguous block
            // of `weight_count` floats per keyframe. We can't know the
            // stride without the paired morph-target count, so we
            // return the full stride via `step_n` / `lerp_n` and let
            // the caller slice.
            let stride = v.len().checked_div(sampler.times.len()).unwrap_or(0);
            if stride == 0 {
                return Some(Sampled::Scalars(Vec::new()));
            }
            let out = match interp {
                Interpolation::Step => step_n(sampler.times.as_slice(), v, stride, t),
                Interpolation::Linear => lerp_n(sampler.times.as_slice(), v, stride, t),
                Interpolation::CubicSpline => cubic_n(sampler.times.as_slice(), v, stride, t),
            };
            Some(Sampled::Scalars(out))
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Pair lookup + bracket helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Find the keyframe pair `(k0, k1)` bracketing `t`, and the blend
/// factor `u ∈ [0, 1]`. Clamps before first / after last keyframe.
///
/// Uses binary search via [`slice::partition_point`] — O(log n) on
/// the keyframe count. `times` is guaranteed monotonic-increasing by
/// the glTF spec, so a partition boundary at the first `times[i] > t`
/// gives us `k0 = i − 1` directly. For a 1000-keyframe channel this
/// is ~10 comparisons per call; the old linear scan was ~500 on
/// average. Per-channel cursor caching (even faster in the common
/// case of monotonically-advancing clip time) remains on the
/// backlog — binary search covers the worst case with zero API
/// change.
fn bracket(times: &[f32], t: f32) -> (usize, usize, f32) {
    if t <= times[0] {
        return (0, 0, 0.0);
    }
    let last = times.len() - 1;
    if t >= times[last] {
        return (last, last, 0.0);
    }
    // Index of the first keyframe strictly after `t`. Guaranteed
    // within `(0, last]` because of the clamp checks above.
    let k1 = times.partition_point(|&x| x <= t);
    let k0 = k1 - 1;
    let u = (t - times[k0]) / (times[k1] - times[k0]);
    (k0, k1, u)
}

// ─────────────────────────────────────────────────────────────────────────────
// Step / Linear / Hermite (Vec3, Vec4, scalar-N)
// ─────────────────────────────────────────────────────────────────────────────

fn step3(times: &[f32], vals: &[[f32; 3]], t: f32) -> [f32; 3] {
    let (k0, _, _) = bracket(times, t);
    vals[k0]
}

fn step4(times: &[f32], vals: &[[f32; 4]], t: f32) -> [f32; 4] {
    let (k0, _, _) = bracket(times, t);
    vals[k0]
}

fn step_n(times: &[f32], vals: &[f32], stride: usize, t: f32) -> Vec<f32> {
    let (k0, _, _) = bracket(times, t);
    vals[k0 * stride..(k0 + 1) * stride].to_vec()
}

fn lerp3(times: &[f32], vals: &[[f32; 3]], t: f32) -> [f32; 3] {
    let (k0, k1, u) = bracket(times, t);
    if k0 == k1 {
        return vals[k0];
    }
    let a = vals[k0];
    let b = vals[k1];
    [
        a[0] + (b[0] - a[0]) * u,
        a[1] + (b[1] - a[1]) * u,
        a[2] + (b[2] - a[2]) * u,
    ]
}

fn slerp4(times: &[f32], vals: &[[f32; 4]], t: f32) -> [f32; 4] {
    let (k0, k1, u) = bracket(times, t);
    if k0 == k1 {
        return vals[k0];
    }
    quat_slerp(vals[k0], vals[k1], u)
}

fn lerp_n(times: &[f32], vals: &[f32], stride: usize, t: f32) -> Vec<f32> {
    let (k0, k1, u) = bracket(times, t);
    if k0 == k1 {
        return vals[k0 * stride..(k0 + 1) * stride].to_vec();
    }
    let a = &vals[k0 * stride..(k0 + 1) * stride];
    let b = &vals[k1 * stride..(k1 + 1) * stride];
    a.iter().zip(b).map(|(&x, &y)| x + (y - x) * u).collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Cubic spline — glTF stores each keyframe as (in_tangent, value, out_tangent)
// ─────────────────────────────────────────────────────────────────────────────
//
// For N times, the values array is 3 * N entries: the i-th keyframe's
// `in_tangent` at `3*i`, `value` at `3*i + 1`, `out_tangent` at `3*i + 2`.
// Tangents are scaled by the keyframe delta `Δt` during Hermite blend.

fn cubic3(times: &[f32], vals: &[[f32; 3]], t: f32) -> [f32; 3] {
    let (k0, k1, u) = bracket(times, t);
    if k0 == k1 {
        return vals[k0 * 3 + 1]; // value component only
    }
    let dt = times[k1] - times[k0];
    let (h00, h10, h01, h11) = hermite_basis(u);
    let p0 = vals[k0 * 3 + 1];
    let m0 = vals[k0 * 3 + 2];
    let p1 = vals[k1 * 3 + 1];
    let m1 = vals[k1 * 3];
    [
        h00 * p0[0] + h10 * dt * m0[0] + h01 * p1[0] + h11 * dt * m1[0],
        h00 * p0[1] + h10 * dt * m0[1] + h01 * p1[1] + h11 * dt * m1[1],
        h00 * p0[2] + h10 * dt * m0[2] + h01 * p1[2] + h11 * dt * m1[2],
    ]
}

fn cubic4(times: &[f32], vals: &[[f32; 4]], t: f32) -> [f32; 4] {
    let (k0, k1, u) = bracket(times, t);
    if k0 == k1 {
        return vals[k0 * 3 + 1];
    }
    let dt = times[k1] - times[k0];
    let (h00, h10, h01, h11) = hermite_basis(u);
    let p0 = vals[k0 * 3 + 1];
    let m0 = vals[k0 * 3 + 2];
    let p1 = vals[k1 * 3 + 1];
    let m1 = vals[k1 * 3];
    [
        h00 * p0[0] + h10 * dt * m0[0] + h01 * p1[0] + h11 * dt * m1[0],
        h00 * p0[1] + h10 * dt * m0[1] + h01 * p1[1] + h11 * dt * m1[1],
        h00 * p0[2] + h10 * dt * m0[2] + h01 * p1[2] + h11 * dt * m1[2],
        h00 * p0[3] + h10 * dt * m0[3] + h01 * p1[3] + h11 * dt * m1[3],
    ]
}

fn cubic_n(times: &[f32], vals: &[f32], stride: usize, t: f32) -> Vec<f32> {
    let (k0, k1, u) = bracket(times, t);
    let value_block = |k: usize| &vals[(k * 3 + 1) * stride..(k * 3 + 2) * stride];
    if k0 == k1 {
        return value_block(k0).to_vec();
    }
    let in_block = |k: usize| &vals[(k * 3) * stride..(k * 3 + 1) * stride];
    let out_block = |k: usize| &vals[(k * 3 + 2) * stride..(k * 3 + 3) * stride];
    let dt = times[k1] - times[k0];
    let (h00, h10, h01, h11) = hermite_basis(u);
    let p0 = value_block(k0);
    let m0 = out_block(k0);
    let p1 = value_block(k1);
    let m1 = in_block(k1);
    (0..stride)
        .map(|i| h00 * p0[i] + h10 * dt * m0[i] + h01 * p1[i] + h11 * dt * m1[i])
        .collect()
}

fn hermite_basis(u: f32) -> (f32, f32, f32, f32) {
    let u2 = u * u;
    let u3 = u2 * u;
    (
        2.0 * u3 - 3.0 * u2 + 1.0, // h00 — p0 weight
        u3 - 2.0 * u2 + u,         // h10 — m0 weight
        -2.0 * u3 + 3.0 * u2,      // h01 — p1 weight
        u3 - u2,                   // h11 — m1 weight
    )
}

// ─────────────────────────────────────────────────────────────────────────────
// Quaternion helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Shortest-arc slerp between two unit quaternions. Falls back to
/// normalized linear interpolation when the quaternions are very
/// close (dot > 0.9995) — standard practice to avoid `acos` instability.
pub fn quat_slerp(a: [f32; 4], b: [f32; 4], u: f32) -> [f32; 4] {
    let mut b = b;
    let mut dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
    if dot < 0.0 {
        // Flip `b` so the interpolation takes the shorter arc.
        dot = -dot;
        b = [-b[0], -b[1], -b[2], -b[3]];
    }
    if dot > 0.9995 {
        // Nearly-parallel → fall back to nlerp to avoid numerical
        // issues in `acos`.
        return normalize4([
            a[0] + (b[0] - a[0]) * u,
            a[1] + (b[1] - a[1]) * u,
            a[2] + (b[2] - a[2]) * u,
            a[3] + (b[3] - a[3]) * u,
        ]);
    }
    let theta_0 = dot.acos();
    let theta = theta_0 * u;
    let sin_theta = theta.sin();
    let sin_theta_0 = theta_0.sin();
    let s0 = ((1.0 - u) * theta_0).sin() / sin_theta_0;
    let s1 = sin_theta / sin_theta_0;
    [
        s0 * a[0] + s1 * b[0],
        s0 * a[1] + s1 * b[1],
        s0 * a[2] + s1 * b[2],
        s0 * a[3] + s1 * b[3],
    ]
}

pub fn normalize4(q: [f32; 4]) -> [f32; 4] {
    let len = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    if len > 1e-8 {
        [q[0] / len, q[1] / len, q[2] / len, q[3] / len]
    } else {
        [0.0, 0.0, 0.0, 1.0]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bracket_clamps_before_and_after() {
        let t = [0.0, 1.0, 2.0];
        assert_eq!(bracket(&t, -5.0), (0, 0, 0.0));
        assert_eq!(bracket(&t, 100.0), (2, 2, 0.0));
    }

    #[test]
    fn bracket_finds_pair_and_u() {
        let t = [0.0, 1.0, 2.0];
        let (k0, k1, u) = bracket(&t, 0.5);
        assert_eq!((k0, k1), (0, 1));
        assert!((u - 0.5).abs() < 1e-6);
        let (k0, k1, u) = bracket(&t, 1.5);
        assert_eq!((k0, k1), (1, 2));
        assert!((u - 0.5).abs() < 1e-6);
    }

    #[test]
    fn bracket_binary_search_finds_correct_pair_in_long_channel() {
        // 1024 keyframes at 0.01s intervals — exercises the
        // partition_point path end-to-end and verifies the O(log n)
        // search returns the same bracket as the old O(n) scan.
        let times: Vec<f32> = (0..1024).map(|i| i as f32 * 0.01).collect();
        let (k0, k1, u) = bracket(&times, 5.125);
        assert_eq!((k0, k1), (512, 513));
        assert!((u - 0.5).abs() < 1e-4);
        // Exact hit on a keyframe: partition_point's `<=` semantics
        // mean we end up with k0 = i - 1, k1 = i (blend factor 1.0).
        let (k0, k1, u) = bracket(&times, 7.00);
        assert_eq!(k1, k0 + 1);
        assert!(u >= 0.0 && u <= 1.0);
    }

    #[test]
    fn bracket_handles_exact_keyframe_times() {
        // t equals a non-endpoint keyframe exactly — partition_point
        // should produce a well-formed bracket with u ∈ [0, 1].
        let t = [0.0, 1.0, 2.0, 3.0];
        let (k0, k1, u) = bracket(&t, 1.0);
        assert!(k0 < k1);
        assert!((0.0..=1.0).contains(&u));
    }

    #[test]
    fn linear_vec3_midpoint() {
        let times = [0.0f32, 2.0];
        let vals = [[0.0, 0.0, 0.0], [10.0, 20.0, 30.0]];
        let r = lerp3(&times, &vals, 1.0);
        assert_eq!(r, [5.0, 10.0, 15.0]);
    }

    #[test]
    fn step_holds_previous_value() {
        let times = [0.0f32, 1.0, 2.0];
        let vals = [[0.0, 0.0, 0.0], [5.0, 5.0, 5.0], [10.0, 10.0, 10.0]];
        assert_eq!(step3(&times, &vals, 1.5), [5.0, 5.0, 5.0]);
    }

    #[test]
    fn quat_slerp_midpoint_is_90_deg_when_endpoints_are_0_and_180() {
        // Rotating around Y: identity → 180° rotation around Y.
        // Midpoint should be 90° around Y → (0, sin45, 0, cos45).
        let a = [0.0, 0.0, 0.0, 1.0];
        let b = [0.0, 1.0, 0.0, 0.0];
        let m = quat_slerp(a, b, 0.5);
        let target = (std::f32::consts::FRAC_PI_4).sin();
        assert!((m[1] - target).abs() < 1e-5, "{:?}", m);
        assert!((m[3] - target).abs() < 1e-5, "{:?}", m);
    }

    #[test]
    fn cubic_spline_matches_linear_with_zero_tangents() {
        // Tangents set to zero → result should match linear
        // interpolation of the values.
        let times = [0.0f32, 1.0];
        // Each keyframe: (in_tangent, value, out_tangent)
        let vals = [
            [0.0, 0.0, 0.0], // in  @ t=0
            [0.0, 0.0, 0.0], // val @ t=0
            [0.0, 0.0, 0.0], // out @ t=0
            [0.0, 0.0, 0.0], // in  @ t=1
            [1.0, 2.0, 3.0], // val @ t=1
            [0.0, 0.0, 0.0], // out @ t=1
        ];
        let r = cubic3(&times, &vals, 0.5);
        // Hermite with zero tangents @ u=0.5: h00=0.5, h01=0.5.
        assert!((r[0] - 0.5).abs() < 1e-5);
        assert!((r[1] - 1.0).abs() < 1e-5);
        assert!((r[2] - 1.5).abs() < 1e-5);
    }
}
