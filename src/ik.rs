//! Analytic two-bone inverse kinematics.
//!
//! Given a three-joint chain (root → middle → end) with fixed bone
//! lengths, plus a target position and a pole vector that controls
//! which side the chain bends toward, compute the world-space
//! positions that place the end on (or as close as possible to) the
//! target.
//!
//! This is a *position* solver — callers that need joint orientations
//! can compose them from the returned positions via
//! [`rotation_from_to`] (e.g. `rotation_from_to(rest_upper_dir,
//! solved_upper_dir)` is the rotation to apply to the root joint).
//! A future revision may layer an orientation-aware wrapper directly
//! onto [`Pose`].
//!
//! # Math
//!
//! Standard law-of-cosines construction. With `l₁ = |root→middle|`,
//! `l₂ = |middle→end|`, and `d = |target − root|`:
//!
//! - If `d > l₁ + l₂` the chain cannot reach — we extend it fully
//!   along `target − root` and report `reached = false`.
//! - If `d < |l₁ − l₂|` one bone would need to fold inside the
//!   other. Clamp `d` to `|l₁ − l₂|` in that case and also report
//!   `reached = false`; the resulting pose is the closest the chain
//!   can get while respecting its bone lengths.
//! - Otherwise, the interior angle at the root is
//!   `α = acos((l₁² + d² − l₂²) / (2·l₁·d))`. The middle joint sits
//!   a distance `l₁·cos(α)` along `(target − root)` from the root,
//!   and `l₁·sin(α)` perpendicular to that direction, toward the
//!   pole.
//!
//! # Pole vector
//!
//! The pole is a world-space hint: whichever side of the
//! `root`-to-`target` line the pole lies on is the side the elbow /
//! knee bends toward. Put the pole in front of a character's knees
//! to keep them from inverting when the target moves behind; put it
//! to the left of an elbow to keep the arm bent outward. If the
//! pole lies exactly on the `root`-to-`target` line (or coincident
//! with either joint), the bending direction is ambiguous and the
//! solver falls back to the world Y-axis.

/// Result of a two-bone solve.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TwoBoneSolution {
    /// World-space position of the middle joint after the solve.
    pub middle: [f32; 3],
    /// World-space position of the end joint after the solve. Equals
    /// `target` when `reached == true`; equals the closest point the
    /// chain can reach otherwise.
    pub end: [f32; 3],
    /// `true` if the target was reachable (within `|l₁ − l₂| ≤ d ≤
    /// l₁ + l₂`) and the end actually lands on `target`. `false`
    /// indicates the solver clamped — the chain is fully extended or
    /// fully folded toward the target direction.
    pub reached: bool,
}

/// Solve a two-bone chain.
///
/// `root` is the world-space position of the first joint. `upper_length`
/// is the distance between the root and middle joints; `lower_length`
/// is the distance between the middle and end joints (both extracted
/// once at setup from the skeleton's rest pose — they don't change
/// during playback).
///
/// `target` is where the end should go. `pole` is a world-space hint
/// that sets the bend direction (see the module docs).
///
/// Degenerate inputs (zero bone length, coincident joints, pole on
/// the solve axis) don't panic — they fall back to sensible defaults
/// and the chain extends along `target − root`.
pub fn solve_two_bone(
    root: [f32; 3],
    upper_length: f32,
    lower_length: f32,
    target: [f32; 3],
    pole: [f32; 3],
) -> TwoBoneSolution {
    let rt = v3_sub(target, root);
    let d = v3_length(rt);
    let dir = v3_normalise_or(rt, [0.0, 1.0, 0.0]);

    // Reach check — full-extension or collapse fall back to the
    // dir-aligned line through root at the clamped distance.
    let max_reach = upper_length + lower_length;
    let min_reach = (upper_length - lower_length).abs();
    let reached = d <= max_reach + 1e-5 && d >= min_reach - 1e-5;

    // Use the clamped distance for angle math so the acos doesn't
    // blow up on out-of-reach targets.
    let d_clamped = d.clamp(min_reach.max(1e-5), max_reach.max(1e-5));

    // Interior angle at the root (law of cosines).
    let cos_alpha = ((upper_length * upper_length + d_clamped * d_clamped
        - lower_length * lower_length)
        / (2.0 * upper_length * d_clamped))
        .clamp(-1.0, 1.0);
    let sin_alpha = (1.0 - cos_alpha * cos_alpha).max(0.0).sqrt();

    // Bending plane: normal is `dir × (pole - root)`; perpendicular
    // within the plane is `normal × dir`. Falls back to world Y when
    // the pole is collinear with `dir`, and world X when Y is too.
    let pole_off = v3_sub(pole, root);
    let normal = v3_cross(dir, pole_off);
    let perp = if v3_length(normal) < 1e-6 {
        // Pole collinear — pick a fallback perpendicular. Try Y, then X.
        let try_y = v3_cross(dir, [0.0, 1.0, 0.0]);
        if v3_length(try_y) >= 1e-6 {
            v3_cross(try_y, dir)
        } else {
            v3_cross([1.0, 0.0, 0.0], dir)
        }
    } else {
        v3_cross(normal, dir)
    };
    let perp = v3_normalise_or(perp, [0.0, 1.0, 0.0]);

    // Middle sits at (cos α) along dir + (sin α) along perp.
    let along = upper_length * cos_alpha;
    let across = upper_length * sin_alpha;
    let middle = [
        root[0] + dir[0] * along + perp[0] * across,
        root[1] + dir[1] * along + perp[1] * across,
        root[2] + dir[2] * along + perp[2] * across,
    ];

    // End lands on `target` when reachable, else at max (or min)
    // extension along `dir`.
    let end = if reached {
        target
    } else {
        [
            root[0] + dir[0] * d_clamped,
            root[1] + dir[1] * d_clamped,
            root[2] + dir[2] * d_clamped,
        ]
    };

    TwoBoneSolution {
        middle,
        end,
        reached,
    }
}

/// Quaternion that rotates unit vector `from` onto unit vector `to`.
/// Exposed so IK callers can convert "upper bone direction before"
/// + "upper bone direction after" into a root-joint rotation delta
/// without pulling a separate math crate.
///
/// Handles the two degenerate cases:
/// - `from ≈ to` → identity
/// - `from ≈ -to` → a 180° rotation around any axis perpendicular to
///   `from` (we pick one deterministically)
pub fn rotation_from_to(from: [f32; 3], to: [f32; 3]) -> [f32; 4] {
    let f = v3_normalise_or(from, [1.0, 0.0, 0.0]);
    let t = v3_normalise_or(to, [1.0, 0.0, 0.0]);
    let d = v3_dot(f, t).clamp(-1.0, 1.0);
    if d > 1.0 - 1e-6 {
        return [0.0, 0.0, 0.0, 1.0];
    }
    if d < -1.0 + 1e-6 {
        // 180° flip — build an orthogonal axis.
        let axis = v3_cross(f, [1.0, 0.0, 0.0]);
        let axis = if v3_length(axis) < 1e-6 {
            v3_cross(f, [0.0, 1.0, 0.0])
        } else {
            axis
        };
        let axis = v3_normalise_or(axis, [0.0, 1.0, 0.0]);
        return [axis[0], axis[1], axis[2], 0.0];
    }
    let axis = v3_cross(f, t);
    let s = ((1.0 + d) * 2.0).sqrt();
    let inv_s = 1.0 / s;
    let q = [axis[0] * inv_s, axis[1] * inv_s, axis[2] * inv_s, 0.5 * s];
    // Guaranteed unit (up to fp error), but renormalise for safety.
    let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    if n < 1e-6 {
        [0.0, 0.0, 0.0, 1.0]
    } else {
        [q[0] / n, q[1] / n, q[2] / n, q[3] / n]
    }
}

// ── Vec3 helpers — tiny and duplicated rather than pulling a dep ────────

fn v3_sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn v3_dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn v3_cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn v3_length(v: [f32; 3]) -> f32 {
    v3_dot(v, v).sqrt()
}

fn v3_normalise_or(v: [f32; 3], fallback: [f32; 3]) -> [f32; 3] {
    let len = v3_length(v);
    if len < 1e-6 {
        fallback
    } else {
        [v[0] / len, v[1] / len, v[2] / len]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_v3(a: [f32; 3], b: [f32; 3], eps: f32) -> bool {
        (a[0] - b[0]).abs() < eps && (a[1] - b[1]).abs() < eps && (a[2] - b[2]).abs() < eps
    }

    #[test]
    fn solve_reaches_target_when_in_range() {
        // Root at origin, 2+2 chain. Target at (2, 2, 0) is within
        // reach — chain bends with elbow above the line.
        let sol = solve_two_bone([0.0, 0.0, 0.0], 2.0, 2.0, [2.0, 2.0, 0.0], [0.0, 5.0, 0.0]);
        assert!(sol.reached);
        assert!(approx_v3(sol.end, [2.0, 2.0, 0.0], 1e-4));
        // Middle should be somewhere in the bending plane.
        assert!((sol.middle[2]).abs() < 1e-4);
    }

    #[test]
    fn solve_extends_fully_when_out_of_reach() {
        // Chain max reach = 4; target at (10, 0, 0) is way out. Chain
        // should extend fully along +X.
        let sol = solve_two_bone([0.0, 0.0, 0.0], 2.0, 2.0, [10.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        assert!(!sol.reached);
        assert!(approx_v3(sol.middle, [2.0, 0.0, 0.0], 1e-4));
        assert!(approx_v3(sol.end, [4.0, 0.0, 0.0], 1e-4));
    }

    #[test]
    fn solve_straight_line_when_target_equals_full_extension() {
        // Target exactly at max reach → straight chain, reached == true.
        let sol = solve_two_bone([0.0, 0.0, 0.0], 2.0, 2.0, [4.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        assert!(sol.reached);
        assert!(approx_v3(sol.end, [4.0, 0.0, 0.0], 1e-4));
        assert!(approx_v3(sol.middle, [2.0, 0.0, 0.0], 1e-4));
    }

    #[test]
    fn pole_flips_bend_direction() {
        // Same target, opposite pole → middle flips across the
        // root-target line.
        let above = solve_two_bone([0.0, 0.0, 0.0], 2.0, 2.0, [2.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        let below = solve_two_bone([0.0, 0.0, 0.0], 2.0, 2.0, [2.0, 0.0, 0.0], [0.0, -1.0, 0.0]);
        // y-components of middle should be mirrored.
        assert!((above.middle[1] + below.middle[1]).abs() < 1e-4);
        assert!((above.middle[1] - below.middle[1]).abs() > 1e-3);
    }

    #[test]
    fn solve_bone_lengths_preserved() {
        // Solved middle and end must sit at the right distances from
        // root and each other (sanity check for the geometry).
        let sol = solve_two_bone([0.0, 0.0, 0.0], 3.0, 1.5, [2.5, 1.0, 0.5], [0.0, 1.0, 0.0]);
        assert!(sol.reached);
        let upper = v3_length(v3_sub(sol.middle, [0.0, 0.0, 0.0]));
        let lower = v3_length(v3_sub(sol.end, sol.middle));
        assert!((upper - 3.0).abs() < 1e-4);
        assert!((lower - 1.5).abs() < 1e-4);
    }

    #[test]
    fn rotation_from_to_identity_for_same_vector() {
        let q = rotation_from_to([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
        assert!(approx_v3([q[0], q[1], q[2]], [0.0, 0.0, 0.0], 1e-5));
        assert!((q[3] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn rotation_from_to_quat_rotates_vector_onto_target() {
        // Build rotation from +X to +Y; apply it to +X; should hit +Y.
        let q = rotation_from_to([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        // Rotate [1, 0, 0] by q using the standard formula
        // v' = v + 2·cross(qxyz, cross(qxyz, v) + qw·v).
        let v = [1.0, 0.0, 0.0];
        let qxyz = [q[0], q[1], q[2]];
        let t = v3_cross(qxyz, v);
        let t = [t[0] * 2.0, t[1] * 2.0, t[2] * 2.0];
        let qw_v = [q[3] * v[0], q[3] * v[1], q[3] * v[2]];
        let inner = [t[0] / 2.0 + qw_v[0], t[1] / 2.0 + qw_v[1], t[2] / 2.0 + qw_v[2]];
        let outer = v3_cross(qxyz, inner);
        let rotated = [v[0] + outer[0] * 2.0, v[1] + outer[1] * 2.0, v[2] + outer[2] * 2.0];
        assert!(approx_v3(rotated, [0.0, 1.0, 0.0], 1e-4));
    }

    #[test]
    fn rotation_from_to_opposite_vectors_is_180() {
        // +X → -X should give a 180° flip; w component = 0.
        let q = rotation_from_to([1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]);
        assert!(q[3].abs() < 1e-5);
        // Axis length should be 1.
        let axis_len = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2]).sqrt();
        assert!((axis_len - 1.0).abs() < 1e-5);
    }
}
