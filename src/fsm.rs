//! Finite-state machine for skeletal animation.
//!
//! Nodes are animation clips; edges carry a [`Condition`] predicate
//! and a crossfade duration. Each tick advances the current state's
//! clip time, checks outgoing transitions, starts a crossfade when
//! any condition fires, and evaluates a blended pose between the
//! source and target for the duration of the fade.
//!
//! # Not to be confused with `blinc_core::fsm`
//!
//! Blinc ships a second state-machine implementation in
//! [`blinc_core::fsm`] — used for widget interaction states (button
//! hover/press, checkbox toggle, etc.). That machine is event-driven
//! (`send(event_id)`), instantaneous (no transition duration), and
//! uses user-supplied `FnMut()` callbacks for entry / exit / action
//! hooks.
//!
//! This machine is **time-driven** (`tick(dt)`), conditions are
//! **re-polled each tick** against named parameters rather than
//! dispatched on discrete events, and transitions **blend over
//! time** via the [`Pose::blend`](crate::Pose::blend) path. Reach
//! for the core version on UI state; reach for this one on
//! character animation.
//!
//! # Structure
//!
//! 1. Build a [`StateMachine`] with a list of [`ClipState`]s and an
//!    initial state index.
//! 2. Add [`Transition`]s between states with [`StateMachine::add_transition`].
//! 3. Each frame: set any game-state parameters (`set_bool`,
//!    `set_float`), call [`StateMachine::tick(dt)`], then read the
//!    blended pose via [`StateMachine::pose`] or fetch GPU skinning
//!    matrices via [`StateMachine::skinning_matrices`].
//!
//! # Example
//!
//! ```ignore
//! use blinc_skeleton::fsm::{ClipState, Condition, StateMachine, Transition};
//!
//! let idle = ClipState::new("idle", &idle_clip).looping(true);
//! let walk = ClipState::new("walk", &walk_clip).looping(true);
//! let run = ClipState::new("run", &run_clip).looping(true);
//!
//! let mut fsm = StateMachine::new(&skin, 0, vec![idle, walk, run]);
//!
//! fsm.add_transition(Transition::new(0, 1, Condition::Bool("moving".into(), true), 0.25));
//! fsm.add_transition(Transition::new(1, 0, Condition::Bool("moving".into(), false), 0.25));
//! fsm.add_transition(Transition::new(1, 2, Condition::FloatGreaterThan("speed".into(), 2.0), 0.3));
//! fsm.add_transition(Transition::new(2, 1, Condition::FloatLessThan("speed".into(), 2.0), 0.3));
//!
//! // Each frame:
//! fsm.set_bool("moving", is_moving);
//! fsm.set_float("speed", stick_magnitude);
//! fsm.tick(dt);
//! let skinning = fsm.skinning_matrices();
//! ```
//!
//! # Limitations
//!
//! - Only clip nodes are supported. Blend-tree nodes (directional
//!   locomotion blends fed by a stick axis) would slot into the
//!   `ClipState` enum as an additional variant; not implemented yet.
//! - While a crossfade is active, newly-fired transitions are
//!   ignored. When the fade completes and the target becomes the
//!   current state, outgoing transitions are evaluated normally from
//!   that point. Re-triggering interrupts (pose-preserving) would
//!   require blending from the in-progress pose into the new target,
//!   which this first cut doesn't do.
//! - All states must reference clips compatible with the same skin.
//!   One skin per `StateMachine` by design.

use std::collections::HashMap;

use blinc_core::Mat4;
use blinc_core::draw::SkinningData;
use blinc_gltf::{GltfAnimation, GltfSkeleton};

use crate::Pose;

/// Opaque handle to a state within a [`StateMachine`]. Returned by
/// `ClipState` registration order — the index into the `states` vec
/// passed to [`StateMachine::new`].
pub type StateIndex = usize;

/// A single animation clip playing at its own rate, with its own
/// looping behaviour. Nodes of the state machine.
pub struct ClipState<'a> {
    pub name: String,
    pub clip: &'a GltfAnimation,
    /// Wrap `time_in_state` at the clip's duration. `false` = one-shot
    /// (the clip holds its last frame once it ends, letting
    /// [`Condition::StateFinished`] fire).
    pub looping: bool,
    /// Playback-rate multiplier for this state's clip. `1.0` plays at
    /// authored speed; negative values play in reverse.
    pub speed: f32,
    /// Cached clip duration (largest keyframe time across all
    /// channels). Computed once at construction.
    duration: f32,
}

impl<'a> ClipState<'a> {
    /// Construct a non-looping clip state at 1.0× playback.
    pub fn new(name: impl Into<String>, clip: &'a GltfAnimation) -> Self {
        Self {
            name: name.into(),
            clip,
            looping: false,
            speed: 1.0,
            duration: clip_duration(clip),
        }
    }

    pub fn looping(mut self, looping: bool) -> Self {
        self.looping = looping;
        self
    }

    pub fn speed(mut self, speed: f32) -> Self {
        self.speed = speed;
        self
    }

    pub fn duration(&self) -> f32 {
        self.duration
    }
}

/// Predicate evaluated against the state machine's parameters and
/// the current state's playback cursor. Fires the associated
/// transition when `true`.
#[derive(Clone, Debug)]
pub enum Condition {
    /// Always true. Unconditional transition — fires the moment the
    /// source state becomes current (useful as a fallback or
    /// immediately-chained intro clip).
    Always,
    /// `parameters.bool(name) == expected`. Missing parameters read
    /// as `false`.
    Bool(String, bool),
    /// `parameters.float(name) > threshold`. Missing parameters read
    /// as `0.0`.
    FloatGreaterThan(String, f32),
    /// `parameters.float(name) < threshold`. Missing parameters read
    /// as `0.0`.
    FloatLessThan(String, f32),
    /// The current state's clip has reached (or passed) its end. For
    /// looping states this fires briefly each loop iteration just
    /// before the wrap; for one-shot states it latches true.
    StateFinished,
    /// Current state has been active for at least `seconds`. Counts
    /// wall-clock time since the state last became current, not
    /// clip time.
    StateDuration(f32),
    /// All sub-conditions must be true.
    All(Vec<Condition>),
    /// Any sub-condition true is enough.
    Any(Vec<Condition>),
}

impl Condition {
    fn evaluate(&self, ctx: &ConditionContext) -> bool {
        match self {
            Condition::Always => true,
            Condition::Bool(name, expected) => {
                ctx.parameters.get_bool(name).unwrap_or(false) == *expected
            }
            Condition::FloatGreaterThan(name, threshold) => {
                ctx.parameters.get_float(name).unwrap_or(0.0) > *threshold
            }
            Condition::FloatLessThan(name, threshold) => {
                ctx.parameters.get_float(name).unwrap_or(0.0) < *threshold
            }
            Condition::StateFinished => ctx.state_finished,
            Condition::StateDuration(seconds) => ctx.time_in_state >= *seconds,
            Condition::All(subs) => subs.iter().all(|c| c.evaluate(ctx)),
            Condition::Any(subs) => subs.iter().any(|c| c.evaluate(ctx)),
        }
    }
}

struct ConditionContext<'p> {
    parameters: &'p Parameters,
    time_in_state: f32,
    state_finished: bool,
}

/// A graph edge — `from` → `to` when `condition` fires, crossfading
/// over `duration` seconds. Built with [`Transition::new`] and
/// registered on the state machine via [`StateMachine::add_transition`].
pub struct Transition {
    pub from: StateIndex,
    pub to: StateIndex,
    pub condition: Condition,
    /// Crossfade length in seconds. `0.0` is a hard snap.
    pub duration: f32,
}

impl Transition {
    pub fn new(from: StateIndex, to: StateIndex, condition: Condition, duration: f32) -> Self {
        Self {
            from,
            to,
            condition,
            duration: duration.max(0.0),
        }
    }
}

/// Named runtime parameters that [`Condition`]s read to decide when
/// transitions fire. Populated each frame by the game code
/// (`fsm.set_bool("grounded", true)`, `fsm.set_float("speed", 3.2)`).
#[derive(Default)]
pub struct Parameters {
    bools: HashMap<String, bool>,
    floats: HashMap<String, f32>,
}

impl Parameters {
    pub fn set_bool(&mut self, name: impl Into<String>, value: bool) {
        self.bools.insert(name.into(), value);
    }

    pub fn set_float(&mut self, name: impl Into<String>, value: f32) {
        self.floats.insert(name.into(), value);
    }

    pub fn get_bool(&self, name: &str) -> Option<bool> {
        self.bools.get(name).copied()
    }

    pub fn get_float(&self, name: &str) -> Option<f32> {
        self.floats.get(name).copied()
    }
}

/// Internal bookkeeping for a crossfade in progress.
struct ActiveTransition {
    target: StateIndex,
    /// How long the target has been fading in. Reaches `duration` at
    /// completion.
    elapsed: f32,
    /// Total crossfade length in seconds.
    duration: f32,
    /// Playback cursor in the target clip's own time. Needs to advance
    /// alongside `elapsed` so the target state is already at the
    /// right clip-time when the fade completes.
    target_time: f32,
}

/// A finite state machine for skeletal animation.
pub struct StateMachine<'a> {
    skin: &'a GltfSkeleton,
    states: Vec<ClipState<'a>>,
    transitions: Vec<Transition>,
    parameters: Parameters,

    current: StateIndex,
    time_in_state: f32,
    active_transition: Option<ActiveTransition>,

    pose: Pose,
    scratch: Pose,
}

impl<'a> StateMachine<'a> {
    /// Build a state machine starting in state `initial` over the
    /// given clip nodes. All clips are assumed to target the same
    /// skin.
    pub fn new(skin: &'a GltfSkeleton, initial: StateIndex, states: Vec<ClipState<'a>>) -> Self {
        assert!(
            initial < states.len(),
            "initial state index {initial} out of range (states.len() = {})",
            states.len()
        );
        let pose = Pose::rest(&skin.skeleton);
        let scratch = Pose::rest(&skin.skeleton);
        let mut fsm = Self {
            skin,
            states,
            transitions: Vec::new(),
            parameters: Parameters::default(),
            current: initial,
            time_in_state: 0.0,
            active_transition: None,
            pose,
            scratch,
        };
        fsm.evaluate_pose();
        fsm
    }

    /// Register an outgoing transition. Transitions are evaluated in
    /// registration order; the first one whose condition fires on a
    /// given tick wins.
    pub fn add_transition(&mut self, transition: Transition) -> &mut Self {
        self.transitions.push(transition);
        self
    }

    pub fn set_bool(&mut self, name: impl Into<String>, value: bool) {
        self.parameters.set_bool(name, value);
    }

    pub fn set_float(&mut self, name: impl Into<String>, value: f32) {
        self.parameters.set_float(name, value);
    }

    pub fn get_bool(&self, name: &str) -> Option<bool> {
        self.parameters.get_bool(name)
    }

    pub fn get_float(&self, name: &str) -> Option<f32> {
        self.parameters.get_float(name)
    }

    /// Currently-active state index (the *source* of any in-progress
    /// crossfade, not the target).
    pub fn current(&self) -> StateIndex {
        self.current
    }

    pub fn current_name(&self) -> &str {
        &self.states[self.current].name
    }

    pub fn is_transitioning(&self) -> bool {
        self.active_transition.is_some()
    }

    /// Fraction of the current crossfade completed, in `[0, 1]`. Zero
    /// when no transition is in progress.
    pub fn transition_progress(&self) -> f32 {
        self.active_transition
            .as_ref()
            .map(|t| {
                if t.duration > 0.0 {
                    (t.elapsed / t.duration).clamp(0.0, 1.0)
                } else {
                    1.0
                }
            })
            .unwrap_or(0.0)
    }

    /// Target of the in-progress crossfade (if any).
    pub fn transition_target(&self) -> Option<StateIndex> {
        self.active_transition.as_ref().map(|t| t.target)
    }

    /// Force-set the current state, clearing any in-progress
    /// crossfade. The new state starts at `time_in_state = 0.0`.
    pub fn force_state(&mut self, state: StateIndex) {
        assert!(state < self.states.len());
        self.current = state;
        self.time_in_state = 0.0;
        self.active_transition = None;
        self.evaluate_pose();
    }

    /// Advance time by `dt` seconds, evaluate pending transitions,
    /// and recompute the output pose.
    pub fn tick(&mut self, dt: f32) {
        if dt < 0.0 {
            return;
        }

        // Advance time in the current state. Clip time wraps for
        // looping states; one-shots latch at the end.
        self.time_in_state += dt;

        // Progress any in-flight crossfade first — once it completes
        // we rebase onto the target state before evaluating new
        // transitions, so a new edge can fire on the very same tick
        // the crossfade ends.
        if let Some(mut t) = self.active_transition.take() {
            t.elapsed += dt;
            t.target_time += dt * self.states[t.target].speed;
            if t.elapsed >= t.duration {
                // Fade complete — rebase onto the target.
                self.current = t.target;
                self.time_in_state = wrap_or_clamp(
                    t.target_time,
                    self.states[self.current].duration,
                    self.states[self.current].looping,
                );
                // No active transition; fall through to outgoing-edge
                // check below.
            } else {
                // Fade still running — push back into the slot.
                self.active_transition = Some(t);
            }
        }

        // Don't fire new transitions while an existing one is still
        // crossfading. Simpler to reason about; pose-preserving
        // interruption is a later feature.
        if self.active_transition.is_none() {
            self.check_outgoing_transitions();
        }

        self.evaluate_pose();
    }

    fn check_outgoing_transitions(&mut self) {
        let current_state = &self.states[self.current];
        let state_finished = if current_state.looping {
            // Looping clips are "finished" only in the wrap moment —
            // effectively never true for FSM purposes.
            false
        } else {
            self.time_in_state >= current_state.duration
        };
        let ctx = ConditionContext {
            parameters: &self.parameters,
            time_in_state: self.time_in_state,
            state_finished,
        };

        for transition in &self.transitions {
            if transition.from != self.current {
                continue;
            }
            if transition.condition.evaluate(&ctx) {
                // Fire this transition.
                if transition.duration <= 0.0 {
                    // Hard snap.
                    self.current = transition.to;
                    self.time_in_state = 0.0;
                } else {
                    self.active_transition = Some(ActiveTransition {
                        target: transition.to,
                        elapsed: 0.0,
                        duration: transition.duration,
                        target_time: 0.0,
                    });
                }
                return;
            }
        }
    }

    fn evaluate_pose(&mut self) {
        // Source pose: sample current state's clip at its own time,
        // wrapped / clamped per looping flag.
        let src_state = &self.states[self.current];
        let src_time = wrap_or_clamp(self.time_in_state, src_state.duration, src_state.looping);
        self.pose = Pose::rest(&self.skin.skeleton);
        self.pose.evaluate(src_state.clip, src_time, self.skin);

        if let Some(t) = &self.active_transition {
            // Target pose: sample in the scratch buffer, then blend
            // into `self.pose` at the current fade fraction.
            let tgt_state = &self.states[t.target];
            let tgt_time = wrap_or_clamp(t.target_time, tgt_state.duration, tgt_state.looping);
            self.scratch = Pose::rest(&self.skin.skeleton);
            self.scratch.evaluate(tgt_state.clip, tgt_time, self.skin);
            let weight = if t.duration > 0.0 {
                (t.elapsed / t.duration).clamp(0.0, 1.0)
            } else {
                1.0
            };
            self.pose.blend(&self.scratch, weight);
        }
    }

    /// Current blended pose. Reflects the last `tick` (or the initial
    /// `force_state` / construction).
    pub fn pose(&self) -> &Pose {
        &self.pose
    }

    /// GPU-ready skinning matrices for the current pose. Direct drop-in
    /// for `MeshData::skin = Some(...)`.
    pub fn skinning_matrices(&self) -> Vec<Mat4> {
        self.pose.skinning_matrices(&self.skin.skeleton)
    }

    pub fn skinning_data(&self) -> SkinningData {
        self.pose.skinning_data(&self.skin.skeleton)
    }
}

/// Wrap a time value at `duration` if the state is looping, or clamp
/// it to `duration` if one-shot. Handles `duration == 0` (empty clip)
/// by returning `0`.
fn wrap_or_clamp(t: f32, duration: f32, looping: bool) -> f32 {
    if duration <= 0.0 {
        return 0.0;
    }
    if looping {
        t.rem_euclid(duration)
    } else {
        t.clamp(0.0, duration)
    }
}

fn clip_duration(anim: &GltfAnimation) -> f32 {
    anim.channels
        .iter()
        .filter_map(|ch| ch.sampler.times.last().copied())
        .fold(0.0f32, f32::max)
}

#[cfg(test)]
mod tests {
    use super::*;
    use blinc_core::draw::{Bone, Skeleton};
    use blinc_gltf::{
        AnimatedProperty, AnimationChannel, AnimationSampler, AnimationTarget, Interpolation,
        KeyframeValues,
    };

    fn ident16() -> [f32; 16] {
        [
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ]
    }

    fn one_bone_skin() -> GltfSkeleton {
        GltfSkeleton {
            name: None,
            skeleton: Skeleton {
                bones: vec![Bone {
                    name: "bone".into(),
                    parent: None,
                    inverse_bind_matrix: ident16(),
                }],
            },
            joint_nodes: vec![0],
        }
    }

    /// Translation clip on node 0. Two keyframes: at `t = 0` value
    /// `[0, 0, 0]`, at `t = duration` value `[x_at_end, 0, 0]`.
    fn translation_clip(duration: f32, x_at_end: f32) -> GltfAnimation {
        GltfAnimation {
            name: None,
            channels: vec![AnimationChannel {
                target: AnimationTarget {
                    node: 0,
                    property: AnimatedProperty::Translation,
                },
                sampler: AnimationSampler {
                    times: vec![0.0, duration],
                    values: KeyframeValues::Vec3(vec![[0.0, 0.0, 0.0], [x_at_end, 0.0, 0.0]]),
                    interpolation: Interpolation::Linear,
                },
            }],
        }
    }

    #[test]
    fn starts_in_initial_state() {
        let skin = one_bone_skin();
        let clip_a = translation_clip(1.0, 1.0);
        let clip_b = translation_clip(1.0, 5.0);
        let states = vec![
            ClipState::new("a", &clip_a).looping(true),
            ClipState::new("b", &clip_b).looping(true),
        ];
        let fsm = StateMachine::new(&skin, 0, states);
        assert_eq!(fsm.current(), 0);
        assert_eq!(fsm.current_name(), "a");
        assert!(!fsm.is_transitioning());
    }

    #[test]
    fn tick_advances_time_and_evaluates_pose() {
        let skin = one_bone_skin();
        let clip = translation_clip(2.0, 4.0);
        let mut fsm = StateMachine::new(&skin, 0, vec![ClipState::new("a", &clip).looping(true)]);
        fsm.tick(0.5);
        // At t = 0.5 of 2s total, x should be ~ 0.25 * 4.0 = 1.0.
        let x = fsm.pose().joints[0].translation[0];
        assert!((x - 1.0).abs() < 1e-4, "x = {}", x);
    }

    #[test]
    fn bool_transition_fires() {
        let skin = one_bone_skin();
        let clip_a = translation_clip(10.0, 0.0);
        let clip_b = translation_clip(10.0, 5.0);
        let states = vec![
            ClipState::new("a", &clip_a).looping(true),
            ClipState::new("b", &clip_b).looping(true),
        ];
        let mut fsm = StateMachine::new(&skin, 0, states);
        fsm.add_transition(Transition::new(
            0,
            1,
            Condition::Bool("go".into(), true),
            0.0, // hard snap
        ));

        // Before flipping the parameter — stays in state 0.
        fsm.tick(0.1);
        assert_eq!(fsm.current(), 0);

        fsm.set_bool("go", true);
        fsm.tick(0.0);
        assert_eq!(fsm.current(), 1);
        assert!(!fsm.is_transitioning());
    }

    #[test]
    fn crossfade_blends_poses_at_midpoint() {
        let skin = one_bone_skin();
        let clip_a = translation_clip(10.0, 0.0); // stays at x = 0
        let clip_b = translation_clip(10.0, 10.0); // grows
        // Doctor clip_b: t=0 → x=0, so scrubbing back to t=0 gives x=0. To
        // make the target pose visible at t=0 of the target clip, seed its
        // start value away from zero.
        let clip_b = GltfAnimation {
            name: None,
            channels: vec![AnimationChannel {
                target: AnimationTarget {
                    node: 0,
                    property: AnimatedProperty::Translation,
                },
                sampler: AnimationSampler {
                    times: vec![0.0, 10.0],
                    values: KeyframeValues::Vec3(vec![[10.0, 0.0, 0.0], [10.0, 0.0, 0.0]]),
                    interpolation: Interpolation::Linear,
                },
            }],
        };
        let states = vec![
            ClipState::new("a", &clip_a).looping(true),
            ClipState::new("b", &clip_b).looping(true),
        ];
        let mut fsm = StateMachine::new(&skin, 0, states);
        fsm.add_transition(Transition::new(
            0,
            1,
            Condition::Always,
            1.0, // 1-second crossfade
        ));

        fsm.tick(0.0);
        // Transition starts immediately via Condition::Always.
        assert!(fsm.is_transitioning());
        // Advance half the crossfade.
        fsm.tick(0.5);
        assert!(fsm.is_transitioning());
        let x = fsm.pose().joints[0].translation[0];
        // At 50% blend between x=0 (from clip_a) and x=10 (from
        // clip_b), expect ~5.
        assert!((x - 5.0).abs() < 1e-3, "x = {}", x);
        // Complete the crossfade.
        fsm.tick(0.6);
        assert!(!fsm.is_transitioning());
        assert_eq!(fsm.current(), 1);
    }

    #[test]
    fn state_finished_fires_on_one_shot() {
        let skin = one_bone_skin();
        let intro = translation_clip(1.0, 0.0);
        let idle = translation_clip(10.0, 0.0);
        let states = vec![
            ClipState::new("intro", &intro),
            ClipState::new("idle", &idle).looping(true),
        ];
        let mut fsm = StateMachine::new(&skin, 0, states);
        fsm.add_transition(Transition::new(
            0,
            1,
            Condition::StateFinished,
            0.0,
        ));

        fsm.tick(0.5);
        assert_eq!(fsm.current(), 0);
        fsm.tick(0.6); // intro clip ends at 1.0
        assert_eq!(fsm.current(), 1);
    }

    #[test]
    fn all_any_composite_conditions() {
        let skin = one_bone_skin();
        let a = translation_clip(10.0, 0.0);
        let b = translation_clip(10.0, 0.0);
        let states = vec![
            ClipState::new("a", &a).looping(true),
            ClipState::new("b", &b).looping(true),
        ];
        let mut fsm = StateMachine::new(&skin, 0, states);
        // Transition only when grounded AND speed > 0.5.
        fsm.add_transition(Transition::new(
            0,
            1,
            Condition::All(vec![
                Condition::Bool("grounded".into(), true),
                Condition::FloatGreaterThan("speed".into(), 0.5),
            ]),
            0.0,
        ));

        fsm.set_bool("grounded", true);
        fsm.set_float("speed", 0.2);
        fsm.tick(0.0);
        assert_eq!(fsm.current(), 0);

        fsm.set_float("speed", 1.0);
        fsm.tick(0.0);
        assert_eq!(fsm.current(), 1);
    }

    #[test]
    fn force_state_clears_transition() {
        let skin = one_bone_skin();
        let a = translation_clip(10.0, 0.0);
        let b = translation_clip(10.0, 0.0);
        let states = vec![
            ClipState::new("a", &a).looping(true),
            ClipState::new("b", &b).looping(true),
        ];
        let mut fsm = StateMachine::new(&skin, 0, states);
        fsm.add_transition(Transition::new(0, 1, Condition::Always, 2.0));
        fsm.tick(0.1);
        assert!(fsm.is_transitioning());

        fsm.force_state(0);
        assert!(!fsm.is_transitioning());
        assert_eq!(fsm.current(), 0);
    }

    #[test]
    fn looping_clip_wraps_time() {
        let skin = one_bone_skin();
        // 1-second looping clip — after 2.5 ticks, effective sample
        // time should wrap to 0.5.
        let clip = translation_clip(1.0, 10.0);
        let mut fsm = StateMachine::new(&skin, 0, vec![ClipState::new("a", &clip).looping(true)]);
        fsm.tick(2.5);
        // Sampling at t=0.5 of a [0→10] linear lerp = 5.0.
        let x = fsm.pose().joints[0].translation[0];
        assert!((x - 5.0).abs() < 1e-3, "x = {}", x);
    }
}
