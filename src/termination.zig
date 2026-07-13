//! Phase 9 learned termination support.
//!
//! A stable-answer rule is the selected termination interface: answer learning
//! is still expressed by the plastic readout, while this controller only asks
//! whether one *unique* answer has remained dominant long enough to commit.
//! It is deliberately independent of answer ID and has no RNG, preserving
//! DEC-004 reproducibility. A timeout is always available as a safety rail.

const std = @import("std");

pub const Outcome = enum { stable_answer, timeout };

pub const StableAnswer = struct {
    required_steps: u32,
    timeout_steps: u32,
    elapsed_steps: u32 = 0,
    consecutive_steps: u32 = 0,
    previous_answer: ?u8 = null,

    pub fn init(required_steps: u32, timeout_steps: u32) StableAnswer {
        std.debug.assert(required_steps > 0);
        std.debug.assert(timeout_steps >= required_steps);
        return .{ .required_steps = required_steps, .timeout_steps = timeout_steps };
    }

    /// Observe the unique dominant answer for one readout step. `null` means
    /// that the evidence was absent or tied, which intentionally breaks
    /// stability rather than allowing a low-ID tie to terminate an episode.
    pub fn observe(self: *StableAnswer, dominant: ?u8) ?Outcome {
        self.elapsed_steps += 1;
        if (dominant) |answer| {
            if (self.previous_answer != null and self.previous_answer.? == answer) {
                self.consecutive_steps += 1;
            } else {
                self.previous_answer = answer;
                self.consecutive_steps = 1;
            }
            if (self.consecutive_steps >= self.required_steps) return .stable_answer;
        } else {
            self.previous_answer = null;
            self.consecutive_steps = 0;
        }
        if (self.elapsed_steps >= self.timeout_steps) return .timeout;
        return null;
    }
};

/// The terminal reward is intentionally sparse: timely correct/incorrect
/// answers get the existing +/- reward; a timeout is a smaller negative value.
/// Callers pass this scalar directly to `Sim.applyReward`, so Phase 3's
/// three-factor eligibility mechanism remains the only learning rule.
pub fn reward(outcome: Outcome, correct: bool, timeout_reward: f32) f32 {
    return switch (outcome) {
        .stable_answer => if (correct) 1.0 else -1.0,
        .timeout => timeout_reward,
    };
}

test "termination: unique answer must remain stable before committing" {
    var tracker = StableAnswer.init(3, 6);
    try std.testing.expectEqual(@as(?Outcome, null), tracker.observe(2));
    try std.testing.expectEqual(@as(?Outcome, null), tracker.observe(2));
    try std.testing.expectEqual(@as(?Outcome, .stable_answer), tracker.observe(2));
}

test "termination: a tie resets stability and timeout has its own reward" {
    var tracker = StableAnswer.init(2, 4);
    try std.testing.expectEqual(@as(?Outcome, null), tracker.observe(1));
    try std.testing.expectEqual(@as(?Outcome, null), tracker.observe(null));
    try std.testing.expectEqual(@as(?Outcome, null), tracker.observe(1));
    try std.testing.expectEqual(@as(?Outcome, .timeout), tracker.observe(2));
    try std.testing.expectEqual(@as(f32, -0.2), reward(.timeout, true, -0.2));
    try std.testing.expectEqual(@as(f32, -1.0), reward(.stable_answer, false, -0.2));
}
