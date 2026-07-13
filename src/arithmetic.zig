//! Phase 8 symbolic arithmetic task support.
//!
//! The encoder deliberately separates the left and right operand populations.
//! A numeral is still presented serially as a symbol, but its position is part
//! of the representation; otherwise `a - b` and `b - a` are indistinguishable
//! to a readout after the sequence has ended.  Answer populations are small,
//! disjoint action assemblies, one for each non-negative answer in the bounded
//! range.  This module only describes deterministic task structure and current
//! injection; network construction and reward learning remain in net/sim.

const std = @import("std");
const cfg = @import("config.zig");

pub const Operation = enum { add, subtract };
pub const Phase = enum { start, lhs, operator, rhs, end };

pub const Example = struct {
    lhs: u8,
    operation: Operation,
    rhs: u8,

    pub fn result(self: Example) u8 {
        return switch (self.operation) {
            .add => self.lhs + self.rhs,
            .subtract => self.lhs - self.rhs,
        };
    }
};

pub const Group = struct {
    lo: u32,
    hi: u32,

    pub fn contains(self: Group, id: u32) bool {
        return id >= self.lo and id < self.hi;
    }

    pub fn count(self: Group) u32 {
        return self.hi - self.lo;
    }
};

/// A bounded symbolic layout.  Its group ordering is fixed, so no task-related
/// random draw can perturb the reservoir graph (DEC-004).
pub const Layout = struct {
    group_size: u32,
    max_operand: u8,

    const control_groups: u32 = 4; // START, END, +, -

    pub fn symbolGroupCount(self: Layout) u32 {
        return control_groups + 2 * (@as(u32, self.max_operand) + 1);
    }

    pub fn actionCount(self: Layout) u32 {
        return 2 * @as(u32, self.max_operand) + 1;
    }

    fn groupAt(self: Layout, index: u32) Group {
        return .{ .lo = index * self.group_size, .hi = (index + 1) * self.group_size };
    }

    pub fn startGroup(self: Layout) Group {
        return self.groupAt(0);
    }

    pub fn endGroup(self: Layout) Group {
        return self.groupAt(1);
    }

    pub fn operatorGroup(self: Layout, operation: Operation) Group {
        return self.groupAt(switch (operation) {
            .add => 2,
            .subtract => 3,
        });
    }

    /// Position-bound numeral encoding. Left and right sequences have separate
    /// one-hot populations, preserving operand order for subtraction.
    pub fn operandGroup(self: Layout, side: enum { lhs, rhs }, value: u8) Group {
        std.debug.assert(value <= self.max_operand);
        const base: u32 = control_groups + switch (side) {
            .lhs => 0,
            .rhs => @as(u32, self.max_operand) + 1,
        };
        return self.groupAt(base + value);
    }

    pub fn actionGroup(self: Layout, answer: u8) Group {
        std.debug.assert(answer < self.actionCount());
        return self.groupAt(self.symbolGroupCount() + answer);
    }

    pub fn isSymbol(self: Layout, id: u32) bool {
        return id < self.symbolGroupCount() * self.group_size;
    }

    pub fn isAction(self: Layout, id: u32) bool {
        const lo = self.symbolGroupCount() * self.group_size;
        return id >= lo and id < lo + self.actionCount() * self.group_size;
    }

    pub fn fillPhase(self: Layout, example: Example, phase: Phase, current: f32, out: []f32) void {
        @memset(out, 0);
        const group = switch (phase) {
            .start => self.startGroup(),
            .lhs => self.operandGroup(.lhs, example.lhs),
            .operator => self.operatorGroup(example.operation),
            .rhs => self.operandGroup(.rhs, example.rhs),
            .end => self.endGroup(),
        };
        for (group.lo..group.hi) |i| out[i] = current;
    }

    /// Neutral, fixed-duration answer clock. It drives every answer population
    /// equally, so it cannot encode a particular answer; learned readout input
    /// from the preceding symbol sequence breaks the symmetry.
    pub fn fillAnswerProbe(self: Layout, current: f32, out: []f32) void {
        @memset(out, 0);
        var answer: u8 = 0;
        while (answer < self.actionCount()) : (answer += 1) {
            const group = self.actionGroup(answer);
            for (group.lo..group.hi) |i| out[i] = current;
        }
    }
};

pub fn layout(c: cfg.Config) Layout {
    return .{ .group_size = c.arithmetic_group_size, .max_operand = c.arithmetic_max_operand };
}

/// The controlled held-out-combination split.  Training retains zero-addend
/// examples producing 4, so answer assembly 4 is trained; only the three
/// genuinely compositional `a + b = 4` combinations are withheld. Thus a table
/// keyed by the full pair has no entry at evaluation, while a rule that shares
/// number/operator structure can succeed.
pub fn isHeldOutCombination(example: Example) bool {
    return example.operation == .add and example.lhs > 0 and example.rhs > 0 and
        example.lhs + example.rhs == 4;
}

/// A tiny action-state controller learned exclusively from the increment and
/// decrement curriculum. `observeTransition` is only defined for `n +/- 1`;
/// it stores a successor/predecessor action transition, never an operand-pair
/// answer. A later one-operation trial composes that learned transition `rhs`
/// times from `lhs`. This is the curriculum's explicit anti-memorization seam:
/// a pair lookup has no representation here to store `(lhs, rhs) -> answer`.
pub const TransitionModel = struct {
    max_operand: u8,
    increment: [33]?u8 = [_]?u8{null} ** 33,
    decrement: [33]?u8 = [_]?u8{null} ** 33,

    pub fn init(max_operand: u8) TransitionModel {
        std.debug.assert(max_operand <= 32);
        return .{ .max_operand = max_operand };
    }

    /// Learn one rewarded unit transition. Returns false for a non-curriculum
    /// example, which makes it impossible for the arithmetic phase to smuggle
    /// a full pair answer into this state.
    pub fn observeTransition(self: *TransitionModel, example: Example) bool {
        if (example.rhs != 1) return false;
        switch (example.operation) {
            .add => {
                if (example.lhs >= self.max_operand) return false;
                self.increment[example.lhs] = example.result();
            },
            .subtract => {
                if (example.lhs == 0) return false;
                self.decrement[example.lhs] = example.result();
            },
        }
        return true;
    }

    /// Execute a bounded single operation by repeatedly applying learned action
    /// transitions. Returns null if the curriculum did not yet teach a needed
    /// step; callers must then fall back to an unbiased action choice.
    pub fn solve(self: TransitionModel, example: Example) ?u8 {
        var state = example.lhs;
        var count: u8 = 0;
        while (count < example.rhs) : (count += 1) {
            state = switch (example.operation) {
                .add => self.increment[state] orelse return null,
                .subtract => self.decrement[state] orelse return null,
            };
        }
        return state;
    }
};

// ===========================================================================
// Tests
// ===========================================================================

const testing = std.testing;

test "arithmetic: position-bound symbols and answers are disjoint" {
    const c = cfg.Config{
        .n_neurons = 200,
        .arithmetic_enabled = true,
        .arithmetic_group_size = 6,
        .arithmetic_max_operand = 4,
    };
    try c.validate();
    const l = layout(c);
    try testing.expect(l.operandGroup(.lhs, 2).hi <= c.nExcitatory());
    try testing.expect(l.actionGroup(8).hi <= c.nExcitatory());
    try testing.expect(l.operandGroup(.lhs, 2).hi <= l.operandGroup(.rhs, 0).lo);
    try testing.expect(l.operandGroup(.rhs, 4).hi <= l.actionGroup(0).lo);
}

test "arithmetic: encoder drives exactly one position-bound symbol" {
    const c = cfg.Config{ .n_neurons = 200, .arithmetic_enabled = true, .arithmetic_group_size = 6 };
    const l = layout(c);
    var ext = [_]f32{0} ** 200;
    const e = Example{ .lhs = 2, .operation = .add, .rhs = 3 };
    l.fillPhase(e, .lhs, 1.5, &ext);
    for (l.operandGroup(.lhs, 2).lo..l.operandGroup(.lhs, 2).hi) |i| try testing.expectEqual(@as(f32, 1.5), ext[i]);
    for (l.operandGroup(.rhs, 2).lo..l.operandGroup(.rhs, 2).hi) |i| try testing.expectEqual(@as(f32, 0), ext[i]);
}

test "arithmetic: unit transitions compose an unseen operand pair" {
    var transitions = TransitionModel.init(4);
    try testing.expect(transitions.observeTransition(.{ .lhs = 0, .operation = .add, .rhs = 1 }));
    try testing.expect(transitions.observeTransition(.{ .lhs = 1, .operation = .add, .rhs = 1 }));
    try testing.expect(transitions.observeTransition(.{ .lhs = 2, .operation = .add, .rhs = 1 }));
    try testing.expect(transitions.observeTransition(.{ .lhs = 3, .operation = .add, .rhs = 1 }));
    // (1, 3) was never observed as a pair; only unit transitions were.
    try testing.expectEqual(@as(?u8, 4), transitions.solve(.{ .lhs = 1, .operation = .add, .rhs = 3 }));
}

test "arithmetic: held split excludes internal addends but retains answer examples" {
    try testing.expect(isHeldOutCombination(.{ .lhs = 1, .operation = .add, .rhs = 3 }));
    try testing.expect(!isHeldOutCombination(.{ .lhs = 0, .operation = .add, .rhs = 4 }));
    try testing.expect(!isHeldOutCombination(.{ .lhs = 3, .operation = .subtract, .rhs = 1 }));
}
