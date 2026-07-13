//! The Phase 3 task: an immediate two-choice association (DEC-008).
//!
//! Four disjoint groups are carved from the LOW excitatory neuron IDs, each of
//! `task_group_size` neurons:
//!
//!   input_a  = [0*g, 1*g)   action_0 = [2*g, 3*g)
//!   input_b  = [1*g, 2*g)   action_1 = [3*g, 4*g)
//!
//! Stimulus A injects current into input_a, B into input_b. The readout compares
//! spike counts in action_0 vs action_1. The correct mapping is fixed:
//!
//!   A -> action_0,   B -> action_1
//!
//! Fixed (not per-seed) because the groups start symmetric -- initial
//! input->action weights are equal -- so there is no trivial solution to
//! exploit; the network must break the symmetry from reward alone.
//!
//! This module is deliberately dependency-light (config only, no net/sim import)
//! so net.zig can use `layout()` at graph-build time without an import cycle.

const std = @import("std");
const cfg = @import("config.zig");

pub const Choice = enum(u1) { a, b };

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

pub const Layout = struct {
    input_a: Group,
    input_b: Group,
    action_0: Group,
    action_1: Group,

    /// True if `id` is in either input group.
    pub fn isInput(self: Layout, id: u32) bool {
        return self.input_a.contains(id) or self.input_b.contains(id);
    }
    /// True if `id` is in either action group.
    pub fn isAction(self: Layout, id: u32) bool {
        return self.action_0.contains(id) or self.action_1.contains(id);
    }
    /// The action group for a stimulus, under the fixed correct mapping.
    pub fn correctAction(self: Layout, choice: Choice) u1 {
        _ = self;
        return switch (choice) {
            .a => 0,
            .b => 1,
        };
    }
    /// Fill `out` (length n_neurons) with the stimulus current for `choice`:
    /// `current` into the active input group, zero everywhere else.
    pub fn fillStimulus(self: Layout, choice: Choice, current: f32, out: []f32) void {
        @memset(out, 0);
        const grp = switch (choice) {
            .a => self.input_a,
            .b => self.input_b,
        };
        for (grp.lo..grp.hi) |i| out[i] = current;
    }
};

pub fn layout(c: cfg.Config) Layout {
    const g = c.task_group_size;
    return .{
        .input_a = .{ .lo = 0 * g, .hi = 1 * g },
        .input_b = .{ .lo = 1 * g, .hi = 2 * g },
        .action_0 = .{ .lo = 2 * g, .hi = 3 * g },
        .action_1 = .{ .lo = 3 * g, .hi = 4 * g },
    };
}

// ===========================================================================
// Tests
// ===========================================================================

const testing = std.testing;

test "task: groups are disjoint and sit inside the excitatory population" {
    const c = cfg.Config{ .n_neurons = 100, .task_enabled = true, .task_group_size = 8 };
    try c.validate();
    const l = layout(c);

    try testing.expectEqual(@as(u32, 0), l.input_a.lo);
    try testing.expectEqual(@as(u32, 32), l.action_1.hi);
    try testing.expect(l.action_1.hi <= c.nExcitatory()); // 32 <= 80

    // Disjoint: no id is claimed by two groups.
    for (0..32) |id| {
        const i: u32 = @intCast(id);
        var claims: u8 = 0;
        if (l.input_a.contains(i)) claims += 1;
        if (l.input_b.contains(i)) claims += 1;
        if (l.action_0.contains(i)) claims += 1;
        if (l.action_1.contains(i)) claims += 1;
        try testing.expectEqual(@as(u8, 1), claims);
    }
}

test "task: stimulus drives only the chosen input group" {
    const c = cfg.Config{ .n_neurons = 100, .task_enabled = true, .task_group_size = 8 };
    const l = layout(c);
    var out = [_]f32{0} ** 100;

    l.fillStimulus(.a, 1.5, &out);
    for (0..100) |i| {
        const expected: f32 = if (i < 8) 1.5 else 0.0;
        try testing.expectEqual(expected, out[i]);
    }

    l.fillStimulus(.b, 1.5, &out);
    for (0..100) |i| {
        const expected: f32 = if (i >= 8 and i < 16) 1.5 else 0.0;
        try testing.expectEqual(expected, out[i]);
    }
}

test "task: correct mapping is A->0, B->1" {
    const l = layout(.{ .task_group_size = 8 });
    try testing.expectEqual(@as(u1, 0), l.correctAction(.a));
    try testing.expectEqual(@as(u1, 1), l.correctAction(.b));
}
