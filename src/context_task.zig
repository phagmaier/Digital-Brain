//! Stage 2 context-dependent delayed mapping (report.md Stage 2 / final.md §7).
//!
//! Six disjoint groups are carved from the LOW excitatory neuron IDs, each of
//! `context_task_group_size` neurons:
//!
//!   context_x = [0*g, 1*g)   cue_a    = [2*g, 3*g)   action_0 = [4*g, 5*g)
//!   context_y = [1*g, 2*g)   cue_b    = [3*g, 4*g)   action_1 = [5*g, 6*g)
//!
//! The correct mapping is the delayed XOR (cross-coupling):
//!
//!   (X, A) → action 0    (X, B) → action 1
//!   (Y, A) → action 1    (Y, B) → action 0
//!
//! Context and cue are presented at separate times. net.zig installs plastic
//! stimulus→action readout edges from both context and cue assemblies, plus
//! fixed self-excitation within each context group so the context assembly can
//! bridge the inter-stimulus delay (same substrate idea as DEC-010). A pure
//! linear combination of the four assemblies cannot implement the XOR mapping
//! (not linearly separable), so the direct readout is not a sufficient shortcut:
//! solving the mapping requires a context-dependent recurrent state when the
//! cue arrives. The Stage 2 question is whether *locally plastic* reservoir
//! edges are necessary for that computation beyond a fixed-reservoir readout.
//!
//! Dependency-light (config only) so net.zig can use `layout()` at graph-build
//! time without an import cycle.

const std = @import("std");
const cfg = @import("config.zig");
const task = @import("task.zig");

pub const Context = enum(u1) { x, y };
pub const Cue = enum(u1) { a, b };

pub const Group = task.Group;

pub const Layout = struct {
    context_x: Group,
    context_y: Group,
    cue_a: Group,
    cue_b: Group,
    action_0: Group,
    action_1: Group,

    pub fn isContext(self: Layout, id: u32) bool {
        return self.context_x.contains(id) or self.context_y.contains(id);
    }

    pub fn isCue(self: Layout, id: u32) bool {
        return self.cue_a.contains(id) or self.cue_b.contains(id);
    }

    pub fn isStimulus(self: Layout, id: u32) bool {
        return self.isContext(id) or self.isCue(id);
    }

    pub fn isAction(self: Layout, id: u32) bool {
        return self.action_0.contains(id) or self.action_1.contains(id);
    }

    /// True if `id` belongs to any of the six reserved groups.
    pub fn isReserved(self: Layout, id: u32) bool {
        return self.isStimulus(id) or self.isAction(id);
    }

    /// Delayed XOR / cross-coupling mapping used by Stage 2.
    pub fn correctAction(self: Layout, context: Context, cue: Cue) u1 {
        _ = self;
        return switch (context) {
            .x => switch (cue) {
                .a => 0,
                .b => 1,
            },
            .y => switch (cue) {
                .a => 1,
                .b => 0,
            },
        };
    }

    pub fn contextGroup(self: Layout, context: Context) Group {
        return switch (context) {
            .x => self.context_x,
            .y => self.context_y,
        };
    }

    pub fn cueGroup(self: Layout, cue: Cue) Group {
        return switch (cue) {
            .a => self.cue_a,
            .b => self.cue_b,
        };
    }

    /// Fill `out` with current into the chosen context assembly only.
    pub fn fillContext(self: Layout, context: Context, current: f32, out: []f32) void {
        @memset(out, 0);
        const grp = self.contextGroup(context);
        for (grp.lo..grp.hi) |i| out[i] = current;
    }

    /// Fill `out` with current into the chosen cue assembly only.
    pub fn fillCue(self: Layout, cue: Cue, current: f32, out: []f32) void {
        @memset(out, 0);
        const grp = self.cueGroup(cue);
        for (grp.lo..grp.hi) |i| out[i] = current;
    }

    /// Highest exclusive ID reserved by the six groups.
    pub fn reservedHi(self: Layout) u32 {
        return self.action_1.hi;
    }
};

pub fn layout(c: cfg.Config) Layout {
    const g = c.context_task_group_size;
    return .{
        .context_x = .{ .lo = 0 * g, .hi = 1 * g },
        .context_y = .{ .lo = 1 * g, .hi = 2 * g },
        .cue_a = .{ .lo = 2 * g, .hi = 3 * g },
        .cue_b = .{ .lo = 3 * g, .hi = 4 * g },
        .action_0 = .{ .lo = 4 * g, .hi = 5 * g },
        .action_1 = .{ .lo = 5 * g, .hi = 6 * g },
    };
}

// ===========================================================================
// Tests
// ===========================================================================

const testing = std.testing;

test "context_task: groups are disjoint and fit the excitatory population" {
    const c = cfg.Config{
        .n_neurons = 100,
        .context_task_enabled = true,
        .context_task_group_size = 6,
    };
    try c.validate();
    const l = layout(c);

    try testing.expectEqual(@as(u32, 0), l.context_x.lo);
    try testing.expectEqual(@as(u32, 36), l.action_1.hi);
    try testing.expect(l.action_1.hi <= c.nExcitatory());

    for (0..36) |id| {
        const i: u32 = @intCast(id);
        var claims: u8 = 0;
        if (l.context_x.contains(i)) claims += 1;
        if (l.context_y.contains(i)) claims += 1;
        if (l.cue_a.contains(i)) claims += 1;
        if (l.cue_b.contains(i)) claims += 1;
        if (l.action_0.contains(i)) claims += 1;
        if (l.action_1.contains(i)) claims += 1;
        try testing.expectEqual(@as(u8, 1), claims);
    }
}

test "context_task: XOR mapping is the Stage 2 cross-coupling" {
    const l = layout(.{ .context_task_group_size = 6 });
    try testing.expectEqual(@as(u1, 0), l.correctAction(.x, .a));
    try testing.expectEqual(@as(u1, 1), l.correctAction(.x, .b));
    try testing.expectEqual(@as(u1, 1), l.correctAction(.y, .a));
    try testing.expectEqual(@as(u1, 0), l.correctAction(.y, .b));
}

test "context_task: stimuli drive only the chosen assembly" {
    const c = cfg.Config{ .n_neurons = 100, .context_task_group_size = 6 };
    const l = layout(c);
    var out = [_]f32{0} ** 100;

    l.fillContext(.x, 1.5, &out);
    for (0..100) |i| {
        const expected: f32 = if (i < 6) 1.5 else 0.0;
        try testing.expectEqual(expected, out[i]);
    }

    l.fillCue(.b, 1.5, &out);
    for (0..100) |i| {
        const expected: f32 = if (i >= 18 and i < 24) 1.5 else 0.0;
        try testing.expectEqual(expected, out[i]);
    }
}
