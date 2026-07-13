//! Network state. Structure-of-arrays throughout.
//!
//! Two invariants are enforced here and asserted on construction:
//!   1. every synapse has delay >= 1                     (DEC-001)
//!   2. every weight is >= 0; sign lives in the neuron   (spec §6)
//!
//! Adjacency is a CSR-style compressed range per source neuron. Synapses are
//! sorted by source, so `out_start[i] .. out_start[i+1]` is neuron i's outgoing
//! slice. This gives a stable, array-ordered traversal, which is what makes the
//! reproducibility invariant (§5) hold: the sequence of random draws depends
//! only on explicit stable state, never on hash-map iteration order.

const std = @import("std");
const cfg = @import("config.zig");
const rng = @import("rng.zig");

const Allocator = std.mem.Allocator;
const NeuronKind = cfg.NeuronKind;

pub const NeuronId = u32;
pub const SynapseId = u32;

// ---------------------------------------------------------------------------
// Neurons
// ---------------------------------------------------------------------------

pub const Neurons = struct {
    n: u32,

    /// REST-RELATIVE membrane potential (DEC-002). Rest is exactly u = 0.
    u: []f32,
    /// theta, rest-relative. Mutable so Phase 2 homeostasis has somewhere to go.
    threshold: []f32,
    /// a_i, spike-frequency adaptation.
    adaptation: []f32,
    /// rho_i, slow firing-rate EMA. Phase 2 uses it; Phase 1 just logs it.
    rate_ema: []f32,
    refractory: []u16,
    fired: []bool,
    kind: []NeuronKind,
    pos_x: []f32,
    pos_y: []f32,
    /// x_i / y_i. Unused in Phase 1; allocated so Phase 3 is a pure addition.
    pre_trace: []f32,
    post_trace: []f32,

    pub fn init(gpa: Allocator, c: cfg.Config, r: *rng.Rng) !Neurons {
        const n = c.n_neurons;

        var self: Neurons = .{
            .n = n,
            .u = try gpa.alloc(f32, n),
            .threshold = try gpa.alloc(f32, n),
            .adaptation = try gpa.alloc(f32, n),
            .rate_ema = try gpa.alloc(f32, n),
            .refractory = try gpa.alloc(u16, n),
            .fired = try gpa.alloc(bool, n),
            .kind = try gpa.alloc(NeuronKind, n),
            .pos_x = try gpa.alloc(f32, n),
            .pos_y = try gpa.alloc(f32, n),
            .pre_trace = try gpa.alloc(f32, n),
            .post_trace = try gpa.alloc(f32, n),
        };

        // Excitatory neurons take the low IDs, inhibitory the high ones.
        // Deterministic by construction -- no shuffle, no RNG draw, no
        // dependence on iteration order.
        const n_exc: u32 = @intFromFloat(@round(@as(f32, @floatFromInt(n)) * c.excitatory_fraction));

        for (0..n) |i| {
            self.kind[i] = if (i < n_exc) .excitatory else .inhibitory;
            self.pos_x[i] = r.float01();
            self.pos_y[i] = r.float01();
        }

        self.resetFast(c);
        // Thresholds and rate EMA persist across episodes (DEC-005), so they
        // are initialised once, here, and NOT touched by resetFast.
        @memset(self.threshold, c.threshold);
        @memset(self.rate_ema, 0);

        return self;
    }

    /// Everything in the "Reset" column of the episode-boundary table (§15).
    /// Adaptation is deliberately configurable -- see DEC-005.
    pub fn resetFast(self: *Neurons, c: cfg.Config) void {
        @memset(self.u, 0); // rest, in rest-relative coordinates
        @memset(self.refractory, 0);
        @memset(self.fired, false);
        @memset(self.pre_trace, 0);
        @memset(self.post_trace, 0);
        if (c.reset_adaptation_between_episodes) @memset(self.adaptation, 0);
    }

    pub fn nExcitatory(self: Neurons) u32 {
        var count: u32 = 0;
        for (self.kind) |k| {
            if (k == .excitatory) count += 1;
        }
        return count;
    }

    pub fn deinit(self: *Neurons, gpa: Allocator) void {
        gpa.free(self.u);
        gpa.free(self.threshold);
        gpa.free(self.adaptation);
        gpa.free(self.rate_ema);
        gpa.free(self.refractory);
        gpa.free(self.fired);
        gpa.free(self.kind);
        gpa.free(self.pos_x);
        gpa.free(self.pos_y);
        gpa.free(self.pre_trace);
        gpa.free(self.post_trace);
    }
};

// ---------------------------------------------------------------------------
// Synapses
// ---------------------------------------------------------------------------

pub const Synapses = struct {
    n: u32,

    source: []NeuronId,
    target: []NeuronId,
    /// w >= 0 ALWAYS. The sign of the effect is the SOURCE NEURON's sign.
    weight: []f32,
    p_release: []f32,
    /// >= 1 ALWAYS (DEC-001).
    delay: []u16,
    /// Phase 3+. Allocated now so adding learning is not a refactor.
    eligibility: []f32,
    /// Phase 5+.
    permanence: []f32,
    age: []u32,
    plastic: []bool,
    alive: []bool,

    /// CSR outgoing ranges: neuron i owns synapses [out_start[i], out_start[i+1]).
    out_start: []u32,

    pub fn deinit(self: *Synapses, gpa: Allocator) void {
        gpa.free(self.source);
        gpa.free(self.target);
        gpa.free(self.weight);
        gpa.free(self.p_release);
        gpa.free(self.delay);
        gpa.free(self.eligibility);
        gpa.free(self.permanence);
        gpa.free(self.age);
        gpa.free(self.plastic);
        gpa.free(self.alive);
        gpa.free(self.out_start);
    }

    pub fn outgoing(self: Synapses, i: NeuronId) struct { start: u32, end: u32 } {
        return .{ .start = self.out_start[i], .end = self.out_start[i + 1] };
    }
};

// ---------------------------------------------------------------------------
// Network
// ---------------------------------------------------------------------------

pub const Network = struct {
    neurons: Neurons,
    synapses: Synapses,
    config: cfg.Config,

    pub fn deinit(self: *Network, gpa: Allocator) void {
        self.neurons.deinit(gpa);
        self.synapses.deinit(gpa);
    }

    /// Build the fixed sparse graph.
    ///
    /// Uses the DERIVED init key (DEC-004), so the topology is identical across
    /// every ablation variant that shares a master seed -- by construction, not
    /// by luck.
    pub fn build(gpa: Allocator, c: cfg.Config) !Network {
        try c.validate();

        var r = rng.derived(c.master_seed, .init_graph, 0);
        var neurons = try Neurons.init(gpa, c, &r);
        errdefer neurons.deinit(gpa);

        const n = c.n_neurons;

        // Candidate edges are enumerated in a fixed order (i ascending, then j
        // ascending) and accepted/rejected with one RNG draw each. Deterministic,
        // and no dependence on container iteration order.
        var src = try std.ArrayList(NeuronId).initCapacity(gpa, n * 8);
        defer src.deinit(gpa);
        var dst = try std.ArrayList(NeuronId).initCapacity(gpa, n * 8);
        defer dst.deinit(gpa);

        const two_sigma_sq = 2.0 * c.spatial_sigma * c.spatial_sigma;

        // Normalize so that mean acceptance probability lands near the target
        // density. With a Gaussian kernel the raw mean is < 1, so we scale up.
        // Cheap enough at this size to just measure it.
        var kernel_sum: f64 = 0;
        var pairs: f64 = 0;
        if (!c.uniform_graph) {
            for (0..n) |i| {
                for (0..n) |j| {
                    if (c.no_self_connections and i == j) continue;
                    const dx = neurons.pos_x[i] - neurons.pos_x[j];
                    const dy = neurons.pos_y[i] - neurons.pos_y[j];
                    const d2 = dx * dx + dy * dy;
                    kernel_sum += @exp(-d2 / two_sigma_sq);
                    pairs += 1;
                }
            }
        }
        const mean_kernel: f32 = if (c.uniform_graph or pairs == 0)
            1.0
        else
            @floatCast(kernel_sum / pairs);
        const scale = c.connection_density / @max(mean_kernel, 1e-6);

        for (0..n) |i| {
            for (0..n) |j| {
                if (c.no_self_connections and i == j) continue;

                const p: f32 = if (c.uniform_graph) c.connection_density else blk: {
                    const dx = neurons.pos_x[i] - neurons.pos_x[j];
                    const dy = neurons.pos_y[i] - neurons.pos_y[j];
                    const d2 = dx * dx + dy * dy;
                    break :blk scale * @exp(-d2 / two_sigma_sq);
                };

                if (r.float01() < @min(p, 1.0)) {
                    src.appendAssumeCapacity(@intCast(i));
                    dst.appendAssumeCapacity(@intCast(j));
                }
            }
            // Grow eagerly rather than assuming capacity forever.
            try src.ensureUnusedCapacity(gpa, n);
            try dst.ensureUnusedCapacity(gpa, n);
        }

        const m: u32 = @intCast(src.items.len);
        if (m == 0) return error.EmptyGraph;

        var syn: Synapses = .{
            .n = m,
            .source = try gpa.alloc(NeuronId, m),
            .target = try gpa.alloc(NeuronId, m),
            .weight = try gpa.alloc(f32, m),
            .p_release = try gpa.alloc(f32, m),
            .delay = try gpa.alloc(u16, m),
            .eligibility = try gpa.alloc(f32, m),
            .permanence = try gpa.alloc(f32, m),
            .age = try gpa.alloc(u32, m),
            .plastic = try gpa.alloc(bool, m),
            .alive = try gpa.alloc(bool, m),
            .out_start = try gpa.alloc(u32, n + 1),
        };
        errdefer syn.deinit(gpa);

        // src is already sorted ascending by construction (outer loop is i), so
        // CSR ranges fall out directly. No sort, no hash map.
        const delay_span: u64 = @as(u64, c.max_delay) - @as(u64, c.min_delay) + 1;

        for (0..m) |k| {
            const s = src.items[k];
            syn.source[k] = s;
            syn.target[k] = dst.items[k];

            syn.weight[k] = switch (neurons.kind[s]) {
                .excitatory => r.range(c.w_exc_init_lo, c.w_exc_init_hi),
                .inhibitory => r.range(c.w_inh_init_lo, c.w_inh_init_hi),
            };
            syn.p_release[k] = c.release_probability;
            syn.delay[k] = @intCast(c.min_delay + r.below(delay_span));

            syn.eligibility[k] = 0;
            syn.permanence[k] = 0.5;
            syn.age[k] = 0;
            syn.plastic[k] = false; // Phase 1: nothing learns. DEC-006 governs later.
            syn.alive[k] = true;

            // THE INVARIANTS. Assert them at the point of construction, not in
            // a comment 400 lines away.
            std.debug.assert(syn.delay[k] >= 1); // DEC-001
            std.debug.assert(syn.delay[k] <= c.max_delay);
            std.debug.assert(syn.weight[k] >= 0.0); // sign lives in the neuron
        }

        // Build CSR offsets.
        var k: usize = 0;
        for (0..n) |i| {
            syn.out_start[i] = @intCast(k);
            while (k < m and syn.source[k] == @as(NeuronId, @intCast(i))) : (k += 1) {}
        }
        syn.out_start[n] = m;

        return .{ .neurons = neurons, .synapses = syn, .config = c };
    }
};

// ===========================================================================
// Tests
// ===========================================================================

const testing = std.testing;

test "net: builds and satisfies invariants" {
    const gpa = testing.allocator;
    const c = cfg.Config{};
    var net = try Network.build(gpa, c);
    defer net.deinit(gpa);

    try testing.expect(net.synapses.n > 0);

    for (0..net.synapses.n) |k| {
        // DEC-001: no zero-delay synapses, ever.
        try testing.expect(net.synapses.delay[k] >= 1);
        try testing.expect(net.synapses.delay[k] <= c.max_delay);
        // w >= 0: sign is carried by the source neuron's type.
        try testing.expect(net.synapses.weight[k] >= 0.0);
        if (c.no_self_connections) {
            try testing.expect(net.synapses.source[k] != net.synapses.target[k]);
        }
    }
}

test "net: CSR ranges are consistent" {
    const gpa = testing.allocator;
    var net = try Network.build(gpa, .{});
    defer net.deinit(gpa);

    const s = net.synapses;
    try testing.expectEqual(@as(u32, 0), s.out_start[0]);
    try testing.expectEqual(s.n, s.out_start[net.neurons.n]);

    for (0..net.neurons.n) |i| {
        const rangeI = s.outgoing(@intCast(i));
        try testing.expect(rangeI.start <= rangeI.end);
        for (rangeI.start..rangeI.end) |k| {
            try testing.expectEqual(@as(NeuronId, @intCast(i)), s.source[k]);
        }
    }
}

test "net: same seed produces identical topology" {
    const gpa = testing.allocator;
    var a = try Network.build(gpa, .{ .master_seed = 42 });
    defer a.deinit(gpa);
    var b = try Network.build(gpa, .{ .master_seed = 42 });
    defer b.deinit(gpa);

    try testing.expectEqual(a.synapses.n, b.synapses.n);
    try testing.expectEqualSlices(NeuronId, a.synapses.source, b.synapses.source);
    try testing.expectEqualSlices(NeuronId, a.synapses.target, b.synapses.target);
    try testing.expectEqualSlices(u16, a.synapses.delay, b.synapses.delay);
    try testing.expectEqualSlices(f32, a.synapses.weight, b.synapses.weight);
}

test "net: E/I split is 80/20 by default" {
    const gpa = testing.allocator;
    var net = try Network.build(gpa, .{ .n_neurons = 100 });
    defer net.deinit(gpa);
    try testing.expectEqual(@as(u32, 80), net.neurons.nExcitatory());
}

test "net: spatial bias produces shorter edges than a uniform graph" {
    // The point of the spatial kernel is that connections are LOCAL. If mean
    // edge length is the same as uniform, the kernel is doing nothing and the
    // Phase 5 structural-growth story has no foundation.
    const gpa = testing.allocator;

    var local = try Network.build(gpa, .{ .master_seed = 9, .spatial_sigma = 0.12 });
    defer local.deinit(gpa);
    var uniform = try Network.build(gpa, .{ .master_seed = 9, .uniform_graph = true });
    defer uniform.deinit(gpa);

    const meanLen = struct {
        fn f(net: Network) f64 {
            var total: f64 = 0;
            for (0..net.synapses.n) |k| {
                const s = net.synapses.source[k];
                const t = net.synapses.target[k];
                const dx = net.neurons.pos_x[s] - net.neurons.pos_x[t];
                const dy = net.neurons.pos_y[s] - net.neurons.pos_y[t];
                total += @sqrt(@as(f64, dx * dx + dy * dy));
            }
            return total / @as(f64, @floatFromInt(net.synapses.n));
        }
    }.f;

    try testing.expect(meanLen(local) < meanLen(uniform) * 0.75);
}
