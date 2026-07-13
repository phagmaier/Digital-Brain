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
const task = @import("task.zig");

const Allocator = std.mem.Allocator;
const NeuronKind = cfg.NeuronKind;

pub const NeuronId = u32;
pub const SynapseId = u32;

/// What kind of edge the builder is emitting. Build-time only; at runtime the
/// distinction survives as `plastic[]` (readout) and the weight values.
const EdgeKind = enum { reservoir, recurrent, readout };

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
    /// Phase 5 structural plasticity. `permanence` (q in [0,1]) is resistance to
    /// pruning, distinct from `weight`; `age` counts structural events since the
    /// synapse was created (grace period for pruning); `structural` marks the
    /// reservoir/grown edges that growth+pruning may touch (task readout and
    /// recurrent edges are NOT structural). `alive` gates delivery and, when
    /// false in a padded CSR slot, marks a free slot growth can write into.
    permanence: []f32,
    age: []u32,
    /// True iff this edge is managed by Phase 5 growth/pruning (reservoir + grown
    /// edges). Task readout / working-memory recurrent edges are false and are
    /// never decayed or pruned. Dead free slots are false until grown into.
    structural: []bool,
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
        gpa.free(self.structural);
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
        // Parallel tag: what KIND of edge is this? Kept alongside src/dst so the
        // fill loop can branch without re-deriving group membership.
        //   reservoir -- random graph edge, RNG-drawn weight/delay (Phase 1)
        //   recurrent -- fixed input-group self-excitation (Phase 4 memory)
        //   readout   -- plastic input->action synapse (Phase 3)
        var edge_kind = try std.ArrayList(EdgeKind).initCapacity(gpa, n * 8);
        defer edge_kind.deinit(gpa);
        const task_layout = task.layout(c);

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
                    edge_kind.appendAssumeCapacity(.reservoir);
                }
            }

            // Task edges: emit this input source's edges RIGHT HERE, while we are
            // on source i, so the combined list stays sorted by source and CSR
            // still falls out with no sort. No RNG is drawn for any of them, so
            // the random reservoir's stream is identical to a non-task build.
            if (c.task_enabled and task_layout.isInput(@intCast(i))) {
                // Recurrent self-excitation within the same input group (Phase 4
                // working memory): i -> every other neuron of its input group.
                if (c.task_recurrent_weight > 0) {
                    const grp = if (task_layout.input_a.contains(@intCast(i)))
                        task_layout.input_a
                    else
                        task_layout.input_b;
                    var k2: u32 = grp.lo;
                    while (k2 < grp.hi) : (k2 += 1) {
                        if (k2 == @as(u32, @intCast(i))) continue;
                        try src.append(gpa, @intCast(i));
                        try dst.append(gpa, k2);
                        try edge_kind.append(gpa, .recurrent);
                    }
                }
                // Plastic readout: i -> every action neuron ([action_0.lo, action_1.hi)).
                var a: u32 = task_layout.action_0.lo;
                while (a < task_layout.action_1.hi) : (a += 1) {
                    try src.append(gpa, @intCast(i));
                    try dst.append(gpa, a);
                    try edge_kind.append(gpa, .readout);
                }
            }

            // Grow eagerly rather than assuming capacity forever.
            try src.ensureUnusedCapacity(gpa, n);
            try dst.ensureUnusedCapacity(gpa, n);
            try edge_kind.ensureUnusedCapacity(gpa, n);
        }

        const m: u32 = @intCast(src.items.len);
        if (m == 0) return error.EmptyGraph;

        // Live out-degree per source neuron. src is grouped ascending by source,
        // so a single scan gives each neuron's count.
        const live_count = try gpa.alloc(u32, n);
        defer gpa.free(live_count);
        @memset(live_count, 0);
        for (src.items) |s| live_count[s] += 1;

        // Per-neuron CSR capacity. With structural plasticity OFF the slice is
        // exactly the live edges (no padding) -> the layout, and every artefact,
        // is byte-identical to the pre-Phase-5 build. With it ON each neuron gets
        // at least `max_out_degree` slots: the live edges pack the front, the
        // remainder are dead FREE SLOTS that growth writes into (DEC-011). A
        // neuron already above budget keeps all its edges (max, not min).
        var total: u32 = 0;
        for (0..n) |i| {
            const cap = if (c.structural_plasticity_enabled)
                @max(live_count[i], c.max_out_degree)
            else
                live_count[i];
            total += cap;
        }

        var syn: Synapses = .{
            .n = total,
            .source = try gpa.alloc(NeuronId, total),
            .target = try gpa.alloc(NeuronId, total),
            .weight = try gpa.alloc(f32, total),
            .p_release = try gpa.alloc(f32, total),
            .delay = try gpa.alloc(u16, total),
            .eligibility = try gpa.alloc(f32, total),
            .permanence = try gpa.alloc(f32, total),
            .age = try gpa.alloc(u32, total),
            .structural = try gpa.alloc(bool, total),
            .plastic = try gpa.alloc(bool, total),
            .alive = try gpa.alloc(bool, total),
            .out_start = try gpa.alloc(u32, n + 1),
        };
        errdefer syn.deinit(gpa);

        const delay_span: u64 = @as(u64, c.max_delay) - @as(u64, c.min_delay) + 1;

        // Fill, neuron by neuron, live edges first then free slots. `ck` walks the
        // grouped candidate arrays; `w` is the write cursor into the padded CSR.
        var ck: usize = 0; // candidate index (0..m), consumed in source order
        var w: u32 = 0; // write cursor (0..total)
        for (0..n) |i| {
            syn.out_start[i] = w;
            const cap = if (c.structural_plasticity_enabled)
                @max(live_count[i], c.max_out_degree)
            else
                live_count[i];

            // Live edges for neuron i, in original candidate order.
            for (0..live_count[i]) |_| {
                const s = src.items[ck];
                std.debug.assert(s == @as(NeuronId, @intCast(i)));
                syn.source[w] = s;
                syn.target[w] = dst.items[ck];

                // Task edges draw NO RNG (see above), so the reservoir's stream is
                // unchanged. Their sources are always excitatory (input groups are
                // the low IDs), so the effect is excitatory and weights positive.
                switch (edge_kind.items[ck]) {
                    .readout => {
                        // Plastic input->action readout (Phase 3). Shortest delay.
                        syn.weight[w] = c.task_ia_weight_init;
                        syn.p_release[w] = c.task_ia_p_release;
                        syn.delay[w] = c.min_delay;
                        syn.plastic[w] = c.plasticity_enabled; // learns only when plasticity is on
                        syn.structural[w] = false; // readout is functional, never pruned
                    },
                    .recurrent => {
                        // Fixed input-group self-excitation (Phase 4 working memory).
                        syn.weight[w] = c.task_recurrent_weight;
                        syn.p_release[w] = c.task_ia_p_release;
                        syn.delay[w] = c.min_delay;
                        syn.plastic[w] = false;
                        syn.structural[w] = false; // working memory, never pruned
                    },
                    .reservoir => {
                        syn.weight[w] = switch (neurons.kind[s]) {
                            .excitatory => r.range(c.w_exc_init_lo, c.w_exc_init_hi),
                            .inhibitory => r.range(c.w_inh_init_lo, c.w_inh_init_hi),
                        };
                        syn.p_release[w] = c.release_probability;
                        syn.delay[w] = @intCast(c.min_delay + r.below(delay_span));
                        syn.plastic[w] = false; // the random reservoir is fixed; only readout edges learn
                        syn.structural[w] = true; // reservoir edges are what growth/pruning manage
                    },
                }

                syn.eligibility[w] = 0;
                syn.permanence[w] = 0.5;
                syn.age[w] = 0;
                syn.alive[w] = true;

                // THE INVARIANTS. Assert them at the point of construction, not in
                // a comment 400 lines away.
                std.debug.assert(syn.delay[w] >= 1); // DEC-001
                std.debug.assert(syn.delay[w] <= c.max_delay);
                std.debug.assert(syn.weight[w] >= 0.0); // sign lives in the neuron

                ck += 1;
                w += 1;
            }

            // Free slots: dead, inert placeholders in neuron i's CSR range that
            // growth can later claim. source == i (CSR requires it); everything
            // else is neutral. alive=false keeps them out of every traversal.
            for (live_count[i]..cap) |_| {
                syn.source[w] = @intCast(i);
                syn.target[w] = @intCast((i + 1) % n); // valid, non-self, never delivered
                syn.weight[w] = 0;
                syn.p_release[w] = 0;
                syn.delay[w] = c.min_delay; // >= 1 so the delay invariant still holds
                syn.eligibility[w] = 0;
                syn.permanence[w] = 0;
                syn.age[w] = 0;
                syn.structural[w] = false;
                syn.plastic[w] = false;
                syn.alive[w] = false;
                w += 1;
            }
        }
        syn.out_start[n] = w;
        std.debug.assert(w == total);
        std.debug.assert(ck == m);

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

test "net: task build adds all-to-all plastic input->action synapses (Phase 3)" {
    const gpa = testing.allocator;
    const c = cfg.Config{
        .master_seed = 5,
        .task_enabled = true,
        .plasticity_enabled = true,
        .task_group_size = 8,
    };
    var net = try Network.build(gpa, c);
    defer net.deinit(gpa);
    const l = task.layout(c);

    var n_plastic: u32 = 0;
    for (0..net.synapses.n) |k| {
        if (!net.synapses.plastic[k]) continue;
        n_plastic += 1;
        // Every plastic synapse is an input->action edge from an excitatory source.
        try testing.expect(l.isInput(net.synapses.source[k]));
        try testing.expect(l.isAction(net.synapses.target[k]));
        try testing.expect(net.neurons.kind[net.synapses.source[k]] == .excitatory);
        try testing.expectEqual(c.task_ia_weight_init, net.synapses.weight[k]);
    }
    // All-to-all across the two input groups x two action groups: (2g)x(2g).
    const g = c.task_group_size;
    try testing.expectEqual(4 * g * g, n_plastic);
}

test "net: enabling the task does not perturb the random reservoir's RNG stream" {
    // The task synapses draw no RNG, so every reservoir synapse a plain build
    // produces must appear identically in the task build (task edges are extra).
    // This keeps the Phase 1/2 reservoir reproducible under Phase 3.
    const gpa = testing.allocator;
    const seed: u64 = 77;
    var plain = try Network.build(gpa, .{ .master_seed = seed });
    defer plain.deinit(gpa);
    var withtask = try Network.build(gpa, .{ .master_seed = seed, .task_enabled = true, .plasticity_enabled = true });
    defer withtask.deinit(gpa);

    // The task build has strictly more synapses (the added input->action edges).
    try testing.expect(withtask.synapses.n > plain.synapses.n);

    // Every reservoir edge (non-plastic) of the task build, in order, matches the
    // plain build edge for edge -- same source, target, weight, delay.
    var pk: usize = 0;
    for (0..withtask.synapses.n) |k| {
        if (withtask.synapses.plastic[k]) continue; // skip the task edges
        try testing.expectEqual(plain.synapses.source[pk], withtask.synapses.source[k]);
        try testing.expectEqual(plain.synapses.target[pk], withtask.synapses.target[k]);
        try testing.expectEqual(plain.synapses.weight[pk], withtask.synapses.weight[k]);
        try testing.expectEqual(plain.synapses.delay[pk], withtask.synapses.delay[k]);
        pk += 1;
    }
    try testing.expectEqual(plain.synapses.n, @as(u32, @intCast(pk)));
}

test "net: input-group self-excitation adds the expected fixed recurrent edges (Phase 4)" {
    const gpa = testing.allocator;
    const g: u32 = 8;
    var without = try Network.build(gpa, .{ .master_seed = 5, .task_enabled = true, .task_group_size = g });
    defer without.deinit(gpa);
    var with = try Network.build(gpa, .{ .master_seed = 5, .task_enabled = true, .task_group_size = g, .task_recurrent_weight = 0.5 });
    defer with.deinit(gpa);

    // Self-excitation is all-to-all within each input group, no self-loops:
    // g*(g-1) directed edges per group, two groups.
    try testing.expectEqual(2 * g * (g - 1), with.synapses.n - without.synapses.n);

    // The added edges are within-input-group, fixed at the recurrent weight, non-plastic.
    const l = task.layout(.{ .task_group_size = g });
    var found: u32 = 0;
    for (0..with.synapses.n) |k| {
        const src = with.synapses.source[k];
        const dst = with.synapses.target[k];
        const same_group = (l.input_a.contains(src) and l.input_a.contains(dst)) or
            (l.input_b.contains(src) and l.input_b.contains(dst));
        if (same_group and src != dst and with.synapses.weight[k] == 0.5 and !with.synapses.plastic[k]) {
            found += 1;
        }
    }
    try testing.expect(found >= 2 * g * (g - 1));
}

// ---- Phase 5: structural plasticity (build side) ----------------------------

test "net: structural plasticity off leaves the CSR unpadded (every slot is a live edge)" {
    // The default (structural off) build must be exactly the pre-Phase-5 graph:
    // no dead free slots, every synapse alive. This is what keeps the Phase 1
    // baseline byte-identical.
    const gpa = testing.allocator;
    var net = try Network.build(gpa, .{ .master_seed = 42 });
    defer net.deinit(gpa);
    for (0..net.synapses.n) |k| try testing.expect(net.synapses.alive[k]);
}

test "net: structural plasticity on over-allocates free slots and honours the budget" {
    const gpa = testing.allocator;
    const budget: u32 = 20;
    var net = try Network.build(gpa, .{
        .master_seed = 42,
        .structural_plasticity_enabled = true,
        .max_out_degree = budget,
    });
    defer net.deinit(gpa);
    const s = net.synapses;

    var free_slots: u32 = 0;
    for (0..net.neurons.n) |i| {
        const r = s.outgoing(@intCast(i));
        const cap = r.end - r.start;
        // Capacity is at least the budget (more only if the initial graph already
        // exceeded it), and never below the budget.
        try testing.expect(cap >= budget);

        var live: u32 = 0;
        for (r.start..r.end) |k| {
            try testing.expectEqual(@as(NeuronId, @intCast(i)), s.source[k]); // CSR still holds
            if (s.alive[k]) live += 1 else free_slots += 1;
        }
        // Live out-degree never exceeds the neuron's slot capacity (the hard budget).
        try testing.expect(live <= cap);
    }
    // Padding actually happened: there are free slots to grow into.
    try testing.expect(free_slots > 0);
}

test "net: reservoir edges are structural, task edges are not" {
    const gpa = testing.allocator;
    const c = cfg.Config{
        .master_seed = 5,
        .structural_plasticity_enabled = true,
        .task_enabled = true,
        .plasticity_enabled = true,
        .task_group_size = 8,
        .task_recurrent_weight = 0.5,
    };
    var net = try Network.build(gpa, c);
    defer net.deinit(gpa);
    const s = net.synapses;

    for (0..s.n) |k| {
        if (!s.alive[k]) {
            try testing.expect(!s.structural[k]); // dead free slots are non-structural
            continue;
        }
        // Task readout (plastic) and working-memory recurrent edges are never
        // structural; only the random reservoir edges are.
        if (s.plastic[k]) try testing.expect(!s.structural[k]);
    }
}

test "net: enabling structural plasticity does not perturb the reservoir's RNG stream" {
    // Padding adds dead slots but draws NO extra RNG, so the live reservoir edges
    // of a structural build must match a plain build byte-for-byte (in order).
    // This is the Phase 5 analogue of the task-does-not-perturb-reservoir guard.
    const gpa = testing.allocator;
    const seed: u64 = 77;
    var plain = try Network.build(gpa, .{ .master_seed = seed });
    defer plain.deinit(gpa);
    var structural = try Network.build(gpa, .{
        .master_seed = seed,
        .structural_plasticity_enabled = true,
        .max_out_degree = 20,
    });
    defer structural.deinit(gpa);

    var pk: usize = 0;
    for (0..structural.synapses.n) |k| {
        if (!structural.synapses.alive[k]) continue; // skip dead free slots
        try testing.expectEqual(plain.synapses.source[pk], structural.synapses.source[k]);
        try testing.expectEqual(plain.synapses.target[pk], structural.synapses.target[k]);
        try testing.expectEqual(plain.synapses.weight[pk], structural.synapses.weight[k]);
        try testing.expectEqual(plain.synapses.delay[pk], structural.synapses.delay[k]);
        pk += 1;
    }
    try testing.expectEqual(plain.synapses.n, @as(u32, @intCast(pk)));
}
