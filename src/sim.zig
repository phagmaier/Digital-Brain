//! The simulator. Fixed step ordering (spec §9), rest-relative membrane,
//! stochastic firing, stochastic release, delayed delivery via a ring buffer.
//!
//! No learning. No growth. No workspace. Phase 1 only.

const std = @import("std");
const cfg = @import("config.zig");
const net = @import("net.zig");
const rng = @import("rng.zig");

const Allocator = std.mem.Allocator;
const Network = net.Network;

// ---------------------------------------------------------------------------
// Delayed event queue
// ---------------------------------------------------------------------------

/// A ring buffer of per-neuron current accumulators, one bucket per possible
/// delay. Because delays are bounded and >= 1, we never need a heap or a sorted
/// queue: schedule() just adds into a future bucket.
///
/// Accumulating a f32 directly (rather than queueing event records) also removes
/// a subtle reproducibility hazard: with event records, delivery order within a
/// bucket would depend on insertion order, and float addition is not associative.
/// Here the addition order is fixed by the neuron/synapse traversal order, which
/// is itself fixed (ascending ID, then CSR array order).
pub const EventQueue = struct {
    /// buckets[b][i] = current arriving at neuron i in bucket b.
    buckets: [][]f32,
    n_buckets: u32,
    n_neurons: u32,

    pub fn init(gpa: Allocator, c: cfg.Config) !EventQueue {
        // Need one bucket per delay in [1, max_delay], plus the current one.
        const n_buckets: u32 = @as(u32, c.max_delay) + 1;
        const buckets = try gpa.alloc([]f32, n_buckets);
        errdefer gpa.free(buckets);

        var made: usize = 0;
        errdefer for (buckets[0..made]) |b| gpa.free(b);

        for (buckets) |*b| {
            b.* = try gpa.alloc(f32, c.n_neurons);
            @memset(b.*, 0);
            made += 1;
        }

        return .{ .buckets = buckets, .n_buckets = n_buckets, .n_neurons = c.n_neurons };
    }

    pub fn deinit(self: *EventQueue, gpa: Allocator) void {
        for (self.buckets) |b| gpa.free(b);
        gpa.free(self.buckets);
    }

    /// Episode boundary: the scheduled event queue is CLEARED (§15).
    pub fn clear(self: *EventQueue) void {
        for (self.buckets) |b| @memset(b, 0);
    }

    /// Schedule `current` to arrive at `target` after `delay` steps.
    ///
    /// delay >= 1 is asserted here as well as at construction, because THIS is
    /// where a zero would do its damage: (t + 0) % n_buckets is the bucket we
    /// are currently draining, so the event would either be dropped or resurface
    /// a full wrap-around later. See DEC-001.
    pub fn schedule(self: *EventQueue, t: u32, delay: u16, target: net.NeuronId, current: f32) void {
        std.debug.assert(delay >= 1);
        std.debug.assert(delay < self.n_buckets);
        const b = (t + delay) % self.n_buckets;
        self.buckets[b][target] += current;
    }

    /// Drain the bucket for timestep t into `out`, then zero it.
    pub fn deliver(self: *EventQueue, t: u32, out: []f32) void {
        const b = t % self.n_buckets;
        const bucket = self.buckets[b];
        @memcpy(out, bucket);
        @memset(bucket, 0);
    }
};

// ---------------------------------------------------------------------------
// Per-step metrics
// ---------------------------------------------------------------------------

pub const StepMetrics = struct {
    t: u32 = 0,
    spikes: u32 = 0,
    exc_spikes: u32 = 0,
    inh_spikes: u32 = 0,
    mean_u: f32 = 0,
    /// Magnitude of excitatory vs inhibitory current arriving this step. The
    /// E/I balance readout -- one of the few numbers that tells you WHY the
    /// network died or exploded, rather than just that it did.
    exc_current: f32 = 0,
    inh_current: f32 = 0,
    scheduled_events: u32 = 0,
    /// Population-mean adaptive threshold. Flat in Phase 1; when homeostasis is
    /// on this is the trace that shows the controller working -- it rises to
    /// suppress activity and relaxes when activity falls. Watching it next to
    /// the firing rate is how you see regulation happen.
    mean_threshold: f32 = 0,
    /// Population-mean per-neuron firing-rate EMA (rho_i). The homeostat's
    /// input signal; converges toward target_rate when regulation is working.
    mean_rate_ema: f32 = 0,
};

// ---------------------------------------------------------------------------
// Simulator
// ---------------------------------------------------------------------------

pub const Sim = struct {
    network: Network,
    queue: EventQueue,

    /// Current arriving this step, per neuron.
    current: []f32,

    /// RUNNING streams (DEC-004). Firing and release are high-volume per-step
    /// draws; cross-variant alignment is impossible here once dynamics diverge,
    /// so a running stream is correct. Task/action/init/growth use derived keys.
    rng_firing: rng.Rng,
    rng_release: rng.Rng,

    /// Running reward baseline (DEC-009). Subtracted from reward in applyReward
    /// so the plastic update is zero-mean once the task is solved. SLOW state:
    /// it persists across episodes and is NOT reset by resetEpisode.
    reward_baseline: f32 = 0,

    t: u32 = 0,

    pub fn init(gpa: Allocator, c: cfg.Config) !Sim {
        var network = try Network.build(gpa, c);
        errdefer network.deinit(gpa);

        var queue = try EventQueue.init(gpa, c);
        errdefer queue.deinit(gpa);

        const current = try gpa.alloc(f32, c.n_neurons);
        errdefer gpa.free(current);
        @memset(current, 0);

        return .{
            .network = network,
            .queue = queue,
            .current = current,
            // Distinct constants so the two streams cannot accidentally align.
            .rng_firing = rng.Rng.init(c.master_seed ^ 0xF1_1E_00_00_00_00_00_01),
            .rng_release = rng.Rng.init(c.master_seed ^ 0xE1_1E_00_00_00_00_00_02),
        };
    }

    pub fn deinit(self: *Sim, gpa: Allocator) void {
        gpa.free(self.current);
        self.queue.deinit(gpa);
        self.network.deinit(gpa);
    }

    /// Episode boundary. See §15 -- the table is normative, and getting this
    /// wrong is the classic way to produce a local-learning result that turns
    /// out to be information leaking between examples.
    ///
    /// Note what is NOT reset: weights, thresholds, rate EMA, reward baseline,
    /// running RNG streams. Note what IS: membrane, refractory, event queue,
    /// traces, and eligibility (each episode's credit assignment is independent).
    pub fn resetEpisode(self: *Sim) void {
        const c = self.network.config;
        self.network.neurons.resetFast(c);
        self.queue.clear();
        @memset(self.current, 0);
        // Eligibility is per-episode fast state: a synapse's tag must not carry
        // credit from a previous example into this one's reward (DEC-005/DEC-009).
        @memset(self.network.synapses.eligibility, 0);
        self.t = 0;
        // RNG streams and reward baseline: continue, never reset.
    }

    /// One simulation step. The ORDER here is the spec (§9) and is load-bearing.
    pub fn step(self: *Sim, external: ?[]const f32) StepMetrics {
        const c = self.network.config;
        const nrn = &self.network.neurons;
        const syn = &self.network.synapses;
        const n = nrn.n;

        var m = StepMetrics{ .t = self.t };

        // 1. Deliver synaptic events scheduled for this timestep.
        self.queue.deliver(self.t, self.current);

        for (self.current) |cur| {
            if (cur > 0) m.exc_current += cur else m.inh_current += -cur;
        }

        // 2. External input. (Phase 1: none. Phase 3: task encoding lands here.)
        if (external) |ext| {
            for (self.current, ext) |*cur, e| cur.* += e;
        }

        // 2b. Background drive. This is the knob that sets the operating point.
        //     NOT beta. See config.zig.
        for (self.current) |*cur| cur.* += c.background_current;

        // 3. Workspace broadcast -- Phase 7. Deliberately absent.

        // 4-6. Membrane, adaptation, refractory, firing probability, spike sample.
        var u_sum: f32 = 0;

        for (0..n) |i| {
            // Adaptation decays regardless of whether the neuron fires.
            if (c.adaptation_enabled) {
                nrn.adaptation[i] *= c.adaptation_decay;
            } else {
                nrn.adaptation[i] = 0;
            }

            // Membrane, REST-RELATIVE (DEC-002):
            //   u(t+1) = lambda_u * u(t) + I(t) - a(t)
            // Leaks toward zero. No ambiguity about the fixed point, because
            // there is only one coordinate system.
            nrn.u[i] = c.membrane_leak * nrn.u[i] + self.current[i] - nrn.adaptation[i];

            u_sum += nrn.u[i];

            // Refractory: tick down, and block firing this step.
            var can_fire = true;
            if (nrn.refractory[i] > 0) {
                nrn.refractory[i] -= 1;
                can_fire = false;
            }

            // Stochastic firing: P = sigmoid(beta * (u - theta)).
            var fired = false;
            if (can_fire) {
                const p = sigmoid(c.beta * (nrn.u[i] - nrn.threshold[i]));
                fired = self.rng_firing.bernoulli(p);
            }
            nrn.fired[i] = fired;

            if (fired) {
                m.spikes += 1;
                switch (nrn.kind[i]) {
                    .excitatory => m.exc_spikes += 1,
                    .inhibitory => m.inh_spikes += 1,
                }

                // Reset. Subtractive by default, decoupled from theta (DEC-003).
                switch (c.reset_rule) {
                    .subtractive => nrn.u[i] -= c.reset_decrement,
                    .hard => nrn.u[i] = 0,
                }

                nrn.refractory[i] = c.refractory_steps;
                if (c.adaptation_enabled) nrn.adaptation[i] += c.adaptation_increment;
            }

            // Slow firing-rate EMA. Phase 2 homeostasis reads this; Phase 1 logs it.
            const s: f32 = if (fired) 1.0 else 0.0;
            nrn.rate_ema[i] = c.rate_ema_decay * nrn.rate_ema[i] + (1.0 - c.rate_ema_decay) * s;
        }

        m.mean_u = u_sum / @as(f32, @floatFromInt(n));

        // 7. Schedule outgoing transmissions into FUTURE buckets.
        //    Traversal is ascending neuron ID, then CSR array order. This fixes
        //    the sequence of release draws, which is what makes the run
        //    reproducible (§5).
        for (0..n) |i| {
            if (!nrn.fired[i]) continue;

            const sign = nrn.kind[i].sign();
            const r = syn.outgoing(@intCast(i));

            for (r.start..r.end) |k| {
                if (!syn.alive[k]) continue;

                // Stochastic release. INDEPENDENT of stochastic firing, so the
                // two randomness sources can be ablated separately.
                if (!self.rng_release.bernoulli(syn.p_release[k])) continue;

                // I_j(t + d) += d_i * w_ij
                // Sign from the neuron, magnitude from the weight. Never mixed.
                self.queue.schedule(self.t, syn.delay[k], syn.target[k], sign * syn.weight[k]);
                m.scheduled_events += 1;
            }
        }

        // 8-9. Pre/post traces and eligibility (Phase 3, DEC-009). Deterministic,
        //      no RNG, stable traversal order -> no reproducibility hazard.
        if (c.plasticity_enabled) self.updateEligibility();

        // 10.  Workspace competition -- Phase 7. Deliberately absent.

        // 11. Homeostasis (Phase 2). In the continuous simulator this runs every
        //     step. The Phase 3 episode driver instead sets homeostasis_per_step
        //     = false and calls applyHomeostasis() once per episode -- the doc's
        //     "Update homeostasis" step, at per-episode/window cadence.
        if (c.homeostasis_per_step) self.applyHomeostasis();

        // Observability: mean threshold and mean rate EMA, so the controller is
        // visible in metrics.csv without post-hoc computation. Recorded after the
        // per-step homeostatic update (if any), i.e. the post-update state.
        var th_sum: f32 = 0;
        var rho_sum: f32 = 0;
        for (0..n) |i| {
            th_sum += nrn.threshold[i];
            rho_sum += nrn.rate_ema[i];
        }
        m.mean_threshold = th_sum / @as(f32, @floatFromInt(n));
        m.mean_rate_ema = rho_sum / @as(f32, @floatFromInt(n));

        self.t += 1;
        return m;
    }

    /// Apply both homeostats once, each gated by its own enable flag. This is
    /// the unit of "Update homeostasis": Sim.step calls it every step when
    /// config.homeostasis_per_step is true; otherwise the caller (the Phase 3
    /// episode driver) calls it explicitly at the end of each episode/window.
    ///
    /// Deterministic -- no RNG -- and traversed in stable order (neurons by
    /// ascending id, synapses in CSR order), so it never touches the
    /// reproducibility invariant. Reads the current rate EMA, which the step
    /// loop has already updated this step.
    pub fn applyHomeostasis(self: *Sim) void {
        const c = self.network.config;
        const nrn = &self.network.neurons;
        const syn = &self.network.synapses;
        const n = nrn.n;

        // 11a. Adaptive thresholds. Negative feedback on each neuron's own rate.
        if (c.homeostasis_enabled) {
            for (0..n) |i| {
                nrn.threshold[i] += c.homeostasis_lr * (nrn.rate_ema[i] - c.target_rate);
                nrn.threshold[i] = std.math.clamp(nrn.threshold[i], c.threshold_min, c.threshold_max);
            }
        }

        // 11b. Synaptic scaling (DEC-007). Postsynaptic, excitatory-input only.
        //      Uses the same rate EMA as the threshold homeostat.
        if (c.weight_normalization_enabled) {
            for (0..syn.n) |k| {
                if (!syn.alive[k]) continue;
                if (nrn.kind[syn.source[k]] != .excitatory) continue;
                const j = syn.target[k];
                const factor = 1.0 + c.weight_norm_lr * (c.target_rate - nrn.rate_ema[j]);
                syn.weight[k] = std.math.clamp(syn.weight[k] * factor, 0.0, c.weight_max);
            }
        }
    }

    /// Advance the pre/post traces and the per-synapse eligibility one step
    /// (DEC-009). Called from step() when plasticity is enabled. The eligibility
    /// update reads the traces as they stood BEFORE this step's spikes, then the
    /// traces are advanced -- so a synapse is tagged for "pre fired, THEN post
    /// fired", the causal order this delay>=1 model can represent.
    fn updateEligibility(self: *Sim) void {
        const c = self.network.config;
        const nrn = &self.network.neurons;
        const syn = &self.network.synapses;
        const n = nrn.n;

        // Eligibility first, using the pre-this-step traces.
        for (0..syn.n) |k| {
            if (!syn.plastic[k]) continue;
            var e = c.eligibility_decay * syn.eligibility[k];
            // LTP: post fires now, credit presynaptic activity that led up to it.
            if (nrn.fired[syn.target[k]]) e += nrn.pre_trace[syn.source[k]];
            // Optional LTD: pre fires now with a recently-active post -> depress.
            if (c.ltd_enabled and nrn.fired[syn.source[k]]) e -= nrn.post_trace[syn.target[k]];
            syn.eligibility[k] = e;
        }

        // Then advance the traces with this step's spikes.
        for (0..n) |i| {
            nrn.pre_trace[i] *= c.pre_trace_decay;
            nrn.post_trace[i] *= c.post_trace_decay;
            if (nrn.fired[i]) {
                nrn.pre_trace[i] += c.trace_increment;
                nrn.post_trace[i] += c.trace_increment;
            }
        }
    }

    /// Deliver a scalar reward and apply the third factor of the learning rule
    /// (DEC-009): for every plastic synapse, w += eta * (reward - baseline) * e.
    /// The baseline (an EMA of reward) makes the update zero-mean as accuracy
    /// rises, so weights stop drifting once the task is solved. Called once per
    /// episode by the training driver, at the "Update eligible synapses" step.
    /// Deterministic; no RNG.
    pub fn applyReward(self: *Sim, reward: f32) void {
        const c = self.network.config;
        if (!c.plasticity_enabled) return;
        const syn = &self.network.synapses;

        const modulator = reward - self.reward_baseline;
        for (0..syn.n) |k| {
            if (!syn.plastic[k]) continue;
            syn.weight[k] = std.math.clamp(
                syn.weight[k] + c.learning_rate * modulator * syn.eligibility[k],
                0.0,
                c.weight_max_plastic,
            );
        }

        self.reward_baseline = c.reward_baseline_decay * self.reward_baseline +
            (1.0 - c.reward_baseline_decay) * reward;
    }
};

fn sigmoid(x: f32) f32 {
    // Clamp before exp to avoid inf/NaN at extreme membrane values. The clamp
    // range is far outside anything a healthy network reaches, so if you ever
    // see it bind, the dynamics are already broken and you want to know.
    const z = std.math.clamp(x, -30.0, 30.0);
    return 1.0 / (1.0 + @exp(-z));
}

// ===========================================================================
// Tests -- these are the Phase 1 completion checklist (§16), as code.
// ===========================================================================

const testing = std.testing;

test "sim: same seed produces identical spike history" {
    const gpa = testing.allocator;
    const c = cfg.Config{ .master_seed = 1234, .steps = 300 };

    var a = try Sim.init(gpa, c);
    defer a.deinit(gpa);
    var b = try Sim.init(gpa, c);
    defer b.deinit(gpa);

    for (0..300) |_| {
        const ma = a.step(null);
        const mb = b.step(null);
        try testing.expectEqual(ma.spikes, mb.spikes);
        try testing.expectEqual(ma.exc_spikes, mb.exc_spikes);
        try testing.expectEqualSlices(bool, a.network.neurons.fired, b.network.neurons.fired);
        try testing.expectEqualSlices(f32, a.network.neurons.u, b.network.neurons.u);
    }
}

test "sim: excitatory spikes raise the target, inhibitory spikes lower it" {
    const gpa = testing.allocator;
    const c = cfg.Config{ .n_neurons = 2, .max_delay = 2 };

    var q = try EventQueue.init(gpa, c);
    defer q.deinit(gpa);

    var out = [_]f32{ 0, 0 };

    // An excitatory source: sign = +1.
    q.schedule(0, 1, 0, cfg.NeuronKind.excitatory.sign() * 2.0);
    // An inhibitory source: sign = -1.
    q.schedule(0, 1, 1, cfg.NeuronKind.inhibitory.sign() * 2.0);

    q.deliver(1, &out);
    try testing.expect(out[0] > 0);
    try testing.expect(out[1] < 0);
}

test "sim: membrane leaks toward rest (zero, rest-relative)" {
    const gpa = testing.allocator;
    // No input, no firing: u must decay geometrically toward 0.
    const c = cfg.Config{
        .n_neurons = 4,
        .background_current = 0,
        .connection_density = 0.5,
        .membrane_leak = 0.5,
        .threshold = 1000.0, // nothing can fire
        .adaptation_enabled = false,
    };

    var s = try Sim.init(gpa, c);
    defer s.deinit(gpa);

    // Push all neurons well away from rest.
    @memset(s.network.neurons.u, 8.0);

    var prev: f32 = 8.0;
    for (0..8) |_| {
        _ = s.step(null);
        const u = s.network.neurons.u[0];
        try testing.expect(u < prev); // strictly decreasing
        try testing.expect(u >= 0); // toward zero, not past it
        prev = u;
    }
    try testing.expect(prev < 0.1);
}

test "sim: refractory period prevents immediate refiring" {
    const gpa = testing.allocator;
    const c = cfg.Config{
        .n_neurons = 4,
        .refractory_steps = 3,
        .threshold = -100.0, // firing probability ~ 1 for any u
        .beta = 10.0,
        .background_current = 5.0,
        // Dense enough that 4 neurons almost always form a non-empty graph.
        // (0.01 left m=0 with high probability and failed Network.build.)
        .connection_density = 0.5,
        .uniform_graph = true,
    };

    var s = try Sim.init(gpa, c);
    defer s.deinit(gpa);

    // Step until neuron 0 fires.
    var fired_at: ?u32 = null;
    for (0..20) |_| {
        _ = s.step(null);
        if (s.network.neurons.fired[0]) {
            fired_at = s.t;
            break;
        }
    }
    try testing.expect(fired_at != null);

    // For the next `refractory_steps` steps it must NOT fire, no matter how
    // hard it is being driven.
    for (0..c.refractory_steps) |_| {
        _ = s.step(null);
        try testing.expect(!s.network.neurons.fired[0]);
    }
}

test "sim: delayed events arrive at exactly the expected timestep" {
    const gpa = testing.allocator;
    const c = cfg.Config{ .n_neurons = 1, .max_delay = 5 };

    var q = try EventQueue.init(gpa, c);
    defer q.deinit(gpa);

    var out = [_]f32{0};

    // Emitted at t=0 with delay 3 -> must arrive at t=3, and nowhere else.
    q.schedule(0, 3, 0, 1.0);

    for (0..8) |t| {
        q.deliver(@intCast(t), &out);
        if (t == 3) {
            try testing.expectApproxEqAbs(@as(f32, 1.0), out[0], 1e-6);
        } else {
            try testing.expectApproxEqAbs(@as(f32, 0.0), out[0], 1e-6);
        }
    }
}

test "sim: ring buffer wraps correctly over a long run" {
    const gpa = testing.allocator;
    const c = cfg.Config{ .n_neurons = 1, .max_delay = 3 };

    var q = try EventQueue.init(gpa, c);
    defer q.deinit(gpa);

    var out = [_]f32{0};

    // Schedule one event per step with delay 2, for 50 steps. Each must land
    // exactly two steps later, even as the ring wraps many times.
    for (0..50) |t| {
        const ti: u32 = @intCast(t);
        q.schedule(ti, 2, 0, 1.0);
        q.deliver(ti, &out);
        if (t >= 2) {
            try testing.expectApproxEqAbs(@as(f32, 1.0), out[0], 1e-6);
        } else {
            try testing.expectApproxEqAbs(@as(f32, 0.0), out[0], 1e-6);
        }
    }
}

test "sim: release probability is statistically correct at the synapse" {
    const gpa = testing.allocator;
    // One neuron driven to fire nearly every non-refractory step, p_release=0.25.
    // Count scheduled events vs. spikes * out_degree.
    const c = cfg.Config{
        .n_neurons = 60,
        .connection_density = 0.2,
        .release_probability = 0.25,
        .threshold = -50.0,
        .beta = 10.0,
        .background_current = 3.0,
        .refractory_steps = 0,
        .adaptation_enabled = false,
    };

    var s = try Sim.init(gpa, c);
    defer s.deinit(gpa);

    var total_scheduled: u64 = 0;
    var total_attempts: u64 = 0;

    for (0..500) |_| {
        const m = s.step(null);
        total_scheduled += m.scheduled_events;

        // Every spike attempts one release per outgoing synapse.
        for (0..s.network.neurons.n) |i| {
            if (!s.network.neurons.fired[i]) continue;
            const r = s.network.synapses.outgoing(@intCast(i));
            total_attempts += r.end - r.start;
        }
    }

    try testing.expect(total_attempts > 10_000); // enough samples to be meaningful
    const observed = @as(f64, @floatFromInt(total_scheduled)) / @as(f64, @floatFromInt(total_attempts));
    try testing.expect(@abs(observed - 0.25) < 0.02);
}

test "sim: activity neither dies nor explodes at default parameters" {
    // The Phase 1 exit criterion, as an assertion. If this fails you have not
    // found a bug -- you have found that the default parameters do not sit in a
    // healthy regime, which is EXP-001's whole job to establish.
    const gpa = testing.allocator;
    var s = try Sim.init(gpa, .{ .master_seed = 2026 });
    defer s.deinit(gpa);

    // Burn in.
    for (0..200) |_| _ = s.step(null);

    var total: u64 = 0;
    const window: u32 = 1000;
    for (0..window) |_| total += s.step(null).spikes;

    const rate = @as(f64, @floatFromInt(total)) /
        (@as(f64, @floatFromInt(window)) * @as(f64, @floatFromInt(s.network.neurons.n)));

    // Sparse but alive. Not dead (>0.1% per neuron per step), not saturated
    // (<20%). Wide on purpose -- this is a smoke test, not a target.
    try testing.expect(rate > 0.001);
    try testing.expect(rate < 0.20);
}

test "sim: inhibition manipulation visibly changes dynamics" {
    // The other Phase 1 exit criterion. If cranking inhibition does nothing,
    // your E/I sign convention is broken somewhere and every later result is junk.
    const gpa = testing.allocator;

    const measure = struct {
        fn f(alloc: Allocator, w_inh: f32) !f64 {
            var s = try Sim.init(alloc, .{
                .master_seed = 55,
                .w_inh_init_lo = w_inh,
                .w_inh_init_hi = w_inh * 2.0,
            });
            defer s.deinit(alloc);
            for (0..200) |_| _ = s.step(null);
            var total: u64 = 0;
            for (0..500) |_| total += s.step(null).spikes;
            return @as(f64, @floatFromInt(total)) / 500.0;
        }
    }.f;

    const weak_inh = try measure(gpa, 0.1);
    const strong_inh = try measure(gpa, 6.0);

    // More inhibition must mean fewer spikes. Not a subtle claim.
    try testing.expect(strong_inh < weak_inh);
}

test "sim: episode reset clears fast state but preserves weights" {
    // DEC-005, as a test. Cross-episode leakage of fast state is the classic
    // source of a local-learning result that isn't real.
    const gpa = testing.allocator;
    var s = try Sim.init(gpa, .{ .master_seed = 8 });
    defer s.deinit(gpa);

    const w_before = try gpa.dupe(f32, s.network.synapses.weight);
    defer gpa.free(w_before);
    const th_before = try gpa.dupe(f32, s.network.neurons.threshold);
    defer gpa.free(th_before);

    for (0..300) |_| _ = s.step(null);
    s.resetEpisode();

    // Fast state: cleared.
    for (s.network.neurons.u) |u| try testing.expectEqual(@as(f32, 0), u);
    for (s.network.neurons.refractory) |r| try testing.expectEqual(@as(u16, 0), r);
    for (s.network.neurons.fired) |f| try testing.expect(!f);
    for (s.queue.buckets) |b| {
        for (b) |x| try testing.expectEqual(@as(f32, 0), x);
    }
    try testing.expectEqual(@as(u32, 0), s.t);

    // Slow state: preserved.
    try testing.expectEqualSlices(f32, w_before, s.network.synapses.weight);
    try testing.expectEqualSlices(f32, th_before, s.network.neurons.threshold);
}

test "sim: firing probability stays in [0,1]" {
    // Including at absurd membrane values, where a naive sigmoid would produce
    // inf or NaN and silently poison the run.
    try testing.expect(sigmoid(1e9) <= 1.0);
    try testing.expect(sigmoid(-1e9) >= 0.0);
    try testing.expect(!std.math.isNan(sigmoid(1e9)));
    try testing.expect(!std.math.isNan(sigmoid(-1e9)));
    try testing.expectApproxEqAbs(@as(f32, 0.5), sigmoid(0), 1e-6);
}

test "sim: hard reset and subtractive reset differ" {
    // DEC-003 says this is a config flag and an ablation, not a rewrite.
    // Prove the flag actually does something.
    const gpa = testing.allocator;

    const meanRate = struct {
        fn f(alloc: Allocator, rule: cfg.ResetRule) !f64 {
            var s = try Sim.init(alloc, .{ .master_seed = 17, .reset_rule = rule });
            defer s.deinit(alloc);
            for (0..200) |_| _ = s.step(null);
            var total: u64 = 0;
            for (0..500) |_| total += s.step(null).spikes;
            return @as(f64, @floatFromInt(total)) / 500.0;
        }
    }.f;

    const soft = try meanRate(gpa, .subtractive);
    const hard = try meanRate(gpa, .hard);
    try testing.expect(soft != hard);
}

// ---- Phase 2: homeostasis (the completion checklist, as code) --------------

test "homeostasis: adaptive thresholds pull an over-active network toward target" {
    // Unregulated, this network sits near 0.14 spikes/neuron/step. Turn the
    // threshold homeostat on with a low target and it must drive the rate down
    // and the mean threshold up -- that is the controller doing its job.
    const gpa = testing.allocator;
    var s = try Sim.init(gpa, .{
        .master_seed = 2026,
        .homeostasis_enabled = true,
        .target_rate = 0.03,
        .homeostasis_lr = 0.05,
    });
    defer s.deinit(gpa);

    var last = StepMetrics{};
    for (0..6000) |_| last = s.step(null);

    // Regulated down into the neighbourhood of target (from ~0.14).
    try testing.expect(last.mean_rate_ema < 0.06);
    try testing.expect(last.mean_rate_ema > 0.01);
    // The controller moved: mean threshold rose well above the initial 1.0.
    try testing.expect(last.mean_threshold > 1.5);
}

test "homeostasis: network returns to target band after a sustained perturbation (Phase 2 exit criterion)" {
    // The exit criterion, as an assertion. Settle under homeostasis, then apply
    // a SUSTAINED moderate drive to every neuron. With homeostasis the rate must
    // return to the target band despite the drive; without it, the same drive
    // leaves the rate pinned above the band. The contrast is the proof.
    const gpa = testing.allocator;

    const base = cfg.Config{
        .master_seed = 0xB0A710,
        .n_neurons = 100,
        .target_rate = 0.05,
        .homeostasis_lr = 0.05,
    };

    const ext = try gpa.alloc(f32, base.n_neurons);
    defer gpa.free(ext);
    @memset(ext, 0.40); // sustained moderate perturbation

    const finalRate = struct {
        fn f(alloc: Allocator, c0: cfg.Config, homeo: bool, e: []const f32) !f32 {
            var c = c0;
            c.homeostasis_enabled = homeo;
            var s = try Sim.init(alloc, c);
            defer s.deinit(alloc);
            for (0..3000) |_| _ = s.step(null); // settle
            var last: f32 = 0;
            for (0..3000) |_| last = s.step(e).mean_rate_ema; // sustained perturb
            return last;
        }
    }.f;

    const band_lo = base.target_rate * 0.5;
    const band_hi = base.target_rate * 1.6;

    const on_final = try finalRate(gpa, base, true, ext);
    const off_final = try finalRate(gpa, base, false, ext);

    try testing.expect(on_final >= band_lo and on_final <= band_hi); // recovered
    try testing.expect(off_final > band_hi); // control stays out -> homeostasis is what recovered it
}

test "homeostasis: synaptic scaling shrinks excitatory inputs to over-active neurons, leaves inhibition alone" {
    // DEC-007. target_rate = 0 makes the direction unambiguous: every excitatory
    // input to a firing neuron is scaled down, silent neurons' inputs are left
    // as-is, so the excitatory total is non-increasing and strictly shrinks
    // wherever there is activity. Inhibitory synapses must be untouched.
    const gpa = testing.allocator;
    var s = try Sim.init(gpa, .{
        .master_seed = 3,
        .weight_normalization_enabled = true,
        .target_rate = 0.0,
        .weight_norm_lr = 0.01,
        .background_current = 1.0, // drive plenty of firing
        .threshold = 0.5,
    });
    defer s.deinit(gpa);

    const syn = &s.network.synapses;
    const before = try gpa.dupe(f32, syn.weight);
    defer gpa.free(before);

    var exc_before: f64 = 0;
    for (0..syn.n) |k| {
        if (s.network.neurons.kind[syn.source[k]] == .excitatory) exc_before += before[k];
    }

    for (0..800) |_| _ = s.step(null);

    var exc_after: f64 = 0;
    for (0..syn.n) |k| {
        try testing.expect(syn.weight[k] >= 0.0); // w >= 0 invariant survives scaling
        if (s.network.neurons.kind[syn.source[k]] == .excitatory) {
            exc_after += syn.weight[k];
        } else {
            try testing.expectEqual(before[k], syn.weight[k]); // inhibition untouched
        }
    }
    try testing.expect(exc_after < exc_before);
}

test "homeostasis: per-episode seam -- step() leaves thresholds alone when per_step is false; applyHomeostasis() regulates" {
    // The Phase 3 cadence: the episode driver runs steps with homeostasis_per_step
    // = false, then calls applyHomeostasis() once at the "Update homeostasis" step.
    // target_rate = 0 makes the direction unambiguous: any active neuron's
    // threshold must rise when the update is applied.
    const gpa = testing.allocator;
    var s = try Sim.init(gpa, .{
        .master_seed = 11,
        .homeostasis_enabled = true,
        .homeostasis_per_step = false, // caller controls cadence
        .target_rate = 0.0,
        .homeostasis_lr = 0.1,
    });
    defer s.deinit(gpa);

    // Build up a nonzero rate EMA WITHOUT per-step regulation.
    for (0..300) |_| _ = s.step(null);

    // step() must not have touched thresholds: they are still the initial 1.0.
    for (s.network.neurons.threshold) |th| try testing.expectEqual(@as(f32, 1.0), th);

    // Now the driver applies homeostasis explicitly, once.
    s.applyHomeostasis();

    // Every neuron that fired (rate_ema > 0) must have had its threshold raised.
    var any_raised = false;
    for (s.network.neurons.threshold, s.network.neurons.rate_ema) |th, r| {
        if (r > 0) {
            try testing.expect(th > 1.0);
            any_raised = true;
        }
    }
    try testing.expect(any_raised); // sanity: the network actually fired
}

test "homeostasis: with both homeostats off, weights are unchanged across a run" {
    // Guards the Phase 1 baseline: none of the Phase 2 machinery may touch
    // weights unless explicitly enabled.
    const gpa = testing.allocator;
    var s = try Sim.init(gpa, .{ .master_seed = 8 });
    defer s.deinit(gpa);

    const before = try gpa.dupe(f32, s.network.synapses.weight);
    defer gpa.free(before);

    for (0..500) |_| _ = s.step(null);
    try testing.expectEqualSlices(f32, before, s.network.synapses.weight);
}

// ---- Phase 3: local reward learning (the completion checklist, as code) -----

test "plasticity: eligibility tags a co-active synapse and reward moves its weight by the reward sign" {
    const task = @import("task.zig");
    const gpa = testing.allocator;
    const c = cfg.Config{
        .master_seed = 1,
        .task_enabled = true,
        .plasticity_enabled = true,
        .task_group_size = 8,
        .background_current = 0.6,
    };
    var s = try Sim.init(gpa, c);
    defer s.deinit(gpa);
    const l = task.layout(c);

    const ext = try gpa.alloc(f32, c.n_neurons);
    defer gpa.free(ext);
    l.fillStimulus(.a, c.task_input_current, ext);

    // Drive stimulus A: input_a fires and drives the action neurons, so the
    // input->action synapses build up positive eligibility.
    for (0..40) |_| _ = s.step(ext);

    var max_e: f32 = 0;
    for (0..s.network.synapses.n) |k| {
        if (s.network.synapses.plastic[k]) max_e = @max(max_e, s.network.synapses.eligibility[k]);
    }
    try testing.expect(max_e > 0); // eligibility actually accumulated

    // Positive reward must not decrease any eligible weight, and must raise at
    // least one (three-factor rule with a positive modulator).
    const before = try gpa.dupe(f32, s.network.synapses.weight);
    defer gpa.free(before);
    s.applyReward(1.0);

    var increased = false;
    for (0..s.network.synapses.n) |k| {
        try testing.expect(s.network.synapses.weight[k] >= 0.0); // w >= 0 invariant
        if (!s.network.synapses.plastic[k]) {
            try testing.expectEqual(before[k], s.network.synapses.weight[k]); // non-plastic untouched
            continue;
        }
        if (s.network.synapses.eligibility[k] > 0) {
            try testing.expect(s.network.synapses.weight[k] >= before[k]);
            if (s.network.synapses.weight[k] > before[k]) increased = true;
        }
    }
    try testing.expect(increased);
}

test "plasticity: with plasticity disabled, the task synapses exist but never change" {
    const task = @import("task.zig");
    const gpa = testing.allocator;
    const c = cfg.Config{
        .master_seed = 2,
        .task_enabled = true,
        .plasticity_enabled = false, // task present, learning off (the control)
        .task_group_size = 8,
        .background_current = 0.6,
    };
    var s = try Sim.init(gpa, c);
    defer s.deinit(gpa);
    const l = task.layout(c);

    // Nothing is plastic when learning is off.
    for (0..s.network.synapses.n) |k| try testing.expect(!s.network.synapses.plastic[k]);

    const before = try gpa.dupe(f32, s.network.synapses.weight);
    defer gpa.free(before);

    const ext = try gpa.alloc(f32, c.n_neurons);
    defer gpa.free(ext);
    l.fillStimulus(.a, c.task_input_current, ext);
    for (0..50) |_| _ = s.step(ext);
    s.applyReward(1.0); // must be a no-op when plasticity is disabled

    try testing.expectEqualSlices(f32, before, s.network.synapses.weight);
}

test "learning: the two-choice association is learned above chance across seeds (Phase 3 exit criterion)" {
    // The exit criterion, as an assertion. Deterministic, so the accuracies are
    // exact; the bounds sit far below the observed values (min ~0.82, mean ~0.95)
    // but far above chance (0.5). This is the full episode loop from the doc.
    const task = @import("task.zig");
    const gpa = testing.allocator;

    const seeds = [_]u64{ 1, 2, 3, 4 };
    const n_episodes: u32 = 1200;
    const stim_steps: u32 = 40;
    const readout_steps: u32 = 25;
    const final_window: u32 = 300;

    var sum_acc: f64 = 0;
    var min_acc: f64 = 1.0;

    for (seeds) |seed| {
        const c = cfg.Config{
            .master_seed = seed,
            .n_neurons = 100,
            .task_enabled = true,
            .plasticity_enabled = true,
            .homeostasis_enabled = true,
            .homeostasis_per_step = false,
            .target_rate = 0.05,
            .homeostasis_lr = 0.05,
            .task_group_size = 8,
        };
        var s = try Sim.init(gpa, c);
        defer s.deinit(gpa);
        const l = task.layout(c);
        const ext = try gpa.alloc(f32, c.n_neurons);
        defer gpa.free(ext);

        var correct_in_window: u32 = 0;
        for (0..n_episodes) |ep_usize| {
            const ep: u32 = @intCast(ep_usize);
            s.resetEpisode();

            var trng = rng.derived(seed, .task, ep);
            const choice: task.Choice = if (trng.below(2) == 0) .a else .b;
            l.fillStimulus(choice, c.task_input_current, ext);

            var count0: u32 = 0;
            var count1: u32 = 0;
            for (0..stim_steps) |step| {
                _ = s.step(ext);
                if (step >= stim_steps - readout_steps) {
                    const fired = s.network.neurons.fired;
                    for (l.action_0.lo..l.action_0.hi) |i| count0 += @intFromBool(fired[i]);
                    for (l.action_1.lo..l.action_1.hi) |i| count1 += @intFromBool(fired[i]);
                }
            }

            var chosen: u1 = undefined;
            if (count0 > count1) {
                chosen = 0;
            } else if (count1 > count0) {
                chosen = 1;
            } else {
                var arng = rng.derived(seed, .action, ep);
                chosen = @intCast(arng.below(2));
            }

            const correct = chosen == l.correctAction(choice);
            s.applyReward(if (correct) 1.0 else -1.0);
            s.applyHomeostasis();

            if (ep >= n_episodes - final_window and correct) correct_in_window += 1;
        }
        const acc = @as(f64, @floatFromInt(correct_in_window)) / @as(f64, @floatFromInt(final_window));
        sum_acc += acc;
        min_acc = @min(min_acc, acc);
    }
    const mean_acc = sum_acc / @as(f64, @floatFromInt(seeds.len));

    try testing.expect(min_acc > 0.65); // every seed clearly above chance
    try testing.expect(mean_acc > 0.75);
}

// ---- Phase 4: delayed learning / working memory -----------------------------
//
// The retention mechanism (stimulus-specific persistence) is an EMERGENT
// property of self-excitation + the homeostatic threshold tuning that training
// installs -- a fresh, untrained network with these knobs is globally saturated
// and shows no specificity. So the honest test of the mechanism is behavioural
// and end-to-end: the delayed association is learned above chance only because
// the assembly holds the stimulus across the delay. The memory-vs-reservoir A/B
// (the mechanism's necessity) is demonstrated in the delay harness, where at the
// target delay the reservoir alone decays to near chance.

test "learning: the delayed association is retained above chance across seeds (Phase 4 exit criterion)" {
    // The exit criterion: with a NONZERO delay, the network still learns the
    // association above chance. This only works because the input assembly holds
    // the stimulus through the delay (the previous test) -- with self-excitation
    // off, this same delay decays the reservoir to chance (shown in the delay
    // harness). Deterministic, so the accuracies are exact.
    const task = @import("task.zig");
    const gpa = testing.allocator;

    const seeds = [_]u64{ 1, 2 };
    const n_episodes: u32 = 1200;
    const stim_steps: u32 = 30;
    const delay_steps: u32 = 20; // nonzero, past the bare reservoir's fading memory
    const readout_steps: u32 = 20;
    const final_window: u32 = 300;

    var min_acc: f64 = 1.0;

    for (seeds) |seed| {
        const c = cfg.Config{
            .master_seed = seed,
            .n_neurons = 100,
            .task_enabled = true,
            .plasticity_enabled = true,
            .homeostasis_enabled = true,
            .homeostasis_per_step = false,
            .target_rate = 0.05,
            .homeostasis_lr = 0.05,
            .task_group_size = 8,
            .task_recurrent_weight = 0.5,
            .eligibility_decay = 0.95,
            .adaptation_enabled = false,
        };
        var s = try Sim.init(gpa, c);
        defer s.deinit(gpa);
        const l = task.layout(c);
        const ext = try gpa.alloc(f32, c.n_neurons);
        defer gpa.free(ext);

        var correct_in_window: u32 = 0;
        for (0..n_episodes) |ep_usize| {
            const ep: u32 = @intCast(ep_usize);
            s.resetEpisode();
            var trng = rng.derived(seed, .task, ep);
            const choice: task.Choice = if (trng.below(2) == 0) .a else .b;
            l.fillStimulus(choice, c.task_input_current, ext);

            for (0..stim_steps) |_| _ = s.step(ext);
            for (0..delay_steps) |_| _ = s.step(null);

            var count0: u32 = 0;
            var count1: u32 = 0;
            for (0..readout_steps) |_| {
                _ = s.step(null);
                const fired = s.network.neurons.fired;
                for (l.action_0.lo..l.action_0.hi) |i| count0 += @intFromBool(fired[i]);
                for (l.action_1.lo..l.action_1.hi) |i| count1 += @intFromBool(fired[i]);
            }

            var chosen: u1 = undefined;
            if (count0 > count1) {
                chosen = 0;
            } else if (count1 > count0) {
                chosen = 1;
            } else {
                var arng = rng.derived(seed, .action, ep);
                chosen = @intCast(arng.below(2));
            }

            const correct = chosen == l.correctAction(choice);
            s.applyReward(if (correct) 1.0 else -1.0);
            s.applyHomeostasis();
            if (ep >= n_episodes - final_window and correct) correct_in_window += 1;
        }
        const acc = @as(f64, @floatFromInt(correct_in_window)) / @as(f64, @floatFromInt(final_window));
        min_acc = @min(min_acc, acc);
    }

    try testing.expect(min_acc > 0.65); // retains across the delay, above chance, on every seed
}
