//! Stage 1 instrumentation (report.md): measure the niches where a local
//! three-factor spiking system is supposed to compete — not final accuracy alone.
//!
//! Four measurement tracks on the two-choice association (DEC-008):
//!
//!   1. ONLINE-UPDATE COST — local eligibility + reward ops vs a dense
//!      "touch every live synapse every step" baseline. The three-factor rule is
//!      O(n_plastic) at reward time and O(n_plastic) per step for eligibility;
//!      a dense BPTT-style update over the full graph is O(n_live · T).
//!   2. SPARSITY — activity (firing rate, active-neuron fraction) and plastic
//!      weight sparsity at the end of training.
//!   3. FORGETTING CURVES — train A to a mastery window, then force only B and
//!      probe frozen A-accuracy over block B, with consolidation ON vs OFF.
//!   4. DISTRIBUTION-SHIFT ADAPTATION — train the fixed A→0/B→1 mapping, flip the
//!      reward mapping mid-run (A→1/B→0), and measure post-shift recovery.
//!
//! Lesion resistance is already the load-bearing probe in `continual.zig`
//! (report.md §5); this harness cross-references that result rather than
//! re-running the 20-seed continual protocol.
//!
//! Outputs (all gitignored artefacts):
//!   instrument_cost.csv       — per-seed cost + sparsity summary
//!   instrument_forgetting.csv — A-retest curve during block B
//!   instrument_shift.csv      — accuracy curve around the mapping flip
//!   instrument.meta.json      — provenance
//!
//! Build/run:  zig build instrument -Doptimize=ReleaseFast

const std = @import("std");
const cfg = @import("config.zig");
const sim = @import("sim.zig");
const task = @import("task.zig");
const rng = @import("rng.zig");
const provenance = @import("provenance.zig");

// ---------------------------------------------------------------------------
// Protocol constants
// ---------------------------------------------------------------------------

const n_seeds = 8;
const seeds: [n_seeds]u64 = blk: {
    var s: [n_seeds]u64 = undefined;
    for (0..n_seeds) |i| s[i] = @as(u64, i) + 1;
    break :blk s;
};

const stim_steps: u32 = 40;
const readout_steps: u32 = 25;

// Track 1+2: cost + sparsity during ordinary training.
const train_episodes: u32 = 1200;
const cost_sample_every: u32 = 50; // sample cost/sparsity every N episodes
const final_window: u32 = 200;

// Track 3: forgetting curve (A → only-B, probe A).
const forget_block_a: u32 = 1400;
const forget_block_b: u32 = 800;
const forget_probe_every: u32 = 50;
const forget_retest: u32 = 40; // short frozen probe (not the continual 200)
const forget_acc_window: u32 = 200;
const forget_growth_interval: u32 = 50;
const forget_block_b_base: u32 = forget_block_a;
const forget_retest_base: u32 = forget_block_a + forget_block_b;
const consolidation_lr_on: f32 = 0.05;

// Track 4: distribution shift (mapping flip).
const shift_pre_episodes: u32 = 1000;
const shift_post_episodes: u32 = 1000;
const shift_block: u32 = 50; // report accuracy every block

// Approximate FLOP model for the three-factor local rule vs a dense baseline.
// These are *accounting* constants, not microbenchmarks — they make the O(·)
// comparison concrete in the CSV without timing noise.
const elig_flops_per_plastic_step: f64 = 4; // decay, conditional LTP add, store
const trace_flops_per_neuron_step: f64 = 4; // pre/post decay + optional bump
const reward_flops_per_plastic: f64 = 6; // modulator * e * eta, clamp
const dense_flops_per_live_step: f64 = 4; // "touch every live edge every step"
const dense_reward_flops_per_live: f64 = 6;

const weight_active_eps: f32 = 1e-3;
const eligible_eps: f32 = 1e-6;

// Soft sanity bands for the instrumentation summary (not exit criteria of a phase).
const sparsity_rate_lo: f64 = 0.005;
const sparsity_rate_hi: f64 = 0.20;
const cost_ratio_hi: f64 = 0.50; // local should be well under half of dense

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

fn trainConfig(seed: u64) cfg.Config {
    return .{
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
}

fn forgetConfig(seed: u64, consolidation_on: bool) cfg.Config {
    // Match continual.zig: consolidation_enabled stays TRUE so plastic readout
    // edges join the slow disuse/prune clock (DEC-012). The OFF arm only zeros
    // consolidation_lr — otherwise unused A pathways never decay and the curve
    // is ceiling-limited for both conditions.
    return .{
        .master_seed = seed,
        .n_neurons = 100,
        .task_enabled = true,
        .plasticity_enabled = true,
        .homeostasis_enabled = true,
        .homeostasis_per_step = false,
        .target_rate = 0.05,
        .homeostasis_lr = 0.05,
        .task_group_size = 8,
        .structural_plasticity_enabled = true,
        .growth_probability = 0.0,
        .consolidation_enabled = true,
        .consolidation_lr = if (consolidation_on) consolidation_lr_on else 0.0,
        .consolidation_use_centered_reward = false,
    };
}

fn choose(seed: u64, ep: u32, count0: u32, count1: u32) u1 {
    if (count0 > count1) return 0;
    if (count1 > count0) return 1;
    var arng = rng.derived(seed, .action, ep);
    return @intCast(arng.below(2));
}

/// Correct action under the original (false) or flipped (true) mapping.
fn correctAction(choice: task.Choice, flipped: bool) u1 {
    const base: u1 = switch (choice) {
        .a => 0,
        .b => 1,
    };
    return if (flipped) base ^ 1 else base;
}

const Present = struct {
    count0: u32,
    count1: u32,
    spikes: u64,
    steps: u32,
};

fn presentAndCount(s: *sim.Sim, l: task.Layout, ext: []f32) Present {
    var count0: u32 = 0;
    var count1: u32 = 0;
    var spikes: u64 = 0;
    for (0..stim_steps) |step| {
        const m = s.step(ext);
        spikes += m.spikes;
        if (step >= stim_steps - readout_steps) {
            const fired = s.network.neurons.fired;
            for (l.action_0.lo..l.action_0.hi) |i| count0 += @intFromBool(fired[i]);
            for (l.action_1.lo..l.action_1.hi) |i| count1 += @intFromBool(fired[i]);
        }
    }
    return .{ .count0 = count0, .count1 = count1, .spikes = spikes, .steps = stim_steps };
}

const EpOutcome = struct {
    correct: bool,
    spikes: u64,
    steps: u32,
};

fn runEpisode(
    s: *sim.Sim,
    l: task.Layout,
    ext: []f32,
    seed: u64,
    ep: u32,
    force_b: bool,
    flipped: bool,
    apply_learning: bool,
) EpOutcome {
    s.resetEpisode();
    const choice: task.Choice = if (force_b) .b else blk: {
        var trng = rng.derived(seed, .task, ep);
        break :blk if (trng.below(2) == 0) .a else .b;
    };
    const c = s.network.config;
    l.fillStimulus(choice, c.task_input_current, ext);
    const p = presentAndCount(s, l, ext);
    const chosen = choose(seed, ep, p.count0, p.count1);
    const correct = chosen == correctAction(choice, flipped);
    if (apply_learning) {
        s.applyReward(if (correct) 1.0 else -1.0);
        s.applyHomeostasis();
    }
    return .{ .correct = correct, .spikes = p.spikes, .steps = p.steps };
}

const PlasticStats = struct {
    n_plastic_alive: u32,
    n_eligible: u32,
    n_weight_active: u32,
    mean_weight: f64,
    n_live_all: u32,
};

fn plasticStats(s: *const sim.Sim) PlasticStats {
    const syn = &s.network.synapses;
    var n_plastic_alive: u32 = 0;
    var n_eligible: u32 = 0;
    var n_weight_active: u32 = 0;
    var n_live_all: u32 = 0;
    var wsum: f64 = 0;
    for (0..syn.n) |k| {
        if (!syn.alive[k]) continue;
        n_live_all += 1;
        if (!syn.plastic[k]) continue;
        n_plastic_alive += 1;
        wsum += syn.weight[k];
        if (@abs(syn.eligibility[k]) > eligible_eps) n_eligible += 1;
        if (syn.weight[k] > weight_active_eps) n_weight_active += 1;
    }
    return .{
        .n_plastic_alive = n_plastic_alive,
        .n_eligible = n_eligible,
        .n_weight_active = n_weight_active,
        .mean_weight = if (n_plastic_alive == 0) 0 else wsum / @as(f64, @floatFromInt(n_plastic_alive)),
        .n_live_all = n_live_all,
    };
}

fn meanRateEma(s: *const sim.Sim) f64 {
    const nrn = &s.network.neurons;
    var sum: f64 = 0;
    for (0..nrn.n) |i| sum += nrn.rate_ema[i];
    return sum / @as(f64, @floatFromInt(nrn.n));
}

/// Fraction of neurons whose rate EMA is at least half the configured target.
/// A tighter band than "ever slightly above zero", so the number actually
/// reflects population sparsity rather than near-universal weak activity.
fn activeNeuronFraction(s: *const sim.Sim) f64 {
    const nrn = &s.network.neurons;
    const floor = 0.5 * s.network.config.target_rate;
    var active: u32 = 0;
    for (0..nrn.n) |i| active += @intFromBool(nrn.rate_ema[i] >= floor);
    return @as(f64, @floatFromInt(active)) / @as(f64, @floatFromInt(nrn.n));
}

/// Accounting model: local three-factor cost over one episode of T steps.
fn localOps(n_plastic: u32, n_neurons: u32, t_steps: u32) f64 {
    const p = @as(f64, @floatFromInt(n_plastic));
    const n = @as(f64, @floatFromInt(n_neurons));
    const t = @as(f64, @floatFromInt(t_steps));
    return t * (p * elig_flops_per_plastic_step + n * trace_flops_per_neuron_step) +
        p * reward_flops_per_plastic;
}

/// Accounting model: dense "touch every live synapse every step" + full reward.
fn denseOps(n_live: u32, t_steps: u32) f64 {
    const l = @as(f64, @floatFromInt(n_live));
    const t = @as(f64, @floatFromInt(t_steps));
    return t * l * dense_flops_per_live_step + l * dense_reward_flops_per_live;
}

const Stat = struct {
    mean: f64,
    ci_half: f64,
    n: u32,
};

fn summarize(xs: []const f64) Stat {
    if (xs.len == 0) return .{ .mean = 0, .ci_half = std.math.nan(f64), .n = 0 };
    var sum: f64 = 0;
    for (xs) |x| sum += x;
    const mean = sum / @as(f64, @floatFromInt(xs.len));
    if (xs.len < 2) return .{ .mean = mean, .ci_half = std.math.nan(f64), .n = @intCast(xs.len) };
    var ss: f64 = 0;
    for (xs) |x| ss += (x - mean) * (x - mean);
    const sd = @sqrt(ss / @as(f64, @floatFromInt(xs.len - 1)));
    const ci_half = 1.96 * sd / @sqrt(@as(f64, @floatFromInt(xs.len)));
    return .{ .mean = mean, .ci_half = ci_half, .n = @intCast(xs.len) };
}

fn writeAtomic(io: std.Io, path: []const u8, bytes: []const u8) !void {
    var atomic = try std.Io.Dir.cwd().createFileAtomic(io, path, .{ .replace = true });
    defer atomic.deinit(io);
    var buf: [4096]u8 = undefined;
    var file_writer = atomic.file.writer(io, &buf);
    try file_writer.interface.writeAll(bytes);
    try file_writer.interface.flush();
    try atomic.replace(io);
}

// ---------------------------------------------------------------------------
// Track 1+2: cost + sparsity
// ---------------------------------------------------------------------------

const CostRow = struct {
    seed: u64,
    final_accuracy: f64,
    mean_firing_rate: f64,
    active_neuron_frac: f64,
    n_plastic_alive: u32,
    n_live_all: u32,
    mean_eligible: f64,
    plastic_weight_active_frac: f64,
    mean_plastic_weight: f64,
    local_ops: f64,
    dense_ops: f64,
    cost_ratio: f64,
};

fn runCostSparsity(gpa: std.mem.Allocator, seed: u64) !CostRow {
    const c = trainConfig(seed);
    try c.validate();
    var s = try sim.Sim.init(gpa, c);
    defer s.deinit(gpa);
    const l = task.layout(c);
    const ext = try gpa.alloc(f32, c.n_neurons);
    defer gpa.free(ext);

    var final_correct: u32 = 0;
    var spike_sum: u64 = 0;
    var step_sum: u64 = 0;
    var eligible_sum: f64 = 0;
    var eligible_samples: u32 = 0;
    var last_stats: PlasticStats = .{
        .n_plastic_alive = 0,
        .n_eligible = 0,
        .n_weight_active = 0,
        .mean_weight = 0,
        .n_live_all = 0,
    };

    for (0..train_episodes) |ep| {
        const r = runEpisode(&s, l, ext, seed, @intCast(ep), false, false, true);
        if (ep >= train_episodes - final_window) {
            final_correct += @intFromBool(r.correct);
            spike_sum += r.spikes;
            step_sum += r.steps;
        }
        if ((ep + 1) % cost_sample_every == 0 or ep + 1 == train_episodes) {
            last_stats = plasticStats(&s);
            eligible_sum += @as(f64, @floatFromInt(last_stats.n_eligible));
            eligible_samples += 1;
        }
    }

    const nf = @as(f64, @floatFromInt(c.n_neurons));
    const mean_rate = if (step_sum == 0)
        0
    else
        @as(f64, @floatFromInt(spike_sum)) / (@as(f64, @floatFromInt(step_sum)) * nf);

    const local = localOps(last_stats.n_plastic_alive, c.n_neurons, stim_steps);
    const dense = denseOps(last_stats.n_live_all, stim_steps);
    const ratio = if (dense == 0) 0 else local / dense;
    const w_active_frac = if (last_stats.n_plastic_alive == 0)
        0
    else
        @as(f64, @floatFromInt(last_stats.n_weight_active)) /
            @as(f64, @floatFromInt(last_stats.n_plastic_alive));

    return .{
        .seed = seed,
        .final_accuracy = @as(f64, @floatFromInt(final_correct)) / @as(f64, @floatFromInt(final_window)),
        .mean_firing_rate = mean_rate,
        .active_neuron_frac = activeNeuronFraction(&s),
        .n_plastic_alive = last_stats.n_plastic_alive,
        .n_live_all = last_stats.n_live_all,
        .mean_eligible = if (eligible_samples == 0) 0 else eligible_sum / @as(f64, @floatFromInt(eligible_samples)),
        .plastic_weight_active_frac = w_active_frac,
        .mean_plastic_weight = last_stats.mean_weight,
        .local_ops = local,
        .dense_ops = dense,
        .cost_ratio = ratio,
    };
}

// ---------------------------------------------------------------------------
// Track 3: forgetting curve
// ---------------------------------------------------------------------------

fn retestA(s: *sim.Sim, l: task.Layout, ext: []f32, seed: u64, base_ep: u32, n: u32) f64 {
    const c = s.network.config;
    var correct: u32 = 0;
    for (0..n) |e| {
        const ep: u32 = base_ep + @as(u32, @intCast(e));
        s.resetEpisode();
        l.fillStimulus(.a, c.task_input_current, ext);
        const p = presentAndCount(s, l, ext);
        const chosen = choose(seed, ep, p.count0, p.count1);
        if (chosen == correctAction(.a, false)) correct += 1;
    }
    return @as(f64, @floatFromInt(correct)) / @as(f64, @floatFromInt(n));
}

fn runForgetting(
    gpa: std.mem.Allocator,
    seed: u64,
    consolidation_on: bool,
    csv: *std.Io.Writer.Allocating,
) !struct { acc_a: f64, retest_end: f64 } {
    const c = forgetConfig(seed, consolidation_on);
    try c.validate();
    var s = try sim.Sim.init(gpa, c);
    defer s.deinit(gpa);
    const l = task.layout(c);
    const ext = try gpa.alloc(f32, c.n_neurons);
    defer gpa.free(ext);

    // Block A: full two-choice training (fixed length so curves align across seeds).
    var ring_correct: u32 = 0;
    const ring = try gpa.alloc(bool, forget_acc_window);
    defer gpa.free(ring);
    @memset(ring, false);
    var filled: u32 = 0;
    var idx: u32 = 0;
    for (0..forget_block_a) |ep| {
        const r = runEpisode(&s, l, ext, seed, @intCast(ep), false, false, true);
        if ((ep + 1) % forget_growth_interval == 0) _ = s.applyStructuralPlasticity();
        if (filled == forget_acc_window) {
            ring_correct -= @as(u32, @intFromBool(ring[idx]));
        } else {
            filled += 1;
        }
        ring[idx] = r.correct;
        ring_correct += @as(u32, @intFromBool(r.correct));
        idx = (idx + 1) % forget_acc_window;
    }
    const acc_a = @as(f64, @floatFromInt(ring_correct)) / @as(f64, @floatFromInt(forget_acc_window));

    // Probe at block-B start (episode 0 of disuse).
    {
        const acc = retestA(&s, l, ext, seed, forget_retest_base, forget_retest);
        try csv.writer.print("{d},{s},{d},{d:.4}\n", .{
            seed,
            if (consolidation_on) "consolidation_on" else "consolidation_off",
            @as(u32, 0),
            acc,
        });
    }

    var retest_end: f64 = 0;
    for (0..forget_block_b) |b| {
        const bep: u32 = forget_block_b_base + @as(u32, @intCast(b));
        _ = runEpisode(&s, l, ext, seed, bep, true, false, true);
        if ((bep + 1) % forget_growth_interval == 0) _ = s.applyStructuralPlasticity();
        if ((b + 1) % forget_probe_every == 0 or b + 1 == forget_block_b) {
            // Offset retest RNG range so probes don't collide with training eps.
            const probe_base = forget_retest_base + @as(u32, @intCast(b + 1)) * forget_retest;
            retest_end = retestA(&s, l, ext, seed, probe_base, forget_retest);
            try csv.writer.print("{d},{s},{d},{d:.4}\n", .{
                seed,
                if (consolidation_on) "consolidation_on" else "consolidation_off",
                @as(u32, @intCast(b + 1)),
                retest_end,
            });
        }
    }
    return .{ .acc_a = acc_a, .retest_end = retest_end };
}

// ---------------------------------------------------------------------------
// Track 4: distribution shift
// ---------------------------------------------------------------------------

const ShiftRow = struct {
    seed: u64,
    pre_acc: f64,
    drop_acc: f64, // first block after flip
    post_acc: f64, // final block after re-adaptation
    episodes_to_recover: ?u32, // first block mean >= 0.70 after flip, if any
};

fn runShift(gpa: std.mem.Allocator, seed: u64, csv: *std.Io.Writer.Allocating) !ShiftRow {
    const c = trainConfig(seed);
    try c.validate();
    var s = try sim.Sim.init(gpa, c);
    defer s.deinit(gpa);
    const l = task.layout(c);
    const ext = try gpa.alloc(f32, c.n_neurons);
    defer gpa.free(ext);

    var block_correct: u32 = 0;
    var pre_acc: f64 = 0;
    var drop_acc: f64 = 0;
    var post_acc: f64 = 0;
    var episodes_to_recover: ?u32 = null;
    const total = shift_pre_episodes + shift_post_episodes;

    for (0..total) |ep| {
        const flipped = ep >= shift_pre_episodes;
        const r = runEpisode(&s, l, ext, seed, @intCast(ep), false, flipped, true);
        block_correct += @intFromBool(r.correct);
        if ((ep + 1) % shift_block == 0) {
            const acc = @as(f64, @floatFromInt(block_correct)) / @as(f64, @floatFromInt(shift_block));
            const phase: []const u8 = if (flipped) "post_shift" else "pre_shift";
            try csv.writer.print("{d},{s},{d},{d:.4}\n", .{ seed, phase, ep + 1, acc });
            if (!flipped) {
                pre_acc = acc; // last pre-shift block
            } else {
                const post_ep: u32 = @intCast(ep + 1 - shift_pre_episodes);
                if (post_ep == shift_block) drop_acc = acc;
                post_acc = acc;
                if (episodes_to_recover == null and acc >= 0.70) {
                    episodes_to_recover = post_ep;
                }
            }
            block_correct = 0;
        }
    }

    return .{
        .seed = seed,
        .pre_acc = pre_acc,
        .drop_acc = drop_acc,
        .post_acc = post_acc,
        .episodes_to_recover = episodes_to_recover,
    };
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;
    var stdout_buf: [4096]u8 = undefined;
    var stdout = std.Io.File.stdout().writer(io, &stdout_buf);
    const o = &stdout.interface;

    var cost_csv: std.Io.Writer.Allocating = .init(gpa);
    defer cost_csv.deinit();
    try cost_csv.writer.print(
        "seed,final_accuracy,mean_firing_rate,active_neuron_frac,n_plastic_alive,n_live_all,mean_eligible,plastic_weight_active_frac,mean_plastic_weight,local_ops,dense_ops,cost_ratio\n",
        .{},
    );

    var forget_csv: std.Io.Writer.Allocating = .init(gpa);
    defer forget_csv.deinit();
    try forget_csv.writer.print("seed,condition,block_b_episode,retest_a\n", .{});

    var shift_csv: std.Io.Writer.Allocating = .init(gpa);
    defer shift_csv.deinit();
    try shift_csv.writer.print("seed,phase,episode,block_accuracy\n", .{});

    var cost_rows: [n_seeds]CostRow = undefined;
    var shift_rows: [n_seeds]ShiftRow = undefined;
    var forget_on_end: [n_seeds]f64 = undefined;
    var forget_off_end: [n_seeds]f64 = undefined;
    var forget_on_a: [n_seeds]f64 = undefined;
    var forget_off_a: [n_seeds]f64 = undefined;

    try o.print("\n-- Stage 1 instrumentation (cost / sparsity / forgetting / shift) --\n\n", .{});

    // ---- Track 1+2 -------------------------------------------------------
    try o.print("  [1/3] online-update cost + sparsity  ({d} seeds × {d} eps)\n", .{ n_seeds, train_episodes });
    for (seeds, 0..) |seed, i| {
        const row = try runCostSparsity(gpa, seed);
        cost_rows[i] = row;
        try cost_csv.writer.print(
            "{d},{d:.4},{d:.5},{d:.4},{d},{d},{d:.2},{d:.4},{d:.4},{d:.0},{d:.0},{d:.4}\n",
            .{
                row.seed,               row.final_accuracy,             row.mean_firing_rate,
                row.active_neuron_frac, row.n_plastic_alive,            row.n_live_all,
                row.mean_eligible,      row.plastic_weight_active_frac, row.mean_plastic_weight,
                row.local_ops,          row.dense_ops,                  row.cost_ratio,
            },
        );
        try o.print(
            "    seed {d}: acc {d:.3}  rate {d:.4}  active {d:.2}  cost_ratio {d:.3}  (plastic {d}/{d} live)\n",
            .{
                seed,           row.final_accuracy,  row.mean_firing_rate, row.active_neuron_frac,
                row.cost_ratio, row.n_plastic_alive, row.n_live_all,
            },
        );
    }

    // ---- Track 3 ---------------------------------------------------------
    try o.print("\n  [2/3] forgetting curves  (consolidation on/off × {d} seeds)\n", .{n_seeds});
    for (seeds, 0..) |seed, i| {
        const on = try runForgetting(gpa, seed, true, &forget_csv);
        const off = try runForgetting(gpa, seed, false, &forget_csv);
        forget_on_a[i] = on.acc_a;
        forget_off_a[i] = off.acc_a;
        forget_on_end[i] = on.retest_end;
        forget_off_end[i] = off.retest_end;
        try o.print(
            "    seed {d}: A-end on {d:.3}/off {d:.3}  →  A-retest after B on {d:.3}/off {d:.3}\n",
            .{ seed, on.acc_a, off.acc_a, on.retest_end, off.retest_end },
        );
    }

    // ---- Track 4 ---------------------------------------------------------
    try o.print("\n  [3/3] distribution shift  (mapping flip at ep {d})\n", .{shift_pre_episodes});
    for (seeds, 0..) |seed, i| {
        const row = try runShift(gpa, seed, &shift_csv);
        shift_rows[i] = row;
        if (row.episodes_to_recover) |e| {
            try o.print(
                "    seed {d}: pre {d:.3} → drop {d:.3} → post {d:.3}  (recover@ {d})\n",
                .{ seed, row.pre_acc, row.drop_acc, row.post_acc, e },
            );
        } else {
            try o.print(
                "    seed {d}: pre {d:.3} → drop {d:.3} → post {d:.3}  (no recover ≥0.70)\n",
                .{ seed, row.pre_acc, row.drop_acc, row.post_acc },
            );
        }
    }

    try writeAtomic(io, "instrument_cost.csv", cost_csv.written());
    try writeAtomic(io, "instrument_forgetting.csv", forget_csv.written());
    try writeAtomic(io, "instrument_shift.csv", shift_csv.written());

    // Aggregate stats
    var accs: [n_seeds]f64 = undefined;
    var rates: [n_seeds]f64 = undefined;
    var actives: [n_seeds]f64 = undefined;
    var ratios: [n_seeds]f64 = undefined;
    var pre: [n_seeds]f64 = undefined;
    var drop: [n_seeds]f64 = undefined;
    var post: [n_seeds]f64 = undefined;
    var forget_gap: [n_seeds]f64 = undefined; // on − off end retest (positive => consolidation helps)
    for (0..n_seeds) |i| {
        accs[i] = cost_rows[i].final_accuracy;
        rates[i] = cost_rows[i].mean_firing_rate;
        actives[i] = cost_rows[i].active_neuron_frac;
        ratios[i] = cost_rows[i].cost_ratio;
        pre[i] = shift_rows[i].pre_acc;
        drop[i] = shift_rows[i].drop_acc;
        post[i] = shift_rows[i].post_acc;
        forget_gap[i] = forget_on_end[i] - forget_off_end[i];
    }
    const s_acc = summarize(&accs);
    const s_rate = summarize(&rates);
    const s_active = summarize(&actives);
    const s_ratio = summarize(&ratios);
    const s_pre = summarize(&pre);
    const s_drop = summarize(&drop);
    const s_post = summarize(&post);
    const s_forget_on = summarize(&forget_on_end);
    const s_forget_off = summarize(&forget_off_end);
    const s_forget_gap = summarize(&forget_gap);

    // Plastic/live from first seed as structural constants (topology identical across seeds
    // for task edges; live reservoir count is seed-dependent — report mean from cost rows).
    var live_sum: f64 = 0;
    var plastic_sum: f64 = 0;
    for (cost_rows) |r| {
        live_sum += @floatFromInt(r.n_live_all);
        plastic_sum += @floatFromInt(r.n_plastic_alive);
    }
    const mean_live = live_sum / @as(f64, @floatFromInt(n_seeds));
    const mean_plastic = plastic_sum / @as(f64, @floatFromInt(n_seeds));

    try provenance.write(io, gpa, "instrument.meta.json", "stage1_instrumentation", .{
        .seeds = seeds,
        .n_seeds = n_seeds,
        .train_config = trainConfig(seeds[0]),
        .forget_config_on = forgetConfig(seeds[0], true),
        .forget_config_off = forgetConfig(seeds[0], false),
        .train_episodes = train_episodes,
        .stim_steps = stim_steps,
        .readout_steps = readout_steps,
        .forget_block_a = forget_block_a,
        .forget_block_b = forget_block_b,
        .forget_probe_every = forget_probe_every,
        .shift_pre_episodes = shift_pre_episodes,
        .shift_post_episodes = shift_post_episodes,
        .flop_model = .{
            .elig_flops_per_plastic_step = elig_flops_per_plastic_step,
            .trace_flops_per_neuron_step = trace_flops_per_neuron_step,
            .reward_flops_per_plastic = reward_flops_per_plastic,
            .dense_flops_per_live_step = dense_flops_per_live_step,
            .dense_reward_flops_per_live = dense_reward_flops_per_live,
        },
        .lesion_resistance = "see continual.zig / findings.md Phase 6 (pathway lesion)",
    });

    const rate_ok = s_rate.mean >= sparsity_rate_lo and s_rate.mean <= sparsity_rate_hi;
    const cost_ok = s_ratio.mean < cost_ratio_hi;
    const shift_adapts = s_post.mean > s_drop.mean + 0.05;
    const complete = rate_ok and cost_ok; // measurement integrity; adaptation is reported not gated

    try o.print(
        \\
        \\  ========== summary (mean ± 95% CI half, {d} seeds) ==========
        \\
        \\  ONLINE-UPDATE COST  (accounting model, ops / episode)
        \\    plastic / live synapses   {d:.0} / {d:.0}
        \\    local / dense ratio       {d:.3} ± {d:.3}   (soft band: < {d:.2})
        \\    interpretation            eligibility+reward touch plastic edges only;
        \\                              dense baseline touches every live edge each step
        \\
        \\  SPARSITY
        \\    mean firing rate          {d:.4} ± {d:.4}   (soft band: [{d:.3}, {d:.2}])
        \\    active neuron frac        {d:.3} ± {d:.3}
        \\    final accuracy (ref)      {d:.3} ± {d:.3}
        \\
        \\  FORGETTING CURVES  (A-retest after block-B disuse)
        \\    consolidation ON          {d:.3} ± {d:.3}
        \\    consolidation OFF         {d:.3} ± {d:.3}
        \\    gap (on − off)            {d:.3} ± {d:.3}
        \\    time series               instrument_forgetting.csv
        \\
        \\  DISTRIBUTION SHIFT  (mapping flip A↔B at ep {d})
        \\    pre-shift accuracy        {d:.3} ± {d:.3}
        \\    first post-shift block    {d:.3} ± {d:.3}
        \\    final post-shift block    {d:.3} ± {d:.3}
        \\    re-adaptation visible     {s}
        \\
        \\  LESION RESISTANCE
        \\    covered by continual.zig (pathway lesion under consolidation);
        \\    see findings.md Phase 6 — not re-run here.
        \\
        \\  STATUS: {s}
        \\  wrote instrument_cost.csv, instrument_forgetting.csv, instrument_shift.csv
        \\
    , .{
        n_seeds,
        mean_plastic,
        mean_live,
        s_ratio.mean,
        s_ratio.ci_half,
        cost_ratio_hi,
        s_rate.mean,
        s_rate.ci_half,
        sparsity_rate_lo,
        sparsity_rate_hi,
        s_active.mean,
        s_active.ci_half,
        s_acc.mean,
        s_acc.ci_half,
        s_forget_on.mean,
        s_forget_on.ci_half,
        s_forget_off.mean,
        s_forget_off.ci_half,
        s_forget_gap.mean,
        s_forget_gap.ci_half,
        shift_pre_episodes,
        s_pre.mean,
        s_pre.ci_half,
        s_drop.mean,
        s_drop.ci_half,
        s_post.mean,
        s_post.ci_half,
        if (shift_adapts) "yes" else "weak / none",
        if (complete)
            "COMPLETE — cost, sparsity, forgetting, and shift artefacts written."
        else
            "COMPLETE WITH WARNINGS — inspect soft bands above; CSVs still written.",
    });
    try o.flush();
}
