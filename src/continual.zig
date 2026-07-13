//! Phase 6 exit-criterion experiment: does CONSOLIDATION let previously-useful
//! pathways survive a period of disuse better than unused tentative pathways?
//!
//! Exit criterion (spec, Phase 6):
//!   "Previously useful pathways survive better than unused tentative pathways."
//!
//! Continual-learning protocol (train A, then B, then retest A):
//!   * Block A  -- train the full two-choice task (A->0, B->1). The correct
//!                 readout synapses become useful and, under consolidation, ratchet
//!                 up their permanence (reward-gated, DEC-012).
//!   * Block B  -- present ONLY stimulus B. The A pathway (input_a->action_0) now
//!                 goes UNUSED; the slow structural clock decays and prunes unused,
//!                 unconsolidated pathways. Consolidated ones resist.
//!   * Retest A -- frozen (no learning): present stimulus A, measure accuracy.
//!
//! Two conditions, identical but for the one mechanism under test:
//!   consolidation ON  (consolidation_lr > 0): rewarded pathways consolidate.
//!   consolidation OFF (consolidation_lr = 0): same decay/prune, no consolidation.
//!
//! The verdict ANDs the two faces of the criterion:
//!   1. SURVIVAL   -- with consolidation on, the previously-useful A pathway
//!                    (input_a->action_0) retains far more weight through block B
//!                    than the unused tentative pathway (input_a->action_1).
//!   2. LESS FORGETTING -- A-retest accuracy is higher with consolidation than
//!                    without (the mechanism is what preserved the pathway).
//!
//! Produces continual.csv (per seed/condition: A acc, retest acc, pathway weights)
//! and a PASS/FAIL verdict.
//!
//! Build/run:  zig build continual   (add -Doptimize=ReleaseFast)

const std = @import("std");
const cfg = @import("config.zig");
const sim = @import("sim.zig");
const task = @import("task.zig");
const rng = @import("rng.zig");

const seeds = [_]u64{ 1, 2, 3, 4 };
const block_a_episodes: u32 = 1400; // train the full task (past the exploration plateau)
const block_b_episodes: u32 = 800; // present only B; A pathway goes unused
const retest_episodes: u32 = 200; // frozen accuracy probe on stimulus A
const stim_steps: u32 = 40;
const readout_steps: u32 = 25;
const acc_window: u32 = 300;
const growth_interval: u32 = 50; // structural (decay/prune) window cadence

const consolidation_lr_on: f32 = 0.05;

// Verdict thresholds.
const pass_survival_margin: f64 = 0.30; // useful-A alive fraction − unused, with consolidation
const pass_retest_margin: f64 = 0.08; // A-retest accuracy: ON − OFF

fn baseConfig(seed: u64, consolidate: bool) cfg.Config {
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
        // Phase 5 slow clock, but no GROWTH -- we isolate consolidation of the
        // learned readout, not reservoir rewiring.
        .structural_plasticity_enabled = true,
        .growth_probability = 0.0,
        // Phase 6:
        .consolidation_enabled = true,
        .consolidation_lr = if (consolidate) consolidation_lr_on else 0.0,
    };
}

/// Read out the network's choice this episode from action-group spike counts.
fn choose(s: *sim.Sim, l: task.Layout, seed: u64, ep: u32, count0: u32, count1: u32) u1 {
    _ = s;
    _ = l;
    if (count0 > count1) return 0;
    if (count1 > count0) return 1;
    var arng = rng.derived(seed, .action, ep);
    return @intCast(arng.below(2));
}

/// One training episode with a fixed choice policy. `force_b` presents stimulus B
/// every time (block B); otherwise the derived task RNG picks A or B (block A).
/// Applies reward + homeostasis; the caller runs the structural window.
fn trainEpisode(s: *sim.Sim, l: task.Layout, ext: []f32, seed: u64, ep: u32, force_b: bool) bool {
    s.resetEpisode();
    const choice: task.Choice = if (force_b) .b else blk: {
        var trng = rng.derived(seed, .task, ep);
        break :blk if (trng.below(2) == 0) .a else .b;
    };
    const c = s.network.config;
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
    const chosen = choose(s, l, seed, ep, count0, count1);
    const correct = chosen == l.correctAction(choice);
    s.applyReward(if (correct) 1.0 else -1.0);
    s.applyHomeostasis();
    return correct;
}

/// Frozen probe: present stimulus A, no reward/homeostasis/structural update.
/// Returns accuracy of choosing action_0 over `retest_episodes`.
fn retestA(s: *sim.Sim, l: task.Layout, ext: []f32, seed: u64) f64 {
    const c = s.network.config;
    var correct: u32 = 0;
    for (0..retest_episodes) |e| {
        const ep: u32 = @intCast(block_a_episodes + block_b_episodes + e);
        s.resetEpisode();
        l.fillStimulus(.a, c.task_input_current, ext);
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
        const chosen = choose(s, l, seed, ep, count0, count1);
        if (chosen == l.correctAction(.a)) correct += 1;
    }
    return @as(f64, @floatFromInt(correct)) / @as(f64, @floatFromInt(retest_episodes));
}

// Permanence bands (§8.3: the tentative/established/consolidated state emerges
// from thresholds over permanence, it need not be an enum). We classify each
// plastic readout synapse by its permanence at the END OF BLOCK A, then ask how
// each band SURVIVES block B's disuse. "Consolidated" == previously useful (its
// permanence was ratcheted up by reward); "tentative" == weak, never consolidated.
const consolidated_q: f32 = 0.6;
const tentative_q: f32 = 0.4;

const Result = struct {
    acc_a: f64, // accuracy at end of block A
    retest: f64, // A-retest accuracy after block B
    n_consolidated: u32, // plastic synapses in the consolidated band after block A
    n_tentative: u32, // plastic synapses in the tentative band after block A
    consolidated_survival: f64, // fraction of the consolidated band still alive after block B
    tentative_survival: f64, // fraction of the tentative band still alive after block B
};

fn runCondition(gpa: std.mem.Allocator, seed: u64, consolidate: bool) !Result {
    const c = baseConfig(seed, consolidate);
    try c.validate();
    var s = try sim.Sim.init(gpa, c);
    defer s.deinit(gpa);
    const l = task.layout(c);
    const ext = try gpa.alloc(f32, c.n_neurons);
    defer gpa.free(ext);

    // Block A: train the full task.
    var correct_a: u32 = 0;
    for (0..block_a_episodes) |ep_usize| {
        const ep: u32 = @intCast(ep_usize);
        const ok = trainEpisode(&s, l, ext, seed, ep, false);
        if ((ep + 1) % growth_interval == 0) _ = s.applyStructuralPlasticity();
        if (ep_usize >= block_a_episodes - acc_window and ok) correct_a += 1;
    }
    const acc_a = @as(f64, @floatFromInt(correct_a)) / @as(f64, @floatFromInt(acc_window));

    // Snapshot the permanence bands (§8.3) at the end of block A: tag each alive
    // plastic synapse as consolidated / tentative / neither, so we can measure
    // how each band survives the coming disuse.
    const syn = &s.network.synapses;
    const band = try gpa.alloc(u8, syn.n); // 0 neither, 1 consolidated, 2 tentative
    defer gpa.free(band);
    @memset(band, 0);
    var n_cons: u32 = 0;
    var n_tent: u32 = 0;
    for (0..syn.n) |k| {
        if (!syn.plastic[k] or !syn.alive[k]) continue;
        if (syn.permanence[k] >= consolidated_q) {
            band[k] = 1;
            n_cons += 1;
        } else if (syn.permanence[k] <= tentative_q) {
            band[k] = 2;
            n_tent += 1;
        }
    }

    // Block B: present only stimulus B -- the A pathway goes unused, so the slow
    // clock decays and prunes whatever is not consolidated.
    for (0..block_b_episodes) |ep_usize| {
        const ep: u32 = @intCast(block_a_episodes + ep_usize);
        _ = trainEpisode(&s, l, ext, seed, ep, true);
        if ((ep + 1) % growth_interval == 0) _ = s.applyStructuralPlasticity();
    }

    // Survival of each band: fraction of its members still alive after block B.
    var cons_alive: u32 = 0;
    var tent_alive: u32 = 0;
    for (0..syn.n) |k| {
        switch (band[k]) {
            1 => cons_alive += @intFromBool(syn.alive[k]),
            2 => tent_alive += @intFromBool(syn.alive[k]),
            else => {},
        }
    }
    const retest = retestA(&s, l, ext, seed);

    return .{
        .acc_a = acc_a,
        .retest = retest,
        .n_consolidated = n_cons,
        .n_tentative = n_tent,
        .consolidated_survival = if (n_cons == 0) 0 else @as(f64, @floatFromInt(cons_alive)) / @as(f64, @floatFromInt(n_cons)),
        .tentative_survival = if (n_tent == 0) 1 else @as(f64, @floatFromInt(tent_alive)) / @as(f64, @floatFromInt(n_tent)),
    };
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    var buf: [4096]u8 = undefined;
    var stdout = std.Io.File.stdout().writer(io, &buf);
    const o = &stdout.interface;

    var csv: std.Io.Writer.Allocating = .init(gpa);
    defer csv.deinit();
    try csv.writer.print("seed,consolidation,acc_a,retest_a,n_consolidated,consolidated_survival,n_tentative,tentative_survival\n", .{});

    try o.print("\n-- Phase 6 consolidation: continual learning (A -> B -> retest A) --\n", .{});
    try o.print("  {s:>4}  {s:>5}  {s:>5}  {s:>7}   {s:>18}   {s:>16}\n", .{
        "seed", "cons", "accA", "retestA", "consolidated surv", "tentative surv",
    });

    var on_survival_gap: f64 = 0; // consolidated − tentative survival, consolidation ON
    var worst_survival_gap: f64 = std.math.inf(f64);
    var retest_on_sum: f64 = 0;
    var retest_off_sum: f64 = 0;

    for (seeds) |seed| {
        const on = try runCondition(gpa, seed, true);
        const off = try runCondition(gpa, seed, false);

        for ([_]struct { name: []const u8, r: Result }{
            .{ .name = "on", .r = on }, .{ .name = "off", .r = off },
        }) |row| {
            try csv.writer.print("{d},{s},{d:.4},{d:.4},{d},{d:.4},{d},{d:.4}\n", .{
                seed,                 row.name,                     row.r.acc_a, row.r.retest,
                row.r.n_consolidated, row.r.consolidated_survival,
                row.r.n_tentative,    row.r.tentative_survival,
            });
            try o.print("  {d:>4}  {s:>5}  {d:>5.3}  {d:>7.3}   {d:>4} @ {d:>5.2}      {d:>4} @ {d:>5.2}\n", .{
                seed,                 row.name,                     row.r.acc_a, row.r.retest,
                row.r.n_consolidated, row.r.consolidated_survival,
                row.r.n_tentative,    row.r.tentative_survival,
            });
        }

        const gap = on.consolidated_survival - on.tentative_survival;
        on_survival_gap += gap;
        worst_survival_gap = @min(worst_survival_gap, gap);
        retest_on_sum += on.retest;
        retest_off_sum += off.retest;
    }

    try writeAtomic(io, "continual.csv", csv.written());

    const nf = @as(f64, @floatFromInt(seeds.len));
    const mean_gap = on_survival_gap / nf;
    const retest_on = retest_on_sum / nf;
    const retest_off = retest_off_sum / nf;
    const retest_gap = retest_on - retest_off;

    const survives = worst_survival_gap >= pass_survival_margin;
    const less_forgetting = retest_gap >= pass_retest_margin;
    const pass = survives and less_forgetting;

    try o.print(
        \\
        \\  exit criterion (both must hold):
        \\    1. SURVIVAL         consolidated − tentative survival (consolidation on):
        \\                        mean {d:.3}, worst seed {d:.3}   (need >= {d:.2})
        \\    2. LESS FORGETTING  A-retest accuracy  on {d:.3}  vs off {d:.3}
        \\                        gap {d:.3}   (need >= {d:.2})
        \\  VERDICT: {s}
        \\
        \\  wrote continual.csv
        \\
    , .{
        mean_gap,   worst_survival_gap, pass_survival_margin,
        retest_on,  retest_off,
        retest_gap, pass_retest_margin,
        if (pass)
            "PASS -- consolidated pathways survive disuse; unused tentative ones decay."
        else
            "FAIL -- see the two conditions above.",
    });
    try o.flush();
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
