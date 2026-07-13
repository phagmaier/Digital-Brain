//! Phase 6 exit-criterion experiment: does CONSOLIDATION let previously-useful
//! pathways survive a period of disuse better than unused tentative pathways?
//!
//! Exit criterion (spec, Phase 6):
//!   "Previously useful pathways survive better than unused tentative pathways."
//!
//! Continual-learning protocol (train A, then B, then retest A):
//!   * Block A  -- train the full two-choice task (A->0, B->1) TO A MASTERY
//!                 CRITERION (report.md §5). The correct readout synapses become
//!                 useful and, under consolidation, ratchet up their permanence
//!                 (reward-gated, DEC-012). A seed that never masters block A is
//!                 reported separately and excluded from the verdict, so a weak
//!                 block-A fit cannot masquerade as "forgetting".
//!   * Block B  -- present ONLY stimulus B. The A pathway (input_a->action_0) now
//!                 goes UNUSED; the slow structural clock decays and prunes unused,
//!                 unconsolidated pathways. Consolidated ones resist.
//!   * Retest A -- frozen (no learning): present stimulus A, measure accuracy,
//!                 THEN lesion the A readout pathway and re-measure, isolating how
//!                 causally load-bearing that specific pathway is (report.md §5).
//!
//! Three conditions, identical but for the consolidation mechanism under test
//! (report.md §5 asks raw-vs-centered to be compared directly):
//!   raw       (consolidation on, RAW reward r): DEC-012's mandated form.
//!   centered  (consolidation on, (r - baseline)): the failure mode DEC-012 warns
//!             about -- consolidation stalls once the reward baseline saturates.
//!   off       (consolidation off): same decay/prune, no consolidation.
//!
//! The verdict ANDs the two faces of the criterion over MASTERED, PAIRED seeds:
//!   1. SURVIVAL   -- with raw consolidation on, the previously-useful A pathway
//!                    (input_a->action_0) retains far more of its band than the
//!                    unused tentative pathway.
//!   2. LESS FORGETTING -- A-retest accuracy is higher with raw consolidation than
//!                    without (the mechanism is what preserved the pathway).
//! Reported with mean and a 95% normal confidence interval over the seed sample.
//!
//! Produces continual.csv (per seed/condition) and a PASS/FAIL verdict.
//!
//! Build/run:  zig build continual   (add -Doptimize=ReleaseFast)

const std = @import("std");
const cfg = @import("config.zig");
const sim = @import("sim.zig");
const task = @import("task.zig");
const rng = @import("rng.zig");
const provenance = @import("provenance.zig");

const n_seeds = 20; // report.md §5: 20-50 paired seeds
const seeds: [n_seeds]u64 = blk: {
    var s: [n_seeds]u64 = undefined;
    for (0..n_seeds) |i| s[i] = @as(u64, i) + 1;
    break :blk s;
};

// Block A now trains to a MASTERY CRITERION rather than a fixed length: at least
// `block_a_min` episodes, then stop as soon as the rolling accuracy over the last
// `acc_window` episodes reaches `mastery_criterion`, up to a `block_a_max` cap. A
// seed still below criterion at the cap is flagged non-mastered.
const block_a_min: u32 = 1400; // train at least this far (past the exploration plateau)
const block_a_max: u32 = 3200; // hard cap on block-A training
const mastery_criterion: f64 = 0.90; // rolling block-A accuracy required to advance
const block_b_episodes: u32 = 800; // present only B; A pathway goes unused
const retest_episodes: u32 = 200; // frozen accuracy probe on stimulus A
const stim_steps: u32 = 40;
const readout_steps: u32 = 25;
const acc_window: u32 = 300;
const growth_interval: u32 = 50; // structural (decay/prune) window cadence

// Fixed, non-overlapping derived-RNG episode ranges (DEC-004), independent of how
// long block A actually ran, so the task stream stays reproducible per seed.
const block_b_base: u32 = block_a_max;
const retest_base: u32 = block_a_max + block_b_episodes;

const consolidation_lr_on: f32 = 0.05;

// Verdict thresholds.
const pass_survival_margin: f64 = 0.30; // useful-A alive fraction − unused, raw consolidation
const pass_retest_margin: f64 = 0.08; // A-retest accuracy: raw − off

const Condition = enum { raw, centered, off };

fn conditionName(cond: Condition) []const u8 {
    return switch (cond) {
        .raw => "raw",
        .centered => "centered",
        .off => "off",
    };
}

fn baseConfig(seed: u64, cond: Condition) cfg.Config {
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
        .consolidation_lr = if (cond == .off) 0.0 else consolidation_lr_on,
        .consolidation_use_centered_reward = cond == .centered,
    };
}

/// Read out the network's choice this episode from action-group spike counts.
fn choose(seed: u64, ep: u32, count0: u32, count1: u32) u1 {
    if (count0 > count1) return 0;
    if (count1 > count0) return 1;
    var arng = rng.derived(seed, .action, ep);
    return @intCast(arng.below(2));
}

/// Run one stimulus presentation and return (count0, count1) over the readout
/// window. Assumes `resetEpisode` and stimulus fill have already been done.
fn presentAndCount(s: *sim.Sim, l: task.Layout, ext: []f32) struct { u32, u32 } {
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
    return .{ count0, count1 };
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
    const counts = presentAndCount(s, l, ext);
    const chosen = choose(seed, ep, counts[0], counts[1]);
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
        const ep: u32 = retest_base + @as(u32, @intCast(e));
        s.resetEpisode();
        l.fillStimulus(.a, c.task_input_current, ext);
        const counts = presentAndCount(s, l, ext);
        const chosen = choose(seed, ep, counts[0], counts[1]);
        if (chosen == l.correctAction(.a)) correct += 1;
    }
    return @as(f64, @floatFromInt(correct)) / @as(f64, @floatFromInt(retest_episodes));
}

/// Lesion the A readout pathway: zero the plastic input_a->action_0 weights, the
/// synapses that directly vote for the correct answer on stimulus A. A causal
/// probe -- if the retested A behaviour is carried by this specific consolidated
/// pathway, the retest accuracy should collapse after the lesion (report.md §5).
fn lesionAPathway(s: *sim.Sim, l: task.Layout) u32 {
    const syn = &s.network.synapses;
    var lesioned: u32 = 0;
    for (0..syn.n) |k| {
        if (!syn.plastic[k]) continue;
        if (l.input_a.contains(syn.source[k]) and l.action_0.contains(syn.target[k])) {
            syn.weight[k] = 0;
            lesioned += 1;
        }
    }
    return lesioned;
}

// Permanence bands (§8.3: the tentative/established/consolidated state emerges
// from thresholds over permanence, it need not be an enum). We classify each
// plastic readout synapse by its permanence at the END OF BLOCK A, then ask how
// each band SURVIVES block B's disuse. "Consolidated" == previously useful (its
// permanence was ratcheted up by reward); "tentative" == weak, never consolidated.
const consolidated_q: f32 = 0.6;
const tentative_q: f32 = 0.4;

const Result = struct {
    mastered: bool, // reached mastery_criterion within block_a_max
    block_a_used: u32, // episodes actually spent on block A
    acc_a: f64, // rolling accuracy at the end of block A
    retest: f64, // A-retest accuracy after block B
    retest_lesioned: f64, // A-retest accuracy after lesioning the A pathway
    n_lesioned: u32, // plastic input_a->action_0 synapses zeroed by the lesion
    n_consolidated: u32, // plastic synapses in the consolidated band after block A
    n_tentative: u32, // plastic synapses in the tentative band after block A
    consolidated_survival: f64, // fraction of the consolidated band still alive after block B
    tentative_survival: f64, // fraction of the tentative band still alive after block B
};

fn runCondition(gpa: std.mem.Allocator, seed: u64, cond: Condition) !Result {
    const c = baseConfig(seed, cond);
    try c.validate();
    var s = try sim.Sim.init(gpa, c);
    defer s.deinit(gpa);
    const l = task.layout(c);
    const ext = try gpa.alloc(f32, c.n_neurons);
    defer gpa.free(ext);

    // Block A: train the full task until the rolling accuracy hits the mastery
    // criterion (or the cap). A ring buffer over the last `acc_window` outcomes
    // gives the rolling accuracy without rescanning.
    const ring = try gpa.alloc(bool, acc_window);
    defer gpa.free(ring);
    @memset(ring, false);
    var filled: u32 = 0;
    var idx: u32 = 0;
    var wins: u32 = 0;
    var block_a_used: u32 = 0;
    var ep: u32 = 0;
    while (ep < block_a_max) : (ep += 1) {
        const ok = trainEpisode(&s, l, ext, seed, ep, false);
        if ((ep + 1) % growth_interval == 0) _ = s.applyStructuralPlasticity();
        if (filled == acc_window) {
            wins -= @as(u32, @intFromBool(ring[idx]));
        } else {
            filled += 1;
        }
        ring[idx] = ok;
        wins += @as(u32, @intFromBool(ok));
        idx = (idx + 1) % acc_window;
        block_a_used = ep + 1;
        if (block_a_used >= block_a_min and filled == acc_window) {
            const racc = @as(f64, @floatFromInt(wins)) / @as(f64, @floatFromInt(acc_window));
            if (racc >= mastery_criterion) break;
        }
    }
    const denom_win: f64 = @floatFromInt(if (filled == 0) 1 else filled);
    const acc_a = @as(f64, @floatFromInt(wins)) / denom_win;
    const mastered = filled == acc_window and acc_a >= mastery_criterion;

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
    for (0..block_b_episodes) |b| {
        const bep: u32 = block_b_base + @as(u32, @intCast(b));
        _ = trainEpisode(&s, l, ext, seed, bep, true);
        if ((bep + 1) % growth_interval == 0) _ = s.applyStructuralPlasticity();
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

    // Retest A intact, then lesion the A readout pathway and retest again. The two
    // probes share the retest RNG range, so the only difference is the lesion.
    const retest = retestA(&s, l, ext, seed);
    const n_lesioned = lesionAPathway(&s, l);
    const retest_lesioned = retestA(&s, l, ext, seed);

    return .{
        .mastered = mastered,
        .block_a_used = block_a_used,
        .acc_a = acc_a,
        .retest = retest,
        .retest_lesioned = retest_lesioned,
        .n_lesioned = n_lesioned,
        .n_consolidated = n_cons,
        .n_tentative = n_tent,
        .consolidated_survival = if (n_cons == 0) 0 else @as(f64, @floatFromInt(cons_alive)) / @as(f64, @floatFromInt(n_cons)),
        .tentative_survival = if (n_tent == 0) 1 else @as(f64, @floatFromInt(tent_alive)) / @as(f64, @floatFromInt(n_tent)),
    };
}

/// Mean and 95% (normal-approx) confidence half-width of a sample. With < 2
/// samples the half-width is undefined and reported as NaN.
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

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    var buf: [4096]u8 = undefined;
    var stdout = std.Io.File.stdout().writer(io, &buf);
    const o = &stdout.interface;

    var csv: std.Io.Writer.Allocating = .init(gpa);
    defer csv.deinit();
    try csv.writer.print("seed,condition,mastered,block_a_used,acc_a,retest_a,retest_a_lesioned,n_lesioned,n_consolidated,consolidated_survival,n_tentative,tentative_survival\n", .{});

    // Per-seed paired samples, collected only over seeds where the compared
    // conditions BOTH mastered block A (report.md §5: exclude non-mastered).
    var survival_gap = std.ArrayList(f64).empty; // raw: consolidated − tentative survival
    defer survival_gap.deinit(gpa);
    var retest_gap = std.ArrayList(f64).empty; // raw.retest − off.retest
    defer retest_gap.deinit(gpa);
    var lesion_drop_raw = std.ArrayList(f64).empty; // raw.retest − raw.retest_lesioned
    defer lesion_drop_raw.deinit(gpa);
    var lesion_drop_off = std.ArrayList(f64).empty; // off.retest − off.retest_lesioned
    defer lesion_drop_off.deinit(gpa);
    var retest_raw_minus_centered = std.ArrayList(f64).empty; // raw.retest − centered.retest
    defer retest_raw_minus_centered.deinit(gpa);

    var non_mastered = std.ArrayList(u64).empty; // seeds with any non-mastered condition
    defer non_mastered.deinit(gpa);

    try o.print("\n-- Phase 6 consolidation: continual learning (A -> B -> retest A) --\n", .{});
    try o.print("   mastery criterion {d:.2} rolling acc over {d} episodes, cap {d}\n\n", .{ mastery_criterion, acc_window, block_a_max });
    try o.print("  {s:>4}  {s:>8}  {s:>4}  {s:>6}  {s:>5}  {s:>7}  {s:>8}   {s:>10}  {s:>10}\n", .{
        "seed", "cond", "mast", "epA", "accA", "retestA", "les.retest", "cons surv", "tent surv",
    });

    for (seeds) |seed| {
        const raw = try runCondition(gpa, seed, .raw);
        const centered = try runCondition(gpa, seed, .centered);
        const off = try runCondition(gpa, seed, .off);

        for ([_]struct { cond: Condition, r: Result }{
            .{ .cond = .raw, .r = raw }, .{ .cond = .centered, .r = centered }, .{ .cond = .off, .r = off },
        }) |row| {
            try csv.writer.print("{d},{s},{d},{d},{d:.4},{d:.4},{d:.4},{d},{d},{d:.4},{d},{d:.4}\n", .{
                seed,                 conditionName(row.cond),     @intFromBool(row.r.mastered), row.r.block_a_used,
                row.r.acc_a,          row.r.retest,                row.r.retest_lesioned,        row.r.n_lesioned,
                row.r.n_consolidated, row.r.consolidated_survival, row.r.n_tentative,            row.r.tentative_survival,
            });
            try o.print("  {d:>4}  {s:>8}  {s:>4}  {d:>6}  {d:>5.3}  {d:>7.3}  {d:>8.3}   {d:>4} @ {d:>4.2}  {d:>4} @ {d:>4.2}\n", .{
                seed,                  conditionName(row.cond),  if (row.r.mastered) "yes" else "NO",
                row.r.block_a_used,    row.r.acc_a,              row.r.retest,
                row.r.retest_lesioned, row.r.n_consolidated,     row.r.consolidated_survival,
                row.r.n_tentative,     row.r.tentative_survival,
            });
        }

        if (!raw.mastered or !centered.mastered or !off.mastered) try non_mastered.append(gpa, seed);

        // Paired samples require both compared conditions to have mastered block A.
        if (raw.mastered and off.mastered) {
            try survival_gap.append(gpa, raw.consolidated_survival - raw.tentative_survival);
            try retest_gap.append(gpa, raw.retest - off.retest);
            try lesion_drop_raw.append(gpa, raw.retest - raw.retest_lesioned);
            try lesion_drop_off.append(gpa, off.retest - off.retest_lesioned);
        }
        if (raw.mastered and centered.mastered) {
            try retest_raw_minus_centered.append(gpa, raw.retest - centered.retest);
        }
    }

    try writeAtomic(io, "continual.csv", csv.written());
    try provenance.write(io, gpa, "continual.meta.json", "continual_consolidation", .{
        .seeds = seeds,
        .conditions = [_][]const u8{ "raw", "centered", "off" },
        .consolidation_raw_config = baseConfig(seeds[0], .raw),
        .consolidation_centered_config = baseConfig(seeds[0], .centered),
        .consolidation_off_config = baseConfig(seeds[0], .off),
        .block_a_min = block_a_min,
        .block_a_max = block_a_max,
        .mastery_criterion = mastery_criterion,
        .block_b_episodes = block_b_episodes,
        .retest_episodes = retest_episodes,
        .stim_steps = stim_steps,
        .readout_steps = readout_steps,
        .acc_window = acc_window,
        .growth_interval = growth_interval,
        .pass_survival_margin = pass_survival_margin,
        .pass_retest_margin = pass_retest_margin,
    });

    const survival = summarize(survival_gap.items);
    const retest = summarize(retest_gap.items);
    const les_raw = summarize(lesion_drop_raw.items);
    const les_off = summarize(lesion_drop_off.items);
    const raw_vs_cent = summarize(retest_raw_minus_centered.items);

    // Verdict on the mastered, paired sample. Both faces of the criterion must
    // clear their margin, judged on the LOWER confidence bound so a noisy sample
    // cannot pass on point estimate alone.
    const survives = (survival.mean - survival.ci_half) >= pass_survival_margin;
    const less_forgetting = (retest.mean - retest.ci_half) >= pass_retest_margin;
    const enough = survival.n >= 2;
    const pass = enough and survives and less_forgetting;

    try o.print(
        \\
        \\  included (mastered raw & off): {d}/{d} seeds
        \\  non-mastered seeds (excluded from verdict): {any}
        \\
        \\  exit criterion (both must hold, judged on 95% CI lower bound):
        \\    1. SURVIVAL         consolidated − tentative survival (raw consolidation):
        \\                        mean {d:.3} ± {d:.3}   (need lower bound >= {d:.2})
        \\    2. LESS FORGETTING  A-retest accuracy  raw − off:
        \\                        mean {d:.3} ± {d:.3}   (need lower bound >= {d:.2})
        \\
        \\  causal pathway lesion (A readout zeroed, retest drop):
        \\    raw consolidation   {d:.3} ± {d:.3}     off   {d:.3} ± {d:.3}
        \\  raw vs centered consolidation (A-retest, raw − centered):
        \\                        {d:.3} ± {d:.3}   ({d} paired seeds)
        \\  VERDICT: {s}
        \\
        \\  wrote continual.csv
        \\
    , .{
        survival.n,      seeds.len,        non_mastered.items,
        survival.mean,   survival.ci_half, pass_survival_margin,
        retest.mean,     retest.ci_half,   pass_retest_margin,
        les_raw.mean,    les_raw.ci_half,  les_off.mean,
        les_off.ci_half, raw_vs_cent.mean, raw_vs_cent.ci_half,
        raw_vs_cent.n,
        if (pass)
            "PASS -- consolidated pathways survive disuse and are causally load-bearing."
        else if (!enough)
            "INCONCLUSIVE -- too few seeds mastered block A; inspect the table."
        else
            "FAIL -- see the conditions above.",
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
