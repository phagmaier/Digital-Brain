//! Stage 3 Track A (report.md / final.md): what is stochasticity doing?
//!
//! Factorial ablation on the two-choice immediate association (DEC-008/009):
//!
//!   firing {stochastic, deterministic} × release {stochastic, deterministic}
//!
//! plus credit/exploration interventions on the full-stochastic baseline:
//!
//!   * forced exploration  — ε-greedy override of the chosen action (derived
//!     `action` stream, DEC-004), so both actions are tried early;
//!   * winner-take-all credit — only plastic synapses into the chosen action
//!     assembly keep their eligibility before `applyReward`.
//!
//! Metrics (per condition × seed):
//!   * final accuracy over a terminal window
//!   * episodes-to-90% (first block of `block_size` with accuracy ≥ 0.90)
//!   * takeoff episode (first block ≥ 0.70, a softer takeoff marker)
//!
//! Summary: mean ± 95% CI over seeds; median episodes-to-90% (null → n_episodes).
//! The scientific goal is characterization, not a single PASS bar — still print
//! a descriptive verdict on whether full stochasticity is necessary and whether
//! forced exploration / WTA credit collapses the exploration plateau.
//!
//! Produces: stochastic.csv, stochastic.meta.json
//! Build/run:  zig build stochastic -Doptimize=ReleaseFast

const std = @import("std");
const cfg = @import("config.zig");
const sim = @import("sim.zig");
const task = @import("task.zig");
const rng = @import("rng.zig");
const provenance = @import("provenance.zig");

const n_seeds = 20;
const seeds: [n_seeds]u64 = blk: {
    var s: [n_seeds]u64 = undefined;
    for (0..n_seeds) |i| s[i] = @as(u64, i) + 1;
    break :blk s;
};

const n_episodes: u32 = 1500;
const stim_steps: u32 = 40;
const readout_steps: u32 = 25;
const block_size: u32 = 50;
const final_window: u32 = 300;
const takeoff_threshold: f64 = 0.70;
const mastery_threshold: f64 = 0.90;
/// Peak ε for forced exploration. Annealed linearly to 0 over the first half
/// of training so early symmetry-breaking is forced without permanently
/// polluting asymptotic accuracy (fixed-ε would cap final performance at 1-ε/2).
const forced_eps: f32 = 0.25;
const explore_anneal_episodes: u32 = n_episodes / 2;

const Condition = struct {
    name: []const u8,
    stochastic_firing: bool,
    stochastic_release: bool,
    forced_exploration: bool,
    wta_credit: bool,
};

const conditions = [_]Condition{
    .{ .name = "fire_stoch_rel_stoch", .stochastic_firing = true, .stochastic_release = true, .forced_exploration = false, .wta_credit = false },
    .{ .name = "fire_stoch_rel_det", .stochastic_firing = true, .stochastic_release = false, .forced_exploration = false, .wta_credit = false },
    .{ .name = "fire_det_rel_stoch", .stochastic_firing = false, .stochastic_release = true, .forced_exploration = false, .wta_credit = false },
    .{ .name = "fire_det_rel_det", .stochastic_firing = false, .stochastic_release = false, .forced_exploration = false, .wta_credit = false },
    .{ .name = "stoch_forced_explore", .stochastic_firing = true, .stochastic_release = true, .forced_exploration = true, .wta_credit = false },
    .{ .name = "stoch_wta_credit", .stochastic_firing = true, .stochastic_release = true, .forced_exploration = false, .wta_credit = true },
    .{ .name = "stoch_explore_wta", .stochastic_firing = true, .stochastic_release = true, .forced_exploration = true, .wta_credit = true },
    .{ .name = "det_explore_wta", .stochastic_firing = false, .stochastic_release = false, .forced_exploration = true, .wta_credit = true },
};

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
        .stochastic_firing = cond.stochastic_firing,
        .stochastic_release = cond.stochastic_release,
    };
}

const ChoiceResult = struct {
    chosen: u1,
    forced: bool,
};

fn explorationEps(episode: u32, enabled: bool) f32 {
    if (!enabled) return 0.0;
    if (episode >= explore_anneal_episodes) return 0.0;
    const frac = 1.0 - @as(f32, @floatFromInt(episode)) / @as(f32, @floatFromInt(explore_anneal_episodes));
    return forced_eps * frac;
}

/// Select the action assembly. With forced exploration, an annealed ε-greedy
/// override from the derived `action` stream (DEC-004) runs first so the
/// explore decision is reproducible and aligned across ablations. Tie-breaks
/// reuse the same stream after the explore draws (or alone when ε = 0).
fn chooseAction(
    count0: u32,
    count1: u32,
    seed: u64,
    episode: u32,
    explore_eps: f32,
) ChoiceResult {
    var arng = rng.derived(seed, .action, episode);
    if (explore_eps > 0.0 and arng.float01() < explore_eps) {
        return .{ .chosen = @intCast(arng.below(2)), .forced = true };
    }
    if (count0 > count1) return .{ .chosen = 0, .forced = false };
    if (count1 > count0) return .{ .chosen = 1, .forced = false };
    return .{ .chosen = @intCast(arng.below(2)), .forced = false };
}

fn runEpisode(
    s: *sim.Sim,
    l: task.Layout,
    ext: []f32,
    seed: u64,
    episode: u32,
    cond: Condition,
) bool {
    s.resetEpisode();

    var trng = rng.derived(seed, .task, episode);
    const choice: task.Choice = if (trng.below(2) == 0) .a else .b;

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

    const eps = explorationEps(episode, cond.forced_exploration);
    const pick = chooseAction(count0, count1, seed, episode, eps);
    const correct = pick.chosen == l.correctAction(choice);

    if (cond.wta_credit) {
        const grp = if (pick.chosen == 0) l.action_0 else l.action_1;
        s.maskEligibilityToTargets(grp.lo, grp.hi);
    }

    s.applyReward(if (correct) 1.0 else -1.0);
    s.applyHomeostasis();
    return correct;
}

const SeedOutcome = struct {
    final_accuracy: f64,
    episodes_to_mastery: ?u32, // first block end with block acc ≥ mastery
    episodes_to_takeoff: ?u32, // first block end with block acc ≥ takeoff
    n_forced: u32,
};

fn runSeed(gpa: std.mem.Allocator, seed: u64, cond: Condition) !SeedOutcome {
    const c = baseConfig(seed, cond);
    try c.validate();
    var s = try sim.Sim.init(gpa, c);
    defer s.deinit(gpa);
    const l = task.layout(c);

    const ext = try gpa.alloc(f32, c.n_neurons);
    defer gpa.free(ext);

    var final_correct: u32 = 0;
    var block_correct: u32 = 0;
    var episodes_to_mastery: ?u32 = null;
    var episodes_to_takeoff: ?u32 = null;
    var n_forced: u32 = 0;

    for (0..n_episodes) |ep| {
        // Recompute forced flag for counting without re-running dynamics: the
        // episode runner already applied the policy; re-derive the explore bit
        // for the same (seed, episode) so the count is exact.
        const eps = explorationEps(@intCast(ep), cond.forced_exploration);
        if (eps > 0.0) {
            var arng = rng.derived(seed, .action, @intCast(ep));
            if (arng.float01() < eps) n_forced += 1;
        }

        const correct = runEpisode(&s, l, ext, seed, @intCast(ep), cond);
        block_correct += @intFromBool(correct);
        if (ep >= n_episodes - final_window) final_correct += @intFromBool(correct);

        if ((ep + 1) % block_size == 0) {
            const acc = @as(f64, @floatFromInt(block_correct)) / @as(f64, @floatFromInt(block_size));
            const end_ep: u32 = @intCast(ep + 1);
            if (episodes_to_takeoff == null and acc >= takeoff_threshold) episodes_to_takeoff = end_ep;
            if (episodes_to_mastery == null and acc >= mastery_threshold) episodes_to_mastery = end_ep;
            block_correct = 0;
        }
    }

    return .{
        .final_accuracy = @as(f64, @floatFromInt(final_correct)) / @as(f64, @floatFromInt(final_window)),
        .episodes_to_mastery = episodes_to_mastery,
        .episodes_to_takeoff = episodes_to_takeoff,
        .n_forced = n_forced,
    };
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

fn medianSorted(xs: []f64) f64 {
    if (xs.len == 0) return std.math.nan(f64);
    std.mem.sort(f64, xs, {}, std.sort.asc(f64));
    const mid = xs.len / 2;
    if (xs.len % 2 == 1) return xs[mid];
    return 0.5 * (xs[mid - 1] + xs[mid]);
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

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    var csv: std.Io.Writer.Allocating = .init(gpa);
    defer csv.deinit();
    try csv.writer.print(
        "condition,seed,final_accuracy,episodes_to_mastery,episodes_to_takeoff,n_forced,stochastic_firing,stochastic_release,forced_exploration,wta_credit\n",
        .{},
    );

    var stdout_buf: [8192]u8 = undefined;
    var stdout = std.Io.File.stdout().writer(io, &stdout_buf);
    const o = &stdout.interface;

    try o.print(
        \\
        \\-- Stage 3 Track A: stochasticity factorial + exploration/WTA -------
        \\  {d} seeds × {d} conditions, {d} episodes (block={d}, final={d})
        \\  forced ε peak = {d:.2}, annealed to 0 over first {d} episodes
        \\
    , .{ n_seeds, conditions.len, n_episodes, block_size, final_window, forced_eps, explore_anneal_episodes });
    try o.flush();

    // Per-condition summaries for the printed table.
    var mean_acc: [conditions.len]Stat = undefined;
    var mean_mastery: [conditions.len]Stat = undefined;
    var median_mastery: [conditions.len]f64 = undefined;
    var mean_takeoff: [conditions.len]Stat = undefined;
    var success_rate: [conditions.len]f64 = undefined; // fraction of seeds with final_acc ≥ 0.60

    for (conditions, 0..) |cond, ci| {
        var accs: [n_seeds]f64 = undefined;
        var mastery_or_cap: [n_seeds]f64 = undefined;
        var takeoff_or_cap: [n_seeds]f64 = undefined;
        var n_success: u32 = 0;

        try o.print("  running {s} ...\n", .{cond.name});
        try o.flush();

        for (seeds, 0..) |seed, si| {
            const out = try runSeed(gpa, seed, cond);
            accs[si] = out.final_accuracy;
            if (out.final_accuracy >= 0.60) n_success += 1;

            const mast: f64 = if (out.episodes_to_mastery) |e| @floatFromInt(e) else @floatFromInt(n_episodes);
            const take: f64 = if (out.episodes_to_takeoff) |e| @floatFromInt(e) else @floatFromInt(n_episodes);
            mastery_or_cap[si] = mast;
            takeoff_or_cap[si] = take;

            const mast_str: []const u8 = if (out.episodes_to_mastery) |e|
                try std.fmt.allocPrint(gpa, "{d}", .{e})
            else
                try std.fmt.allocPrint(gpa, "", .{});
            defer gpa.free(mast_str);
            const take_str: []const u8 = if (out.episodes_to_takeoff) |e|
                try std.fmt.allocPrint(gpa, "{d}", .{e})
            else
                try std.fmt.allocPrint(gpa, "", .{});
            defer gpa.free(take_str);

            try csv.writer.print(
                "{s},{d},{d:.4},{s},{s},{d},{d},{d},{d},{d}\n",
                .{
                    cond.name,
                    seed,
                    out.final_accuracy,
                    mast_str,
                    take_str,
                    out.n_forced,
                    @intFromBool(cond.stochastic_firing),
                    @intFromBool(cond.stochastic_release),
                    @intFromBool(cond.forced_exploration),
                    @intFromBool(cond.wta_credit),
                },
            );
        }

        mean_acc[ci] = summarize(&accs);
        mean_mastery[ci] = summarize(&mastery_or_cap);
        mean_takeoff[ci] = summarize(&takeoff_or_cap);
        var med_buf = mastery_or_cap;
        median_mastery[ci] = medianSorted(&med_buf);
        success_rate[ci] = @as(f64, @floatFromInt(n_success)) / @as(f64, @floatFromInt(n_seeds));
    }

    try writeAtomic(io, "stochastic.csv", csv.written());
    try provenance.write(io, gpa, "stochastic.meta.json", "stage3_stochasticity_factorial", .{
        .seeds = seeds,
        .n_episodes = n_episodes,
        .stim_steps = stim_steps,
        .readout_steps = readout_steps,
        .block_size = block_size,
        .final_window = final_window,
        .takeoff_threshold = takeoff_threshold,
        .mastery_threshold = mastery_threshold,
        .forced_eps = forced_eps,
        .explore_anneal_episodes = explore_anneal_episodes,
        .conditions = conditions,
        .config_template = baseConfig(seeds[0], conditions[0]),
    });

    // ---- report ---------------------------------------------------------
    try o.print(
        \\
        \\  summary (mean ± 95% CI half-width over {d} seeds):
        \\  condition               final_acc          success≥0.6  med ep→0.9  mean ep→0.7
        \\
    , .{n_seeds});

    for (conditions, 0..) |cond, ci| {
        const a = mean_acc[ci];
        const t = mean_takeoff[ci];
        try o.print(
            "  {s:<22} {d:.3} ± {d:.3}     {d:.2}         {d:>8.0}    {d:.0} ± {d:.0}\n",
            .{
                cond.name,
                a.mean,
                a.ci_half,
                success_rate[ci],
                median_mastery[ci],
                t.mean,
                t.ci_half,
            },
        );
    }

    // Characterization verdicts (descriptive, not a single exit criterion).
    const base = mean_acc[0]; // fire_stoch_rel_stoch
    const det_det = mean_acc[3]; // fire_det_rel_det
    const forced = mean_acc[4];
    const wta = mean_acc[5];
    const both = mean_acc[6];

    const stoch_helps = base.mean - det_det.mean > 0.05;
    const explore_helps = forced.mean - base.mean > 0.02 or median_mastery[4] + 50.0 < median_mastery[0];
    const wta_helps = wta.mean - base.mean > 0.02 or median_mastery[5] + 50.0 < median_mastery[0];

    try o.print(
        \\
        \\  characterization:
        \\    full stochastic vs fully deterministic final_acc gap: {d:.3}
        \\      -> stochasticity {s} for asymptotic accuracy
        \\    forced exploration vs baseline: final {d:.3} vs {d:.3}, med→0.9 {d:.0} vs {d:.0}
        \\      -> forced exploration {s}
        \\    WTA credit vs baseline: final {d:.3} vs {d:.3}, med→0.9 {d:.0} vs {d:.0}
        \\      -> WTA credit {s}
        \\    explore+WTA final_acc: {d:.3} ± {d:.3}
        \\
        \\  wrote stochastic.csv / stochastic.meta.json
        \\
    , .{
        base.mean - det_det.mean,
        if (stoch_helps) "appears helpful" else "not clearly required (at this budget)",
        forced.mean,
        base.mean,
        median_mastery[4],
        median_mastery[0],
        if (explore_helps) "appears to speed/raise learning" else "no clear gain with annealed ε",
        wta.mean,
        base.mean,
        median_mastery[5],
        median_mastery[0],
        if (wta_helps) "appears to improve credit assignment" else "no clear gain alone",
        both.mean,
        both.ci_half,
    });
    try o.flush();
}
