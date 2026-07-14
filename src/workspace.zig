//! Phase 7 exit-criterion experiment: does a capacity-limited, igniting
//! workspace broadcast causally improve a delayed association?
//!
//! Both conditions use the same delayed two-choice task with the Phase 4
//! self-exciting working-memory assembly deliberately OFF. The only difference
//! is the Phase 7 workspace switch. Candidate input assemblies compete for one
//! broadcast slot; the admitted identity decays unless refreshed and feeds back
//! to its own assembly plus a weak broad excitatory broadcast. A reliable
//! accuracy improvement over the switch-off ablation is causal evidence for the
//! workspace mechanism rather than a claim about an unrestricted memory store.
//!
//! Protocol (Stage 1 / report.md § Stage 1): 20 paired seeds, same seed run under
//! both conditions, with mean and a 95% normal confidence interval over the seed
//! sample. The verdict is judged on the CI *lower* bounds so a noisy sample
//! cannot pass on the point estimate alone (same standard as continual.zig).
//!
//! Produces workspace.csv and a PASS/FAIL verdict. Run with:
//!   zig build workspace -Doptimize=ReleaseFast

const std = @import("std");
const cfg = @import("config.zig");
const sim = @import("sim.zig");
const task = @import("task.zig");
const rng = @import("rng.zig");
const provenance = @import("provenance.zig");

const n_seeds = 20; // report.md Stage 1: 20–50 paired seeds (match continual)
const seeds: [n_seeds]u64 = blk: {
    var s: [n_seeds]u64 = undefined;
    for (0..n_seeds) |i| s[i] = @as(u64, i) + 1;
    break :blk s;
};
const n_episodes: u32 = 1200;
const stim_steps: u32 = 30;
const delay_steps: u32 = 40;
const readout_steps: u32 = 20;
const final_window: u32 = 250;

const Condition = struct { name: []const u8, workspace_enabled: bool };
const conditions = [_]Condition{
    .{ .name = "workspace", .workspace_enabled = true },
    .{ .name = "ablated", .workspace_enabled = false },
};

// Verdict thresholds (judged on 95% CI lower bound over the paired seed sample).
const pass_workspace_mean: f64 = 0.65;
const pass_gap: f64 = 0.10;

fn baseConfig(seed: u64, workspace_enabled: bool) cfg.Config {
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
        .eligibility_decay = 0.97,
        .adaptation_enabled = false,
        // Deliberately no Phase 4 self-excitation: this is a workspace vs no
        // workspace ablation, not a comparison against the old memory assembly.
        .task_recurrent_weight = 0.0,
        .workspace_enabled = workspace_enabled,
        .workspace_capacity = 1,
        .workspace_candidate_decay = 0.85,
        .workspace_ignition_threshold = 0.75,
        .workspace_state_decay = 0.90,
        .workspace_feedback_current = 0.45,
        .workspace_broadcast_current = 0.03,
    };
}

const EpisodeResult = struct {
    correct: bool,
    mean_delay_workspace_state: f64,
};

fn runEpisode(s: *sim.Sim, l: task.Layout, ext: []f32, seed: u64, episode: u32) EpisodeResult {
    s.resetEpisode();
    var trng = rng.derived(seed, .task, episode);
    const choice: task.Choice = if (trng.below(2) == 0) .a else .b;
    const c = s.network.config;
    l.fillStimulus(choice, c.task_input_current, ext);

    for (0..stim_steps) |_| _ = s.step(ext);
    var delay_workspace_sum: f64 = 0;
    for (0..delay_steps) |_| {
        const m = s.step(null);
        delay_workspace_sum += m.workspace_state;
    }

    var count0: u32 = 0;
    var count1: u32 = 0;
    for (0..readout_steps) |_| {
        _ = s.step(null);
        const fired = s.network.neurons.fired;
        for (l.action_0.lo..l.action_0.hi) |i| count0 += @intFromBool(fired[i]);
        for (l.action_1.lo..l.action_1.hi) |i| count1 += @intFromBool(fired[i]);
    }

    const chosen: u1 = if (count0 > count1) 0 else if (count1 > count0) 1 else blk: {
        var arng = rng.derived(seed, .action, episode);
        break :blk @intCast(arng.below(2));
    };
    const correct = chosen == l.correctAction(choice);
    s.applyReward(if (correct) 1.0 else -1.0);
    s.applyHomeostasis();
    return .{
        .correct = correct,
        .mean_delay_workspace_state = delay_workspace_sum / @as(f64, @floatFromInt(delay_steps)),
    };
}

const Result = struct { accuracy: f64, mean_delay_workspace_state: f64 };

fn trainOne(gpa: std.mem.Allocator, seed: u64, workspace_enabled: bool) !Result {
    const c = baseConfig(seed, workspace_enabled);
    var s = try sim.Sim.init(gpa, c);
    defer s.deinit(gpa);
    const l = task.layout(c);
    const ext = try gpa.alloc(f32, c.n_neurons);
    defer gpa.free(ext);

    var correct: u32 = 0;
    var workspace_sum: f64 = 0;
    for (0..n_episodes) |ep| {
        const r = runEpisode(&s, l, ext, seed, @intCast(ep));
        if (ep >= n_episodes - final_window) {
            correct += @intFromBool(r.correct);
            workspace_sum += r.mean_delay_workspace_state;
        }
    }
    return .{
        .accuracy = @as(f64, @floatFromInt(correct)) / @as(f64, @floatFromInt(final_window)),
        .mean_delay_workspace_state = workspace_sum / @as(f64, @floatFromInt(final_window)),
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
    var stdout_buf: [4096]u8 = undefined;
    var stdout = std.Io.File.stdout().writer(io, &stdout_buf);
    const out_stdout = &stdout.interface;

    var csv: std.Io.Writer.Allocating = .init(gpa);
    defer csv.deinit();
    try csv.writer.print("condition,seed,delay,accuracy,mean_delay_workspace_state\n", .{});

    // Paired per-seed samples (same seed under both conditions).
    var workspace_accs: [n_seeds]f64 = undefined;
    var ablated_accs: [n_seeds]f64 = undefined;
    var gaps: [n_seeds]f64 = undefined;
    var workspace_states: [n_seeds]f64 = undefined;
    var ablated_states: [n_seeds]f64 = undefined;

    try out_stdout.print(
        "\n-- Phase 7 workspace delayed-association ablation -----------\n" ++
            "   {d} paired seeds, delay {d}, final window {d}\n\n",
        .{ n_seeds, delay_steps, final_window },
    );
    try out_stdout.print("  {s:>4}  {s:>10}  {s:>10}  {s:>8}  {s:>12}  {s:>12}\n", .{
        "seed", "workspace", "ablated", "gap", "ws state", "abl state",
    });

    for (seeds, 0..) |seed, i| {
        const ws = try trainOne(gpa, seed, true);
        const ab = try trainOne(gpa, seed, false);

        workspace_accs[i] = ws.accuracy;
        ablated_accs[i] = ab.accuracy;
        gaps[i] = ws.accuracy - ab.accuracy;
        workspace_states[i] = ws.mean_delay_workspace_state;
        ablated_states[i] = ab.mean_delay_workspace_state;

        try csv.writer.print("{s},{d},{d},{d:.4},{d:.5}\n", .{
            "workspace", seed, delay_steps, ws.accuracy, ws.mean_delay_workspace_state,
        });
        try csv.writer.print("{s},{d},{d},{d:.4},{d:.5}\n", .{
            "ablated", seed, delay_steps, ab.accuracy, ab.mean_delay_workspace_state,
        });
        try out_stdout.print("  {d:>4}  {d:>10.3}  {d:>10.3}  {d:>8.3}  {d:>12.3}  {d:>12.3}\n", .{
            seed, ws.accuracy, ab.accuracy, gaps[i], ws.mean_delay_workspace_state, ab.mean_delay_workspace_state,
        });
    }

    try writeAtomic(io, "workspace.csv", csv.written());
    try provenance.write(io, gpa, "workspace.meta.json", "workspace_broadcast", .{
        .seeds = seeds,
        .n_seeds = n_seeds,
        .workspace_config = baseConfig(seeds[0], true),
        .ablated_config = baseConfig(seeds[0], false),
        .conditions = conditions,
        .n_episodes = n_episodes,
        .stim_steps = stim_steps,
        .delay_steps = delay_steps,
        .readout_steps = readout_steps,
        .final_window = final_window,
        .pass_workspace_mean = pass_workspace_mean,
        .pass_gap = pass_gap,
        .ci_level = 0.95,
        .verdict_on = "ci_lower_bound",
    });

    const ws_stat = summarize(&workspace_accs);
    const ab_stat = summarize(&ablated_accs);
    const gap_stat = summarize(&gaps);
    const ws_state_stat = summarize(&workspace_states);
    const ab_state_stat = summarize(&ablated_states);

    // Verdict: both faces clear their margin on the LOWER confidence bound.
    const enough = ws_stat.n >= 2;
    const workspace_ok = (ws_stat.mean - ws_stat.ci_half) >= pass_workspace_mean;
    const gap_ok = (gap_stat.mean - gap_stat.ci_half) >= pass_gap;
    const pass = enough and workspace_ok and gap_ok;

    try out_stdout.print(
        \\
        \\  paired seeds: {d}
        \\
        \\  exit criterion (both must hold, judged on 95% CI lower bound):
        \\    1. WORKSPACE MEAN   accuracy with workspace on:
        \\                        {d:.3} ± {d:.3}   (need lower bound >= {d:.2})
        \\    2. CAUSAL GAIN      workspace − ablated accuracy (paired):
        \\                        {d:.3} ± {d:.3}   (need lower bound >= {d:.2})
        \\
        \\  ablated mean accuracy {d:.3} ± {d:.3}
        \\  delay workspace state  on {d:.3} ± {d:.3}   off {d:.3} ± {d:.3}
        \\  VERDICT: {s}
        \\
        \\  wrote workspace.csv
        \\
    , .{
        n_seeds,
        ws_stat.mean,
        ws_stat.ci_half,
        pass_workspace_mean,
        gap_stat.mean,
        gap_stat.ci_half,
        pass_gap,
        ab_stat.mean,
        ab_stat.ci_half,
        ws_state_stat.mean,
        ws_state_stat.ci_half,
        ab_state_stat.mean,
        ab_state_stat.ci_half,
        if (pass)
            "PASS -- bottlenecked broadcast improves delayed performance."
        else if (!enough)
            "INCONCLUSIVE -- too few seeds; inspect the table."
        else
            "FAIL -- inspect workspace.csv.",
    });
    try stdout.interface.flush();
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
