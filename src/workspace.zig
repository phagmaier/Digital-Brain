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
//! Produces workspace.csv and a PASS/FAIL verdict. Run with:
//!   zig build workspace -Doptimize=ReleaseFast

const std = @import("std");
const cfg = @import("config.zig");
const sim = @import("sim.zig");
const task = @import("task.zig");
const rng = @import("rng.zig");

const seeds = [_]u64{ 1, 2, 3, 4 };
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

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;
    var stdout_buf: [4096]u8 = undefined;
    var stdout = std.Io.File.stdout().writer(io, &stdout_buf);
    const out_stdout = &stdout.interface;

    var csv: std.Io.Writer.Allocating = .init(gpa);
    defer csv.deinit();
    try csv.writer.print("condition,seed,delay,accuracy,mean_delay_workspace_state\n", .{});

    var workspace_mean: f64 = 0;
    var ablated_mean: f64 = 0;
    try out_stdout.print("\n-- Phase 7 workspace delayed-association ablation -----------\n", .{});
    for (conditions) |condition| {
        var mean: f64 = 0;
        try out_stdout.print("  {s}:\n", .{condition.name});
        for (seeds) |seed| {
            const r = try trainOne(gpa, seed, condition.workspace_enabled);
            try csv.writer.print("{s},{d},{d},{d:.4},{d:.5}\n", .{
                condition.name, seed, delay_steps, r.accuracy, r.mean_delay_workspace_state,
            });
            try out_stdout.print("    seed {d}: accuracy {d:.3}, delay workspace {d:.3}\n", .{
                seed, r.accuracy, r.mean_delay_workspace_state,
            });
            mean += r.accuracy;
        }
        mean /= @as(f64, @floatFromInt(seeds.len));
        try out_stdout.print("    mean accuracy {d:.3}\n", .{mean});
        if (condition.workspace_enabled) workspace_mean = mean else ablated_mean = mean;
    }
    try writeAtomic(io, "workspace.csv", csv.written());

    const gap = workspace_mean - ablated_mean;
    const pass = workspace_mean >= pass_workspace_mean and gap >= pass_gap;
    try out_stdout.print(
        "\n  exit criterion -- workspace at delay {d} versus an otherwise-identical ablation:\n" ++
            "    workspace mean    {d:.3}   (need >= {d:.2})\n" ++
            "    ablated mean      {d:.3}\n" ++
            "    causal gain       {d:.3}   (need >= {d:.2})\n" ++
            "  VERDICT: {s}\n\n  wrote workspace.csv\n\n",
        .{ delay_steps, workspace_mean, pass_workspace_mean, ablated_mean, gap, pass_gap, if (pass) "PASS -- bottlenecked broadcast improves delayed performance." else "FAIL -- inspect workspace.csv." },
    );
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
