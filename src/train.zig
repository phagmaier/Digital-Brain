//! Phase 3 exit-criterion experiment: does the network learn a two-choice
//! immediate association above chance, across multiple random seeds?
//!
//! One episode (the doc's training loop):
//!   1. resetEpisode -- clear fast state + eligibility, keep weights.
//!   2. Present stimulus A or B (chosen by the derived `task` RNG, DEC-004, so
//!      episode e is the same stimulus under every variant).
//!   3. Run the stimulus window; over its tail, count spikes in action_0 vs
//!      action_1. The more active assembly is the network's choice.
//!   4. Reward +1 if the choice matches the correct mapping, else -1.
//!   5. applyReward  -- the three-factor update on eligible synapses (DEC-009).
//!   6. applyHomeostasis -- the per-episode "Update homeostasis" (Phase 2 seam).
//!   7. Log.
//!
//! Writes train.csv (block accuracy per seed over training) and prints a
//! PASS/FAIL verdict against chance.
//!
//! Build/run:  zig build train

const std = @import("std");
const cfg = @import("config.zig");
const sim = @import("sim.zig");
const task = @import("task.zig");
const rng = @import("rng.zig");
const provenance = @import("provenance.zig");

const seeds = [_]u64{ 1, 2, 3, 4, 5, 6, 7, 8 };
const n_episodes: u32 = 1500;
const stim_steps: u32 = 40; // timesteps the stimulus is presented
const readout_steps: u32 = 25; // spikes counted over the last this-many steps
const block_size: u32 = 50; // accuracy is reported per block of episodes
const final_window: u32 = 300; // final accuracy is the mean over the last window

// Verdict thresholds. Chance is 0.5; over `final_window` episodes the chance
// standard deviation is ~0.029, so these are several sigma above chance.
const pass_mean: f64 = 0.60; // mean final accuracy across seeds
const pass_min: f64 = 0.55; // every seed must clear this (above chance)

fn baseConfig(seed: u64) cfg.Config {
    return .{
        .master_seed = seed,
        .n_neurons = 100,
        .task_enabled = true,
        .plasticity_enabled = true,
        // Homeostasis on, per-episode cadence -- exactly the training regime.
        .homeostasis_enabled = true,
        .homeostasis_per_step = false,
        .target_rate = 0.05,
        .homeostasis_lr = 0.05,
        .task_group_size = 8,
    };
}

/// Run one episode and return whether the network chose correctly.
fn runEpisode(s: *sim.Sim, l: task.Layout, ext: []f32, seed: u64, episode: u32) bool {
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

    var chosen: u1 = undefined;
    if (count0 > count1) {
        chosen = 0;
    } else if (count1 > count0) {
        chosen = 1;
    } else {
        // Tie -> explore via the derived `action` stream (reproducible).
        var arng = rng.derived(seed, .action, episode);
        chosen = @intCast(arng.below(2));
    }

    const correct = chosen == l.correctAction(choice);
    s.applyReward(if (correct) 1.0 else -1.0);
    s.applyHomeostasis();
    return correct;
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    var out: std.Io.Writer.Allocating = .init(gpa);
    defer out.deinit();
    const w = &out.writer;
    try w.print("episode,seed,block_accuracy\n", .{});

    const results = try gpa.alloc(bool, n_episodes);
    defer gpa.free(results);

    var final_acc: [seeds.len]f64 = undefined;

    for (seeds, 0..) |seed, si| {
        const c = baseConfig(seed);
        try c.validate();
        var s = try sim.Sim.init(gpa, c);
        defer s.deinit(gpa);
        const l = task.layout(c);

        const ext = try gpa.alloc(f32, c.n_neurons);
        defer gpa.free(ext);

        var block_correct: u32 = 0;
        for (0..n_episodes) |ep| {
            const correct = runEpisode(&s, l, ext, seed, @intCast(ep));
            results[ep] = correct;
            block_correct += @intFromBool(correct);
            if ((ep + 1) % block_size == 0) {
                const acc = @as(f64, @floatFromInt(block_correct)) / @as(f64, @floatFromInt(block_size));
                try w.print("{d},{d},{d:.4}\n", .{ ep + 1, seed, acc });
                block_correct = 0;
            }
        }

        var fc: u32 = 0;
        for (n_episodes - final_window..n_episodes) |ep| fc += @intFromBool(results[ep]);
        final_acc[si] = @as(f64, @floatFromInt(fc)) / @as(f64, @floatFromInt(final_window));
    }

    try writeAtomic(io, "train.csv", out.written());
    try provenance.write(io, gpa, "train.meta.json", "immediate_association", .{
        .seeds = seeds,
        .config_template = baseConfig(seeds[0]),
        .n_episodes = n_episodes,
        .stim_steps = stim_steps,
        .readout_steps = readout_steps,
        .block_size = block_size,
        .final_window = final_window,
        .pass_mean = pass_mean,
        .pass_min = pass_min,
    });

    // ---- verdict --------------------------------------------------------
    var sum: f64 = 0;
    var min_acc: f64 = 1.0;
    for (final_acc) |a| {
        sum += a;
        min_acc = @min(min_acc, a);
    }
    const mean_acc = sum / @as(f64, @floatFromInt(seeds.len));
    const pass = mean_acc >= pass_mean and min_acc >= pass_min;

    var buf: [4096]u8 = undefined;
    var stdout = std.Io.File.stdout().writer(io, &buf);
    const o = &stdout.interface;
    try o.print(
        \\
        \\-- Phase 3 local reward learning: two-choice association -------
        \\  seeds              {d}
        \\  episodes/seed      {d}
        \\  final accuracy over last {d} episodes (chance = 0.500):
        \\
    , .{ seeds.len, n_episodes, final_window });
    for (seeds, final_acc) |seed, a| {
        try o.print("    seed {d:>2}          {d:.3}\n", .{ seed, a });
    }
    try o.print(
        \\
        \\  mean               {d:.3}   (need >= {d:.2})
        \\  worst seed         {d:.3}   (need >= {d:.2})
        \\  VERDICT: {s}
        \\
        \\  wrote train.csv
        \\
    , .{
        mean_acc,                                                                                                           pass_mean,
        min_acc,                                                                                                            pass_min,
        if (pass) "PASS -- learns the association above chance on every seed." else "FAIL -- see per-seed accuracy above.",
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
