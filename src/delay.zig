//! Phase 4 exit-criterion experiment: does the network retain stimulus
//! information across a NONZERO delay, better than chance?
//!
//! Same two-choice association as Phase 3, but the episode inserts a delay with
//! no input between stimulus and readout:
//!
//!   present stimulus (stim_steps, input on)
//!   delay          (delay_steps, NO input)   <- the network must hold the info
//!   readout        (readout_steps, NO input) <- count action spikes, choose
//!
//! Two conditions, to separate the mechanism from the substrate:
//!   memory         -- input-group self-excitation on (DEC-010 working memory)
//!   reservoir_only -- self-excitation off; only the fixed reservoir's fading
//!                     memory is available.
//! The bare reservoir retains a stimulus for a few steps (echo-state fading
//! memory) then decays to chance; the self-exciting assembly holds it across
//! long delays. The divergence of the two accuracy-vs-delay curves is the
//! evidence that the working-memory mechanism -- not just the substrate -- is
//! doing the retaining.
//!
//! Produces:
//!   * delay.csv     -- accuracy per (condition, delay, seed): the retention curves
//!   * retention.csv -- recurrent-state analysis: mean input-assembly activity
//!                      per timestep for the active vs the other assembly, showing
//!                      the kicked assembly persists stimulus-specifically
//!   * a PASS/FAIL verdict at the target delay
//!
//! Build/run:  zig build delay

const std = @import("std");
const cfg = @import("config.zig");
const sim = @import("sim.zig");
const task = @import("task.zig");
const rng = @import("rng.zig");
const provenance = @import("provenance.zig");

const seeds = [_]u64{ 1, 2, 3, 4 };
const n_episodes: u32 = 1500;
const stim_steps: u32 = 30;
const readout_steps: u32 = 20;
const final_window: u32 = 300;

const delays = [_]u32{ 0, 5, 10, 20, 40 };
const target_delay: u32 = 20; // nonzero, and past the bare reservoir's fading memory

const memory_weight: f32 = 0.5; // self-excitation in the "memory" condition
const Condition = struct { name: []const u8, w: f32 };
const conditions = [_]Condition{
    .{ .name = "memory", .w = memory_weight },
    .{ .name = "reservoir_only", .w = 0.0 },
};

const pass_mean: f64 = 0.60;
const pass_min: f64 = 0.55;

fn baseConfig(seed: u64, recurrent_weight: f32) cfg.Config {
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
        // Phase 4 knobs:
        .task_recurrent_weight = recurrent_weight, // working-memory self-excitation
        .eligibility_decay = 0.95, // longer, to span the delay (build item)
        .adaptation_enabled = false, // adaptation tuning: don't fight persistence
    };
}

fn runEpisode(s: *sim.Sim, l: task.Layout, ext: []f32, seed: u64, episode: u32, delay_steps: u32) bool {
    s.resetEpisode();

    var trng = rng.derived(seed, .task, episode);
    const choice: task.Choice = if (trng.below(2) == 0) .a else .b;

    const c = s.network.config;
    l.fillStimulus(choice, c.task_input_current, ext);

    for (0..stim_steps) |_| _ = s.step(ext); // stimulus on
    for (0..delay_steps) |_| _ = s.step(null); // delay, no input

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
        var arng = rng.derived(seed, .action, episode);
        chosen = @intCast(arng.below(2));
    }

    const correct = chosen == l.correctAction(choice);
    s.applyReward(if (correct) 1.0 else -1.0);
    s.applyHomeostasis();
    return correct;
}

/// Train one seed at one delay from scratch; return final-window accuracy.
fn trainOne(gpa: std.mem.Allocator, seed: u64, delay_steps: u32, recurrent_weight: f32) !f64 {
    const c = baseConfig(seed, recurrent_weight);
    var s = try sim.Sim.init(gpa, c);
    defer s.deinit(gpa);
    const l = task.layout(c);
    const ext = try gpa.alloc(f32, c.n_neurons);
    defer gpa.free(ext);

    var correct_in_window: u32 = 0;
    for (0..n_episodes) |ep| {
        const correct = runEpisode(&s, l, ext, seed, @intCast(ep), delay_steps);
        if (ep >= n_episodes - final_window and correct) correct_in_window += 1;
    }
    return @as(f64, @floatFromInt(correct_in_window)) / @as(f64, @floatFromInt(final_window));
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    var buf: [4096]u8 = undefined;
    var stdout = std.Io.File.stdout().writer(io, &buf);
    const o = &stdout.interface;

    var out: std.Io.Writer.Allocating = .init(gpa);
    defer out.deinit();
    try out.writer.print("condition,delay,seed,accuracy\n", .{});

    var target_sum: f64 = 0;
    var target_min: f64 = 1.0;

    try o.print("\n-- Phase 4 delayed association: accuracy vs delay --------------\n", .{});
    for (conditions) |cond| {
        try o.print("  {s}:\n", .{cond.name});
        for (delays) |d| {
            var sum: f64 = 0;
            for (seeds) |seed| {
                const acc = try trainOne(gpa, seed, d, cond.w);
                try out.writer.print("{s},{d},{d},{d:.4}\n", .{ cond.name, d, seed, acc });
                sum += acc;
                if (d == target_delay and cond.w == memory_weight) {
                    target_sum += acc;
                    target_min = @min(target_min, acc);
                }
            }
            try o.print("    delay {d:>2}        mean accuracy {d:.3}\n", .{ d, sum / @as(f64, @floatFromInt(seeds.len)) });
        }
    }
    try writeAtomic(io, "delay.csv", out.written());
    try provenance.write(io, gpa, "delay.meta.json", "delayed_association", .{
        .seeds = seeds,
        .config_template = baseConfig(seeds[0], memory_weight),
        .conditions = conditions,
        .delays = delays,
        .n_episodes = n_episodes,
        .stim_steps = stim_steps,
        .readout_steps = readout_steps,
        .final_window = final_window,
        .target_delay = target_delay,
        .pass_mean = pass_mean,
        .pass_min = pass_min,
    });

    // ---- recurrent-state analysis --------------------------------------
    try analyzeRetention(gpa, io, seeds[0]);

    // ---- verdict --------------------------------------------------------
    const target_mean = target_sum / @as(f64, @floatFromInt(seeds.len));
    const pass = target_mean >= pass_mean and target_min >= pass_min;
    try o.print(
        \\
        \\  exit criterion -- "memory" condition at target delay {d} (chance = 0.500):
        \\    mean             {d:.3}   (need >= {d:.2})
        \\    worst seed       {d:.3}   (need >= {d:.2})
        \\  VERDICT: {s}
        \\
        \\  wrote delay.csv, retention.csv
        \\
    , .{
        target_delay,
        target_mean,
        pass_mean,
        target_min,
        pass_min,
        if (pass) "PASS -- retains the stimulus across the delay, above chance." else "FAIL -- see accuracy vs delay above.",
    });
    try o.flush();
}

/// Recurrent-state analysis: train a memory network, then over evaluation
/// episodes record, at each timestep of a long trial, the mean firing rate of
/// the stimulus's OWN input assembly vs the OTHER assembly. The own-assembly
/// rate staying high through the delay while the other stays low is the direct
/// neural signature of stimulus-specific retention.
fn analyzeRetention(gpa: std.mem.Allocator, io: std.Io, seed: u64) !void {
    const analysis_delay: u32 = 40;
    const trial_len = stim_steps + analysis_delay + readout_steps;
    const n_eval: u32 = 200;

    const c = baseConfig(seed, memory_weight);
    var s = try sim.Sim.init(gpa, c);
    defer s.deinit(gpa);
    const l = task.layout(c);
    const ext = try gpa.alloc(f32, c.n_neurons);
    defer gpa.free(ext);

    // Train (at the target delay) so the readout is realistic; retention of the
    // input assembly does not depend on the readout, but we keep it consistent.
    for (0..n_episodes) |ep| _ = runEpisode(&s, l, ext, seed, @intCast(ep), target_delay);

    const own = try gpa.alloc(f64, trial_len);
    defer gpa.free(own);
    const other = try gpa.alloc(f64, trial_len);
    defer gpa.free(other);
    @memset(own, 0);
    @memset(other, 0);

    // Frozen evaluation: no reward, no homeostasis -- just watch the activity.
    for (0..n_eval) |e| {
        const ep: u32 = @intCast(n_episodes + e);
        s.resetEpisode();
        var trng = rng.derived(seed, .task, ep);
        const choice: task.Choice = if (trng.below(2) == 0) .a else .b;
        l.fillStimulus(choice, c.task_input_current, ext);

        const own_grp = if (choice == .a) l.input_a else l.input_b;
        const other_grp = if (choice == .a) l.input_b else l.input_a;

        for (0..trial_len) |t| {
            const input_on = t < stim_steps;
            _ = s.step(if (input_on) ext else null);
            const fired = s.network.neurons.fired;
            var own_spikes: u32 = 0;
            var other_spikes: u32 = 0;
            for (own_grp.lo..own_grp.hi) |i| own_spikes += @intFromBool(fired[i]);
            for (other_grp.lo..other_grp.hi) |i| other_spikes += @intFromBool(fired[i]);
            own[t] += @floatFromInt(own_spikes);
            other[t] += @floatFromInt(other_spikes);
        }
    }

    const norm = @as(f64, @floatFromInt(n_eval)) * @as(f64, @floatFromInt(c.task_group_size));
    var out: std.Io.Writer.Allocating = .init(gpa);
    defer out.deinit();
    try out.writer.print("t,phase,own_assembly_rate,other_assembly_rate\n", .{});
    for (0..trial_len) |t| {
        const phase = if (t < stim_steps) "stimulus" else if (t < stim_steps + analysis_delay) "delay" else "readout";
        try out.writer.print("{d},{s},{d:.5},{d:.5}\n", .{ t, phase, own[t] / norm, other[t] / norm });
    }
    try writeAtomic(io, "retention.csv", out.written());
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
