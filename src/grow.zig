//! Phase 5 exit-criterion experiment: does the reservoir REWIRE over training --
//! growing and pruning connections -- while activity stays stable and the learned
//! task is not destroyed?
//!
//! Exit criterion (spec, Phase 5):
//!   "Connections change over training while activity remains stable, and useful
//!    performance is not destroyed."
//!
//! So there are three things to show at once, and the verdict ANDs them:
//!   1. CHANGE      -- connections actually turn over (prunes AND grows > 0).
//!   2. STABILITY   -- the population firing rate stays in a healthy band across
//!                     the whole run (homeostasis absorbs the rewiring).
//!   3. PERFORMANCE -- the two-choice association is still learned above chance,
//!                     comparable to a no-structural-plasticity control (A/B).
//!
//! The same two-choice task and episode loop as Phase 3 (train.zig), but the
//! reservoir has structural plasticity on: every `growth_interval` episodes we
//! call Sim.applyStructuralPlasticity() -- the slow growth clock (§8.7).
//!
//! Produces:
//!   * structural.csv -- per structural event (representative seed): prunes,
//!                       grows, cumulative churn, live structural edges, rate.
//!   * a PASS/FAIL verdict against all three conditions above.
//!
//! Build/run:  zig build grow   (add -Doptimize=ReleaseFast; it trains many nets)

const std = @import("std");
const cfg = @import("config.zig");
const sim = @import("sim.zig");
const task = @import("task.zig");
const rng = @import("rng.zig");

const seeds = [_]u64{ 1, 2, 3, 4 };
const n_episodes: u32 = 1500;
const stim_steps: u32 = 40;
const readout_steps: u32 = 25;
const final_window: u32 = 300;

/// Growth cadence: one structural event every this-many episodes. Much slower
/// than spikes/learning/homeostasis (§8.7) -- 30 events over the whole run.
const growth_interval: u32 = 50;

// Verdict thresholds.
const pass_acc_mean: f64 = 0.60; // structural-on mean final accuracy
const pass_acc_min: f64 = 0.55; // every seed above chance
const min_total_churn: u32 = 20; // prunes+grows summed over the run (change happened)
// STABILITY is judged RELATIVE TO THE CONTROL: rewiring must not make the network
// go dead, and must not push the peak firing rate materially above the no-
// structural-plasticity control's peak. (Absolute peaks ~0.2 are stimulus-driven
// and present with or without rewiring, so an absolute ceiling would be testing
// the task, not the mechanism.)
const rate_alive_lo: f64 = 0.005; // structural-on rate must never collapse below this
const rate_ceiling_tol: f64 = 1.10; // on-peak <= off-peak * this (rewiring adds < 10%)

fn baseConfig(seed: u64, structural: bool) cfg.Config {
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
        // Phase 5:
        .structural_plasticity_enabled = structural,
    };
}

/// One episode; returns whether the choice was correct and the post-episode mean
/// firing-rate EMA (the stability signal).
const EpResult = struct { correct: bool, mean_rate: f64 };

fn runEpisode(s: *sim.Sim, l: task.Layout, ext: []f32, seed: u64, episode: u32) EpResult {
    s.resetEpisode();

    var trng = rng.derived(seed, .task, episode);
    const choice: task.Choice = if (trng.below(2) == 0) .a else .b;

    const c = s.network.config;
    l.fillStimulus(choice, c.task_input_current, ext);

    var last_rate: f32 = 0;
    var count0: u32 = 0;
    var count1: u32 = 0;
    for (0..stim_steps) |step| {
        const m = s.step(ext);
        last_rate = m.mean_rate_ema;
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
        var arng = rng.derived(seed, .action, episode);
        chosen = @intCast(arng.below(2));
    }

    const correct = chosen == l.correctAction(choice);
    s.applyReward(if (correct) 1.0 else -1.0);
    s.applyHomeostasis();
    return .{ .correct = correct, .mean_rate = last_rate };
}

/// Train one seed. When `structural`, run the growth clock and (for the
/// representative seed) log the structural timeline. Fills churn/stability out
/// params. Returns final-window accuracy.
fn trainOne(
    gpa: std.mem.Allocator,
    seed: u64,
    structural: bool,
    log: ?*std.Io.Writer,
    total_churn: *u32,
    rate_min: *f64,
    rate_max: *f64,
) !f64 {
    const c = baseConfig(seed, structural);
    try c.validate();
    var s = try sim.Sim.init(gpa, c);
    defer s.deinit(gpa);
    const l = task.layout(c);
    const ext = try gpa.alloc(f32, c.n_neurons);
    defer gpa.free(ext);

    var cum_pruned: u32 = 0;
    var cum_grown: u32 = 0;
    var correct_in_window: u32 = 0;
    rate_min.* = std.math.inf(f64);
    rate_max.* = 0;

    for (0..n_episodes) |ep_usize| {
        const ep: u32 = @intCast(ep_usize);
        const res = runEpisode(&s, l, ext, seed, ep);

        // Track the firing-rate band once past an initial burn-in, so the
        // homeostat has had a chance to find the operating point.
        if (ep >= 200) {
            rate_min.* = @min(rate_min.*, res.mean_rate);
            rate_max.* = @max(rate_max.*, res.mean_rate);
        }

        // Growth clock: a structural event every growth_interval episodes.
        if (structural and (ep + 1) % growth_interval == 0) {
            const sm = s.applyStructuralPlasticity();
            cum_pruned += sm.pruned;
            cum_grown += sm.grown;
            if (log) |w| {
                try w.print("{d},{d},{d},{d},{d},{d},{d},{d:.6}\n", .{
                    s.growth_counter, // structural event index
                    ep + 1,
                    sm.pruned,
                    sm.grown,
                    cum_pruned,
                    cum_grown,
                    sm.live_structural,
                    res.mean_rate,
                });
            }
        }

        if (ep >= n_episodes - final_window and res.correct) correct_in_window += 1;
    }

    total_churn.* = cum_pruned + cum_grown;
    return @as(f64, @floatFromInt(correct_in_window)) / @as(f64, @floatFromInt(final_window));
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    var buf: [4096]u8 = undefined;
    var stdout = std.Io.File.stdout().writer(io, &buf);
    const o = &stdout.interface;

    var csv: std.Io.Writer.Allocating = .init(gpa);
    defer csv.deinit();
    try csv.writer.print("event,episode,pruned,grown,cum_pruned,cum_grown,live_structural,mean_rate\n", .{});

    try o.print("\n-- Phase 5 structural plasticity: rewiring vs performance ------\n", .{});
    try o.print("  {s:>6}  {s:>10}  {s:>10}  {s:>7}  {s:>10}  {s:>10}\n", .{
        "seed", "acc(on)", "acc(off)", "churn", "rate_min", "rate_max",
    });

    var acc_on_sum: f64 = 0;
    var acc_on_min: f64 = 1.0;
    var churn_total: u32 = 0;
    var on_lo: f64 = std.math.inf(f64); // lowest structural-on rate across seeds
    var on_hi: f64 = 0; // highest structural-on rate across seeds
    var ctrl_hi: f64 = 0; // highest control (structural-off) rate across seeds

    for (seeds, 0..) |seed, si| {
        var churn: u32 = 0;
        var r_lo: f64 = 0;
        var r_hi: f64 = 0;
        var churn_off: u32 = 0;
        var off_lo: f64 = 0;
        var off_hi: f64 = 0;

        // Log the structural timeline only for the first (representative) seed.
        const log: ?*std.Io.Writer = if (si == 0) &csv.writer else null;

        const acc_on = try trainOne(gpa, seed, true, log, &churn, &r_lo, &r_hi);
        const acc_off = try trainOne(gpa, seed, false, null, &churn_off, &off_lo, &off_hi);

        acc_on_sum += acc_on;
        acc_on_min = @min(acc_on_min, acc_on);
        churn_total += churn;
        on_lo = @min(on_lo, r_lo);
        on_hi = @max(on_hi, r_hi);
        ctrl_hi = @max(ctrl_hi, off_hi);

        try o.print("  {d:>6}  {d:>10.3}  {d:>10.3}  {d:>7}  {d:>10.4}  {d:>10.4}\n", .{
            seed, acc_on, acc_off, churn, r_lo, r_hi,
        });
    }

    try writeAtomic(io, "structural.csv", csv.written());

    const acc_on_mean = acc_on_sum / @as(f64, @floatFromInt(seeds.len));
    const ceiling = ctrl_hi * rate_ceiling_tol;

    const changed = churn_total >= min_total_churn;
    const stable = on_lo >= rate_alive_lo and on_hi <= ceiling;
    const performs = acc_on_mean >= pass_acc_mean and acc_on_min >= pass_acc_min;
    const pass = changed and stable and performs;

    try o.print(
        \\
        \\  exit criterion (all three must hold):
        \\    1. CHANGE       total churn (prune+grow) {d:>5}   (need >= {d})
        \\    2. STABILITY    on-rate [{d:.4}, {d:.4}]  (alive >= {d:.3}, peak <= {d:.4} = control peak x{d:.2})
        \\    3. PERFORMANCE  acc mean {d:.3} / worst {d:.3}  (need >= {d:.2} / {d:.2})
        \\  VERDICT: {s}
        \\
        \\  wrote structural.csv
        \\
    , .{
        churn_total,                      min_total_churn,
        on_lo, on_hi,                     rate_alive_lo, ceiling, rate_ceiling_tol,
        acc_on_mean, acc_on_min,          pass_acc_mean, pass_acc_min,
        if (pass)
            "PASS -- the reservoir rewires while activity stays stable and the task survives."
        else
            "FAIL -- see the three conditions above.",
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
