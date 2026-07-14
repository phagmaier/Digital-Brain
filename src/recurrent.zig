//! Stage 2 exit-criterion experiment (report.md / final.md §7):
//! can *locally plastic recurrent (reservoir) edges* become necessary for a
//! context-dependent delayed mapping that a fixed-reservoir + plastic-readout
//! system cannot solve as well?
//!
//! Task — delayed XOR / cross-coupling (DEC-014 / context_task.zig):
//!   context X + cue A → action 0
//!   context X + cue B → action 1
//!   context Y + cue A → action 1
//!   context Y + cue B → action 0
//!
//! Context and cue are presented at separate times. Plastic stimulus→action
//! readout edges exist from both context and cue assemblies; a pure linear
//! combination of those four assemblies cannot implement XOR (not linearly
//! separable), so the direct path is not a sufficient shortcut. Context
//! assemblies also carry fixed self-excitation so the context can bridge the
//! inter-stimulus delay. Solving the mapping therefore requires a
//! context-dependent recurrent state when the cue arrives.
//!
//! Conditions (paired over seeds):
//!   1. readout_only      — fixed reservoir + plastic cue readout
//!   2. structural_only   — structural rewiring only (no reservoir weight learning)
//!   3. recurrent         — locally plastic reservoir edges + plastic cue readout
//!   4. recurrent_consol  — recurrent + structural + reward-gated consolidation
//!   5. lesion            — after `recurrent` training, restore reservoir plastic
//!                          weights to their post-build init and retest frozen
//!
//! Success is NOT "above chance". The verdict requires:
//!   * recurrent accuracy substantially above readout-only (CI lower bound)
//!   * lesion of the learned recurrent changes collapses that advantage
//!   * representational probe: reservoir activity during the cue is more
//!     separable by context under recurrent plasticity than under readout-only
//!
//! Produces:
//!   * recurrent.csv          — per (condition, seed) accuracy / lesion / probe
//!   * recurrent_repr.csv     — mean reservoir rate vectors by context (seed 1)
//!   * recurrent.meta.json    — provenance
//!
//! Build/run:  zig build recurrent -Doptimize=ReleaseFast

const std = @import("std");
const cfg = @import("config.zig");
const sim = @import("sim.zig");
const ctx_task = @import("context_task.zig");
const rng = @import("rng.zig");
const provenance = @import("provenance.zig");

const n_seeds = 12; // Stage 2: paired seeds; raise toward 20–50 for publication runs
const seeds: [n_seeds]u64 = blk: {
    var s: [n_seeds]u64 = undefined;
    for (0..n_seeds) |i| s[i] = @as(u64, i) + 1;
    break :blk s;
};

const n_episodes: u32 = 2500;
const final_window: u32 = 400;
const stim_steps: u32 = 30;
const delay_after_context: u32 = 10;
const cue_steps: u32 = 30;
const delay_after_cue: u32 = 5;
const readout_steps: u32 = 25;
const growth_interval: u32 = 50;
const n_eval_probe: u32 = 120;
const n_eval_lesion: u32 = 200;

// Verdict thresholds (judged on 95% CI lower bounds over the paired seed sample).
// Stage 2 is the open scientific criterion: an honest FAIL is acceptable and
// expected until recurrent credit assignment is shown to clear these bars.
const pass_recurrent_mean: f64 = 0.58;
const pass_gap_readout: f64 = 0.06;
const pass_gap_lesion: f64 = 0.05;
const pass_sep_gap: f64 = 0.0; // recurrent separability ≥ readout-only

const TrainCondition = enum {
    readout_only,
    structural_only,
    recurrent,
    recurrent_consol,
};

fn trainName(cond: TrainCondition) []const u8 {
    return switch (cond) {
        .readout_only => "readout_only",
        .structural_only => "structural_only",
        .recurrent => "recurrent",
        .recurrent_consol => "recurrent_consol",
    };
}

const train_conditions = [_]TrainCondition{
    .readout_only,
    .structural_only,
    .recurrent,
    .recurrent_consol,
};

fn baseConfig(seed: u64, cond: TrainCondition) cfg.Config {
    const reservoir_plastic = cond == .recurrent or cond == .recurrent_consol;
    const structural = cond == .structural_only or cond == .recurrent_consol;
    const consol = cond == .recurrent_consol;
    return .{
        .master_seed = seed,
        .n_neurons = 100,
        .context_task_enabled = true,
        .context_task_group_size = 6,
        .context_hold_weight = 0.65,
        .plasticity_enabled = true,
        .reservoir_plasticity_enabled = reservoir_plastic,
        .structural_plasticity_enabled = structural,
        .consolidation_enabled = consol,
        .consolidation_lr = 0.03,
        .homeostasis_enabled = true,
        .homeostasis_per_step = false,
        .target_rate = 0.05,
        .homeostasis_lr = 0.05,
        .adaptation_enabled = false,
        .eligibility_decay = 0.99,
        .learning_rate = if (reservoir_plastic) 0.03 else 0.05,
        .task_input_current = 1.5,
        .task_ia_weight_init = 0.25,
        .background_current = 0.35,
        // Structural knobs (only active when structural is on).
        .max_out_degree = 20,
        .target_out_degree = 10,
        .growth_probability = if (cond == .structural_only) 0.15 else 0.08,
        .structural_interval_steps = 0, // harness drives the slow clock
    };
}

const EpisodeDraw = struct {
    context: ctx_task.Context,
    cue: ctx_task.Cue,
};

fn drawEpisode(seed: u64, episode: u32) EpisodeDraw {
    var trng = rng.derived(seed, .task, episode);
    const context: ctx_task.Context = if (trng.below(2) == 0) .x else .y;
    const cue: ctx_task.Cue = if (trng.below(2) == 0) .a else .b;
    return .{ .context = context, .cue = cue };
}

fn chooseAction(count0: u32, count1: u32, seed: u64, episode: u32) u1 {
    if (count0 > count1) return 0;
    if (count1 > count0) return 1;
    var arng = rng.derived(seed, .action, episode);
    return @intCast(arng.below(2));
}

fn runEpisode(
    s: *sim.Sim,
    l: ctx_task.Layout,
    ext: []f32,
    seed: u64,
    episode: u32,
    learn: bool,
) bool {
    s.resetEpisode();
    const draw = drawEpisode(seed, episode);
    const c = s.network.config;

    l.fillContext(draw.context, c.task_input_current, ext);
    for (0..stim_steps) |_| _ = s.step(ext);
    for (0..delay_after_context) |_| _ = s.step(null);

    l.fillCue(draw.cue, c.task_input_current, ext);
    for (0..cue_steps) |_| _ = s.step(ext);
    for (0..delay_after_cue) |_| _ = s.step(null);

    var count0: u32 = 0;
    var count1: u32 = 0;
    for (0..readout_steps) |_| {
        _ = s.step(null);
        const fired = s.network.neurons.fired;
        for (l.action_0.lo..l.action_0.hi) |i| count0 += @intFromBool(fired[i]);
        for (l.action_1.lo..l.action_1.hi) |i| count1 += @intFromBool(fired[i]);
    }

    const chosen = chooseAction(count0, count1, seed, episode);
    const correct = chosen == l.correctAction(draw.context, draw.cue);
    if (learn) {
        s.applyReward(if (correct) 1.0 else -1.0);
        s.applyHomeostasis();
    }
    return correct;
}

const TrainResult = struct {
    accuracy: f64,
    /// Frozen retest after restoring structural plastic weights to post-build init.
    lesion_accuracy: f64,
    /// Mean |rate_x − rate_y| over non-reserved neurons during the cue window.
    context_separability: f64,
};

fn measureAccuracy(s: *sim.Sim, l: ctx_task.Layout, ext: []f32, seed: u64, base_ep: u32, n: u32) f64 {
    var correct: u32 = 0;
    for (0..n) |i| {
        const ep: u32 = base_ep + @as(u32, @intCast(i));
        if (runEpisode(s, l, ext, seed, ep, false)) correct += 1;
    }
    return @as(f64, @floatFromInt(correct)) / @as(f64, @floatFromInt(n));
}

/// During the cue window, accumulate mean firing rates of non-reserved neurons
/// conditioned on the active context. Separability = mean absolute difference
/// between the two context-conditioned rate vectors.
fn measureSeparability(
    gpa: std.mem.Allocator,
    s: *sim.Sim,
    l: ctx_task.Layout,
    ext: []f32,
    seed: u64,
    base_ep: u32,
    /// Optional per-neuron dump: writes (id, reserved, rate_x, rate_y, abs_diff).
    repr_out: ?*std.Io.Writer.Allocating,
) !f64 {
    const n = s.network.neurons.n;
    const sum_x = try gpa.alloc(f64, n);
    defer gpa.free(sum_x);
    const sum_y = try gpa.alloc(f64, n);
    defer gpa.free(sum_y);
    @memset(sum_x, 0);
    @memset(sum_y, 0);
    var n_x: u32 = 0;
    var n_y: u32 = 0;

    const c = s.network.config;
    for (0..n_eval_probe) |i| {
        const ep: u32 = base_ep + @as(u32, @intCast(i));
        s.resetEpisode();
        const draw = drawEpisode(seed, ep);

        l.fillContext(draw.context, c.task_input_current, ext);
        for (0..stim_steps) |_| _ = s.step(ext);
        for (0..delay_after_context) |_| _ = s.step(null);

        l.fillCue(draw.cue, c.task_input_current, ext);
        // Accumulate over the second half of the cue window (settled cue + held context).
        for (0..cue_steps) |t| {
            _ = s.step(ext);
            if (t < cue_steps / 2) continue;
            const fired = s.network.neurons.fired;
            const dest = if (draw.context == .x) sum_x else sum_y;
            for (0..n) |j| {
                if (l.isReserved(@intCast(j))) continue;
                dest[j] += @floatFromInt(@intFromBool(fired[j]));
            }
        }
        if (draw.context == .x) n_x += 1 else n_y += 1;
    }
    if (n_x == 0 or n_y == 0) return 0;

    const half: f64 = @floatFromInt(cue_steps - cue_steps / 2);
    var diff_sum: f64 = 0;
    var count: u32 = 0;
    if (repr_out) |out| {
        try out.writer.print("neuron_id,reserved,rate_context_x,rate_context_y,abs_diff\n", .{});
    }
    for (0..n) |j| {
        const rx = sum_x[j] / (@as(f64, @floatFromInt(n_x)) * half);
        const ry = sum_y[j] / (@as(f64, @floatFromInt(n_y)) * half);
        const reserved = l.isReserved(@intCast(j));
        if (repr_out) |out| {
            try out.writer.print("{d},{d},{d:.6},{d:.6},{d:.6}\n", .{
                j,
                @intFromBool(reserved),
                rx,
                ry,
                @abs(rx - ry),
            });
        }
        if (reserved) continue;
        diff_sum += @abs(rx - ry);
        count += 1;
    }
    if (count == 0) return 0;
    return diff_sum / @as(f64, @floatFromInt(count));
}

fn trainOne(
    gpa: std.mem.Allocator,
    seed: u64,
    cond: TrainCondition,
    /// When non-null and cond is recurrent, fill with the per-neuron rate dump.
    repr_out: ?*std.Io.Writer.Allocating,
) !TrainResult {
    const c = baseConfig(seed, cond);
    var s = try sim.Sim.init(gpa, c);
    defer s.deinit(gpa);
    const l = ctx_task.layout(c);
    const ext = try gpa.alloc(f32, c.n_neurons);
    defer gpa.free(ext);

    // Snapshot post-build weights so the lesion can restore *learned recurrent
    // changes* without touching the cue readout.
    const init_w = try gpa.dupe(f32, s.network.synapses.weight);
    defer gpa.free(init_w);

    var correct_window: u32 = 0;
    for (0..n_episodes) |ep| {
        const correct = runEpisode(&s, l, ext, seed, @intCast(ep), true);
        if (ep >= n_episodes - final_window and correct) correct_window += 1;
        if (c.structural_plasticity_enabled and (ep + 1) % growth_interval == 0)
            _ = s.applyStructuralPlasticity();
    }
    const accuracy = @as(f64, @floatFromInt(correct_window)) / @as(f64, @floatFromInt(final_window));

    // Representational probe (frozen). Dump vectors only for the requested run.
    const want_repr = repr_out != null and cond == .recurrent;
    const sep = try measureSeparability(
        gpa,
        &s,
        l,
        ext,
        seed,
        n_episodes,
        if (want_repr) repr_out else null,
    );

    // Lesion only meaningful when reservoir edges were plastic and trained.
    var lesion_acc: f64 = accuracy;
    if (c.reservoir_plasticity_enabled) {
        const syn = &s.network.synapses;
        for (0..syn.n) |k| {
            if (syn.structural[k] and syn.plastic[k] and syn.alive[k])
                syn.weight[k] = init_w[k];
        }
        lesion_acc = measureAccuracy(&s, l, ext, seed, n_episodes + n_eval_probe, n_eval_lesion);
    }

    return .{
        .accuracy = accuracy,
        .lesion_accuracy = lesion_acc,
        .context_separability = sep,
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

    var stdout_buf: [4096]u8 = undefined;
    var stdout = std.Io.File.stdout().writer(io, &stdout_buf);
    const o = &stdout.interface;

    var csv: std.Io.Writer.Allocating = .init(gpa);
    defer csv.deinit();
    try csv.writer.print(
        "condition,seed,accuracy,lesion_accuracy,context_separability\n",
        .{},
    );

    var acc_readout: [n_seeds]f64 = undefined;
    var acc_structural: [n_seeds]f64 = undefined;
    var acc_recurrent: [n_seeds]f64 = undefined;
    var acc_consol: [n_seeds]f64 = undefined;
    var acc_lesion: [n_seeds]f64 = undefined;
    var sep_readout: [n_seeds]f64 = undefined;
    var sep_recurrent: [n_seeds]f64 = undefined;
    var gap_readout: [n_seeds]f64 = undefined;
    var gap_lesion: [n_seeds]f64 = undefined;
    var gap_sep: [n_seeds]f64 = undefined;

    try o.print(
        \\
        \\-- Stage 2: recurrent plasticity on context-dependent delayed XOR --
        \\  {d} paired seeds × {d} train conditions, {d} episodes
        \\  episode: context {d} → delay {d} → cue {d} → delay {d} → readout {d}
        \\
    , .{
        n_seeds,
        train_conditions.len,
        n_episodes,
        stim_steps,
        delay_after_context,
        cue_steps,
        delay_after_cue,
        readout_steps,
    });

    var repr: std.Io.Writer.Allocating = .init(gpa);
    defer repr.deinit();

    for (seeds, 0..) |seed, si| {
        try o.print("  seed {d}:\n", .{seed});
        for (train_conditions) |cond| {
            const dump_repr = (seed == seeds[0] and cond == .recurrent);
            const r = try trainOne(gpa, seed, cond, if (dump_repr) &repr else null);
            try csv.writer.print("{s},{d},{d:.4},{d:.4},{d:.6}\n", .{
                trainName(cond),
                seed,
                r.accuracy,
                r.lesion_accuracy,
                r.context_separability,
            });
            try o.print("    {s:<18} acc={d:.3}  lesion={d:.3}  sep={d:.4}\n", .{
                trainName(cond),
                r.accuracy,
                r.lesion_accuracy,
                r.context_separability,
            });
            switch (cond) {
                .readout_only => {
                    acc_readout[si] = r.accuracy;
                    sep_readout[si] = r.context_separability;
                },
                .structural_only => acc_structural[si] = r.accuracy,
                .recurrent => {
                    acc_recurrent[si] = r.accuracy;
                    acc_lesion[si] = r.lesion_accuracy;
                    sep_recurrent[si] = r.context_separability;
                },
                .recurrent_consol => acc_consol[si] = r.accuracy,
            }
        }
        gap_readout[si] = acc_recurrent[si] - acc_readout[si];
        gap_lesion[si] = acc_recurrent[si] - acc_lesion[si];
        gap_sep[si] = sep_recurrent[si] - sep_readout[si];
    }

    try writeAtomic(io, "recurrent.csv", csv.written());
    try writeAtomic(io, "recurrent_repr.csv", repr.written());
    try provenance.write(io, gpa, "recurrent.meta.json", "recurrent_context_xor", .{
        .seeds = seeds,
        .config_templates = .{
            .readout_only = baseConfig(seeds[0], .readout_only),
            .structural_only = baseConfig(seeds[0], .structural_only),
            .recurrent = baseConfig(seeds[0], .recurrent),
            .recurrent_consol = baseConfig(seeds[0], .recurrent_consol),
        },
        .n_episodes = n_episodes,
        .final_window = final_window,
        .stim_steps = stim_steps,
        .delay_after_context = delay_after_context,
        .cue_steps = cue_steps,
        .delay_after_cue = delay_after_cue,
        .readout_steps = readout_steps,
        .growth_interval = growth_interval,
        .n_eval_probe = n_eval_probe,
        .n_eval_lesion = n_eval_lesion,
        .pass_recurrent_mean = pass_recurrent_mean,
        .pass_gap_readout = pass_gap_readout,
        .pass_gap_lesion = pass_gap_lesion,
        .pass_sep_gap = pass_sep_gap,
    });

    const st_readout = summarize(&acc_readout);
    const st_structural = summarize(&acc_structural);
    const st_recurrent = summarize(&acc_recurrent);
    const st_consol = summarize(&acc_consol);
    const st_lesion = summarize(&acc_lesion);
    const st_gap_r = summarize(&gap_readout);
    const st_gap_l = summarize(&gap_lesion);
    const st_sep_r = summarize(&sep_readout);
    const st_sep_rec = summarize(&sep_recurrent);
    const st_gap_sep = summarize(&gap_sep);

    const rec_lo = st_recurrent.mean - st_recurrent.ci_half;
    const gap_r_lo = st_gap_r.mean - st_gap_r.ci_half;
    const gap_l_lo = st_gap_l.mean - st_gap_l.ci_half;
    const sep_lo = st_gap_sep.mean - st_gap_sep.ci_half;

    const pass = rec_lo >= pass_recurrent_mean and
        gap_r_lo >= pass_gap_readout and
        gap_l_lo >= pass_gap_lesion and
        sep_lo >= pass_sep_gap;

    try o.print(
        \\
        \\  summary (mean ± 95% CI half-width over {d} paired seeds):
        \\    readout_only       {d:.3} ± {d:.3}
        \\    structural_only    {d:.3} ± {d:.3}
        \\    recurrent          {d:.3} ± {d:.3}   (lo={d:.3}, need ≥ {d:.2})
        \\    recurrent_consol   {d:.3} ± {d:.3}
        \\    lesion (recurrent) {d:.3} ± {d:.3}
        \\    gap recurrent−readout  {d:.3} ± {d:.3}   (lo={d:.3}, need ≥ {d:.2})
        \\    gap recurrent−lesion   {d:.3} ± {d:.3}   (lo={d:.3}, need ≥ {d:.2})
        \\    sep readout / recurrent {d:.4} / {d:.4}   (Δ lo={d:.4}, need ≥ {d:.2})
        \\
        \\  VERDICT: {s}
        \\
        \\  wrote recurrent.csv, recurrent_repr.csv, recurrent.meta.json
        \\
    , .{
        n_seeds,
        st_readout.mean,
        st_readout.ci_half,
        st_structural.mean,
        st_structural.ci_half,
        st_recurrent.mean,
        st_recurrent.ci_half,
        rec_lo,
        pass_recurrent_mean,
        st_consol.mean,
        st_consol.ci_half,
        st_lesion.mean,
        st_lesion.ci_half,
        st_gap_r.mean,
        st_gap_r.ci_half,
        gap_r_lo,
        pass_gap_readout,
        st_gap_l.mean,
        st_gap_l.ci_half,
        gap_l_lo,
        pass_gap_lesion,
        st_sep_r.mean,
        st_sep_rec.mean,
        sep_lo,
        pass_sep_gap,
        if (pass)
            "PASS — recurrent plasticity beats readout-only, lesion collapses the gain, context-separable reservoir state."
        else
            "FAIL — Stage 2 criterion not met (see gaps / CIs above). Mechanism code is in place; tune / re-seed as needed.",
    });
    try o.flush();
}
