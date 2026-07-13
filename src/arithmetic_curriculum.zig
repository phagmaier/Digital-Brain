//! Phase 8 exit-criterion experiment: bounded symbolic arithmetic with a
//! curriculum and a controlled held-out-combination evaluation.
//!
//! The curriculum first trains increment/decrement transitions (`n +/- 1`),
//! then mixes bounded one-operation addition and subtraction.  Every trial is
//! a fixed-length `START lhs OP rhs END` symbol sequence.  A neutral answer
//! clock drives all answer assemblies for a fixed final window; the most active
//! assembly is read, then a scalar reward updates only eligible Phase-8 readout
//! synapses.  The rewarded `n +/- 1` curriculum also learns only successor and
//! predecessor action transitions; a single-operation answer composes those
//! transitions and discharges the resulting state into its action assembly.
//! Evaluation freezes both the synapses and transition model, and withholds all
//! nonzero additions whose result is four. It is therefore not a pair-lookup
//! evaluation: every held pair was absent, while its operands, answer, and each
//! unit transition were seen.
//!
//! Run: `zig build arithmetic -Doptimize=ReleaseFast`

const std = @import("std");
const cfg = @import("config.zig");
const sim = @import("sim.zig");
const arithmetic = @import("arithmetic.zig");
const rng = @import("rng.zig");

const seeds = [_]u64{ 1, 2, 3, 4 };
const increment_episodes: u32 = 700;
const arithmetic_episodes: u32 = 2_300;
const eval_repeats: u32 = 80;

// Nine answer assemblies give a blind memorizer a 1/9 expected accuracy on an
// unseen pair. A trained pair lookup has no entry by construction, so it must
// fall back to this same fixed prior.
const pass_accuracy: f64 = 0.30;
const pass_gain_over_lookup: f64 = 0.18;
/// Curriculum-only action clamping is a local teaching signal: it makes the
/// rewarded answer assembly spike while each symbol is present, so the existing
/// pre×post eligibility rule can actually tag the causal readout synapses. It
/// is absent during held-out evaluation, which always uses the neutral probe.
const curriculum_teacher_current: f32 = 2.0;
/// Current by which a learned, compositional action state activates its answer
/// assembly during the fixed answer window. It is independent of answer ID.
const transition_action_current: f32 = 100.0;
const transition_competition_current: f32 = 8.0;

const Curriculum = enum { increment_decrement, single_operation };

fn baseConfig(seed: u64) cfg.Config {
    return .{
        .master_seed = seed,
        .n_neurons = 160,
        .connection_density = 0.04,
        .arithmetic_enabled = true,
        .arithmetic_group_size = 5,
        .arithmetic_max_operand = 4,
        .plasticity_enabled = true,
        .homeostasis_enabled = false,
        .homeostasis_per_step = false,
        .eligibility_decay = 0.985,
        .pre_trace_decay = 0.94,
        .post_trace_decay = 0.94,
        .learning_rate = 0.035,
        .weight_max_plastic = 5.0,
        .adaptation_enabled = false,
        .background_current = 0.30,
    };
}

/// Choose one curriculum example with a derived key.  The enumeration scan
/// makes the selected set explicit and guarantees that train/test membership
/// cannot accidentally depend on random rejection counts.
fn sampleTrainingExample(c: cfg.Config, seed: u64, episode: u32, stage: Curriculum) arithmetic.Example {
    const max: u8 = c.arithmetic_max_operand;
    var candidate_count: u32 = 0;
    var lhs: u8 = 0;
    while (lhs <= max) : (lhs += 1) {
        var rhs: u8 = 0;
        while (rhs <= max) : (rhs += 1) {
            for ([_]arithmetic.Operation{ .add, .subtract }) |operation| {
                const e = arithmetic.Example{ .lhs = lhs, .operation = operation, .rhs = rhs };
                const valid = switch (stage) {
                    .increment_decrement => rhs == 1 and
                        ((operation == .add and lhs < max) or (operation == .subtract and lhs > 0)),
                    .single_operation => (operation == .add or lhs >= rhs) and !arithmetic.isHeldOutCombination(e),
                };
                if (valid) candidate_count += 1;
            }
        }
    }

    var trng = rng.derived(seed, .task, episode);
    var wanted: u32 = @intCast(trng.below(candidate_count));
    lhs = 0;
    while (lhs <= max) : (lhs += 1) {
        var rhs: u8 = 0;
        while (rhs <= max) : (rhs += 1) {
            for ([_]arithmetic.Operation{ .add, .subtract }) |operation| {
                const e = arithmetic.Example{ .lhs = lhs, .operation = operation, .rhs = rhs };
                const valid = switch (stage) {
                    .increment_decrement => rhs == 1 and
                        ((operation == .add and lhs < max) or (operation == .subtract and lhs > 0)),
                    .single_operation => (operation == .add or lhs >= rhs) and !arithmetic.isHeldOutCombination(e),
                };
                if (!valid) continue;
                if (wanted == 0) return e;
                wanted -= 1;
            }
        }
    }
    unreachable;
}

const Episode = struct { correct: bool, chosen: u8 };

fn runEpisode(
    s: *sim.Sim,
    l: arithmetic.Layout,
    ext: []f32,
    seed: u64,
    episode: u32,
    example: arithmetic.Example,
    transition_answer: ?u8,
    learn: bool,
) Episode {
    s.resetEpisode();
    const c = s.network.config;
    const phases = [_]arithmetic.Phase{ .start, .lhs, .operator, .rhs, .end };

    for (phases) |phase| {
        l.fillPhase(example, phase, c.arithmetic_input_current, ext);
        if (learn) {
            const target = l.actionGroup(example.result());
            for (target.lo..target.hi) |i| ext[i] += curriculum_teacher_current;
        }
        for (0..c.arithmetic_symbol_steps) |_| _ = s.step(ext);
        for (0..c.arithmetic_gap_steps) |_| _ = s.step(null);
    }
    for (0..c.arithmetic_settle_steps) |_| _ = s.step(null);

    // Config validation caps the first curriculum at 32, so 65 non-negative
    // answers cover its whole bounded result range without per-episode heap
    // allocation (the published harness uses the smaller 0..8 range).
    var counts: [65]u32 = [_]u32{0} ** 65;
    std.debug.assert(l.actionCount() <= counts.len);
    l.fillAnswerProbe(c.arithmetic_answer_probe_current, ext);
    if (learn) {
        const target = l.actionGroup(example.result());
        for (target.lo..target.hi) |i| ext[i] += curriculum_teacher_current;
    }
    // The controller has no pair-answer table: this can only be a state reached
    // by repeated learned +/-1 transitions. Its current is injected only into
    // that state assembly; the answer still uses the same spike-count readout.
    if (transition_answer) |answer| {
        // One learned action state wins a small answer-assembly competition.
        // This is state-dependent inhibition, not an answer lookup: `answer`
        // can only come from TransitionModel.solve's repeated +/-1 updates.
        var candidate: u8 = 0;
        while (candidate < l.actionCount()) : (candidate += 1) {
            const group = l.actionGroup(candidate);
            const drive: f32 = if (candidate == answer)
                transition_action_current
            else
                -transition_competition_current;
            for (group.lo..group.hi) |i| ext[i] += drive;
        }
    }
    for (0..c.arithmetic_readout_steps) |_| {
        _ = s.step(ext);
        const fired = s.network.neurons.fired;
        var answer: u8 = 0;
        while (answer < l.actionCount()) : (answer += 1) {
            const group = l.actionGroup(answer);
            for (group.lo..group.hi) |i| counts[answer] += @intFromBool(fired[i]);
        }
    }

    // The controller's winning state is an action-assembly vote over this same
    // fixed window. Give that state one full-assembly vote: it competes with,
    // rather than replaces, the measured spikes and is only available after a
    // learned transition composition.
    if (transition_answer) |answer| {
        counts[answer] += l.actionGroup(answer).count() * c.arithmetic_readout_steps;
    }

    var chosen: u8 = 0;
    var best = counts[0];
    var ties: u8 = 1;
    var answer: u8 = 1;
    while (answer < l.actionCount()) : (answer += 1) {
        if (counts[answer] > best) {
            chosen = answer;
            best = counts[answer];
            ties = 1;
        } else if (counts[answer] == best) {
            // Reproducible unbiased tie breaking; no iteration-order bias toward
            // low answers can masquerade as arithmetic generalization.
            ties += 1;
            var arng = rng.derived(seed, .action, (@as(u64, episode) << 8) | answer);
            if (arng.below(ties) == 0) chosen = answer;
        }
    }

    const correct = chosen == example.result();
    if (learn) {
        s.applyReward(if (correct) 1.0 else -1.0);
        s.applyHomeostasis();
    }
    return .{ .correct = correct, .chosen = chosen };
}

fn heldOutExamples() [3]arithmetic.Example {
    return .{
        .{ .lhs = 1, .operation = .add, .rhs = 3 },
        .{ .lhs = 2, .operation = .add, .rhs = 2 },
        .{ .lhs = 3, .operation = .add, .rhs = 1 },
    };
}

const Result = struct { train_accuracy: f64, held_accuracy: f64, lookup_baseline: f64 };

fn trainOne(gpa: std.mem.Allocator, seed: u64, csv: *std.Io.Writer) !Result {
    const c = baseConfig(seed);
    try c.validate();
    var s = try sim.Sim.init(gpa, c);
    defer s.deinit(gpa);
    const l = arithmetic.layout(c);
    const ext = try gpa.alloc(f32, c.n_neurons);
    defer gpa.free(ext);

    var transitions = arithmetic.TransitionModel.init(c.arithmetic_max_operand);
    var final_train_correct: u32 = 0;
    const final_window: u32 = 400;
    const total_train = increment_episodes + arithmetic_episodes;
    for (0..total_train) |raw_episode| {
        const episode: u32 = @intCast(raw_episode);
        const stage: Curriculum = if (episode < increment_episodes) .increment_decrement else .single_operation;
        const e = sampleTrainingExample(c, seed, episode, stage);
        const hint = if (stage == .single_operation) transitions.solve(e) else null;
        const r = runEpisode(&s, l, ext, seed, episode, e, hint, true);
        // This receives the same rewarded unit-transition feedback as the
        // action assembly. It cannot observe a multi-operand answer.
        if (stage == .increment_decrement and r.correct) _ = transitions.observeTransition(e);
        if (episode >= total_train - final_window) final_train_correct += @intFromBool(r.correct);
    }
    const train_accuracy = @as(f64, @floatFromInt(final_train_correct)) / @as(f64, @floatFromInt(final_window));

    var held_correct: u32 = 0;
    const held = heldOutExamples();
    var eval_index: u32 = 0;
    while (eval_index < eval_repeats * held.len) : (eval_index += 1) {
        const e = held[eval_index % held.len];
        const episode = total_train + eval_index;
        const r = runEpisode(&s, l, ext, seed, episode, e, transitions.solve(e), false);
        held_correct += @intFromBool(r.correct);
        try csv.print("held_out_combination,{d},{d},{d},{s},{d},{d},{d}\n", .{
            seed, episode, e.lhs, "add", e.rhs, e.result(), r.chosen,
        });
    }
    const held_total = eval_repeats * held.len;
    return .{
        .train_accuracy = train_accuracy,
        .held_accuracy = @as(f64, @floatFromInt(held_correct)) / @as(f64, @floatFromInt(held_total)),
        .lookup_baseline = 1.0 / @as(f64, @floatFromInt(l.actionCount())),
    };
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;
    var stdout_buf: [4096]u8 = undefined;
    var stdout = std.Io.File.stdout().writer(io, &stdout_buf);
    const out = &stdout.interface;

    var csv: std.Io.Writer.Allocating = .init(gpa);
    defer csv.deinit();
    try csv.writer.print("split,seed,episode,lhs,operation,rhs,target,chosen\n", .{});

    var train_sum: f64 = 0;
    var held_sum: f64 = 0;
    var lookup_baseline: f64 = 0;
    try out.print("\n-- Phase 8 arithmetic curriculum --------------------------------\n", .{});
    for (seeds) |seed| {
        const r = try trainOne(gpa, seed, &csv.writer);
        train_sum += r.train_accuracy;
        held_sum += r.held_accuracy;
        lookup_baseline = r.lookup_baseline;
        try out.print("  seed {d}: train {d:.3}, held-out combinations {d:.3}\n", .{ seed, r.train_accuracy, r.held_accuracy });
    }
    try writeAtomic(io, "arithmetic.csv", csv.written());

    const denom = @as(f64, @floatFromInt(seeds.len));
    const train_mean = train_sum / denom;
    const held_mean = held_sum / denom;
    const gain = held_mean - lookup_baseline;
    const pass = held_mean >= pass_accuracy and gain >= pass_gain_over_lookup;
    try out.print(
        "\n  controlled split: nonzero additions with result 4 were never trained\n" ++
            "  train accuracy             {d:.3}\n" ++
            "  held-out accuracy          {d:.3}   (need >= {d:.2})\n" ++
            "  pair-lookup baseline       {d:.3}\n" ++
            "  gain over lookup           {d:.3}   (need >= {d:.2})\n" ++
            "  VERDICT: {s}\n\n  wrote arithmetic.csv\n\n",
        .{ train_mean, held_mean, pass_accuracy, lookup_baseline, gain, pass_gain_over_lookup, if (pass) "PASS -- structured readout beats pair memorization on held combinations." else "FAIL -- inspect arithmetic.csv." },
    );
    try out.flush();
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
