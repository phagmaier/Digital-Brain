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
const termination = @import("termination.zig");
const provenance = @import("provenance.zig");

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
/// A composed controller state is a full action-assembly decision, not one
/// stochastic neuron. Give that state a commensurate vote in the online answer
/// evidence so its availability can satisfy the stable-answer rule before the
/// fallback timeout window has elapsed.
const transition_termination_vote: u32 = 10;

/// Independently toggleable controller effects, so the held-out evaluation can
/// separate "the compositional controller injects an answer-specific current
/// into the spiking assemblies" from "the controller adds a direct vote to the
/// spike-count readout". Section 1 of report.md: the published PASS conflated
/// both with the learned readout, so the 1.000 score could not distinguish a
/// necessary spiking readout from a finite-state controller writing the answer.
const Controller = struct {
    /// Inject `transition_action_current` into the composed answer assembly and
    /// inhibit the competitors (the state-dependent action competition).
    inject_current: bool = false,
    /// Add `transition_termination_vote` group-sized votes straight into the
    /// spike-count readout each step (bypassing the spiking dynamics entirely).
    add_vote: bool = false,
};

const full_controller: Controller = .{ .inject_current = true, .add_vote = true };

const Curriculum = enum { increment_decrement, single_operation };

fn baseConfig(seed: u64) cfg.Config {
    return .{
        .master_seed = seed,
        .n_neurons = 160,
        .connection_density = 0.04,
        .arithmetic_enabled = true,
        .arithmetic_group_size = 5,
        .arithmetic_max_operand = 4,
        .termination_enabled = true,
        .termination_stable_steps = 4,
        .termination_timeout_steps = 40,
        .termination_timeout_reward = -0.2,
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

const Episode = struct {
    correct: bool,
    chosen: u8,
    termination: ?termination.Outcome = null,
    reward: f32 = 0.0,
    controller_used: bool = false,
};

/// Return the unique active leader of the existing spike-count answer readout.
/// An all-zero or tied readout is not termination evidence; accepting either
/// would create an answer-ID bias unrelated to learned readout activity.
fn uniqueWinner(counts: []const u32, answer_count: u32) ?u8 {
    std.debug.assert(answer_count > 0 and answer_count <= counts.len);
    var winner: u32 = 0;
    var best = counts[0];
    var tied = false;
    var answer: u32 = 1;
    while (answer < answer_count) : (answer += 1) {
        if (counts[answer] > best) {
            winner = answer;
            best = counts[answer];
            tied = false;
        } else if (counts[answer] == best) {
            tied = true;
        }
    }
    return if (best == 0 or tied) null else @intCast(winner);
}

test "phase 9: only a unique active action leader can support termination" {
    const unique = [_]u32{ 0, 3, 1 };
    try std.testing.expectEqual(@as(?u8, 1), uniqueWinner(&unique, 3));
    const tied = [_]u32{ 2, 2, 0 };
    try std.testing.expectEqual(@as(?u8, null), uniqueWinner(&tied, 3));
    const silent = [_]u32{ 0, 0, 0 };
    try std.testing.expectEqual(@as(?u8, null), uniqueWinner(&silent, 3));
}

fn runEpisode(
    s: *sim.Sim,
    l: arithmetic.Layout,
    ext: []f32,
    seed: u64,
    episode: u32,
    example: arithmetic.Example,
    transition_answer: ?u8,
    controller: Controller,
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
    if (controller.inject_current) if (transition_answer) |answer| {
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
    };
    var termination_tracker: ?termination.StableAnswer = if (c.termination_enabled)
        termination.StableAnswer.init(c.termination_stable_steps, c.termination_timeout_steps)
    else
        null;
    var terminal: ?termination.Outcome = null;
    const max_readout_steps: u32 = if (c.termination_enabled)
        c.termination_timeout_steps
    else
        c.arithmetic_readout_steps;
    for (0..max_readout_steps) |_| {
        _ = s.step(ext);
        const fired = s.network.neurons.fired;
        var answer: u8 = 0;
        while (answer < l.actionCount()) : (answer += 1) {
            const group = l.actionGroup(answer);
            for (group.lo..group.hi) |i| {
                const spike = @intFromBool(fired[i]);
                counts[answer] += spike;
            }
        }
        // The compositional controller's reached state is already part of the
        // answer interface. Add its one assembly vote at each readout step so
        // stable termination observes the same state as the final reader,
        // rather than delaying that evidence until after the timeout loop.
        if (controller.add_vote) if (transition_answer) |controller_answer| {
            counts[controller_answer] += l.actionGroup(controller_answer).count() * transition_termination_vote;
        };
        if (termination_tracker) |*tracker| {
            if (tracker.observe(uniqueWinner(&counts, l.actionCount()))) |outcome| {
                terminal = outcome;
                break;
            }
        }
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
    const terminal_reward: f32 = if (terminal) |outcome|
        termination.reward(outcome, correct, c.termination_timeout_reward)
    else if (correct)
        1.0
    else
        -1.0;
    if (learn) {
        // Phase 9 changes when the episode ends, not the local learning rule:
        // the sparse terminal scalar still modulates only eligible synapses.
        s.applyReward(terminal_reward);
        s.applyHomeostasis();
    }
    return .{
        .correct = correct,
        .chosen = chosen,
        .termination = terminal,
        .reward = terminal_reward,
        .controller_used = transition_answer != null and (controller.inject_current or controller.add_vote),
    };
}

fn heldOutExamples() [3]arithmetic.Example {
    return .{
        .{ .lhs = 1, .operation = .add, .rhs = 3 },
        .{ .lhs = 2, .operation = .add, .rhs = 2 },
        .{ .lhs = 3, .operation = .add, .rhs = 1 },
    };
}

/// One row of the controller-ablation matrix (report.md §1). Each condition
/// isolates a different combination of {trained spiking readout, controller
/// current injection, controller direct vote} so the held-out score can no
/// longer be attributed to the composed controller by default.
const EvalCondition = struct {
    name: []const u8,
    /// Run the spiking network. When false this is the pure finite-state
    /// controller baseline (`TransitionModel.solve`), no SNN at all.
    use_snn: bool,
    /// Evaluate on the reward-trained readout. When false a freshly initialized
    /// (untrained) network is used, isolating how much the controller alone
    /// drives the answer through an unlearned readout.
    trained_readout: bool,
    /// Supply `TransitionModel.solve(e)` as the composed answer to the SNN.
    use_controller_value: bool,
    controller: Controller,
};

/// The five reported conditions. `full` reproduces the original published
/// evaluation; the rest dismantle the confound one factor at a time.
const eval_conditions = [_]EvalCondition{
    // Original published evaluation: trained readout + current injection + vote.
    .{ .name = "full", .use_snn = true, .trained_readout = true, .use_controller_value = true, .controller = full_controller },
    // Controller current into the SNN, but read spike counts only (no vote).
    .{ .name = "current_no_vote", .use_snn = true, .trained_readout = true, .use_controller_value = true, .controller = .{ .inject_current = true, .add_vote = false } },
    // The learned spiking readout on its own, controller removed entirely.
    .{ .name = "learned_readout", .use_snn = true, .trained_readout = true, .use_controller_value = false, .controller = .{} },
    // Untrained/frozen readout with the full controller present.
    .{ .name = "frozen_controller", .use_snn = true, .trained_readout = false, .use_controller_value = true, .controller = full_controller },
    // Pure finite-state controller, no spiking network at all.
    .{ .name = "controller_only", .use_snn = false, .trained_readout = false, .use_controller_value = true, .controller = full_controller },
};

/// Index of the condition whose gain over the pair-lookup baseline is the
/// honest scientific claim: the learned spiking readout with no controller aid.
const honest_condition_index: usize = 2;

const AblationResult = struct {
    held_accuracy: f64,
    stable_termination_rate: f64,
};

const Result = struct {
    train_accuracy: f64,
    lookup_baseline: f64,
    conditions: [eval_conditions.len]AblationResult,
};

/// Evaluate the held-out combinations on one network under one condition. The
/// per-condition `base_episode` offset keeps every condition's derived RNG keys
/// (tie-breaking, DEC-004) disjoint and reproducible.
fn evalHeldOnSim(
    s: *sim.Sim,
    l: arithmetic.Layout,
    ext: []f32,
    seed: u64,
    transitions: arithmetic.TransitionModel,
    cond: EvalCondition,
    base_episode: u32,
    csv: *std.Io.Writer,
) !AblationResult {
    var held_correct: u32 = 0;
    var held_stable: u32 = 0;
    const held = heldOutExamples();
    var eval_index: u32 = 0;
    while (eval_index < eval_repeats * held.len) : (eval_index += 1) {
        const e = held[eval_index % held.len];
        const episode = base_episode + eval_index;
        const value: ?u8 = if (cond.use_controller_value) transitions.solve(e) else null;
        const r = runEpisode(s, l, ext, seed, episode, e, value, cond.controller, false);
        held_correct += @intFromBool(r.correct);
        held_stable += @intFromBool(r.termination == .stable_answer);
        try csv.print("held_out_combination,{s},{d},{d},{d},{s},{d},{d},{d},{s},{d:.1},{s}\n", .{
            cond.name,
            seed,
            episode,
            e.lhs,
            "add",
            e.rhs,
            e.result(),
            r.chosen,
            if (r.termination) |outcome| @tagName(outcome) else "fixed_window",
            r.reward,
            if (r.controller_used) "yes" else "no",
        });
    }
    const held_total = eval_repeats * held.len;
    return .{
        .held_accuracy = @as(f64, @floatFromInt(held_correct)) / @as(f64, @floatFromInt(held_total)),
        .stable_termination_rate = @as(f64, @floatFromInt(held_stable)) / @as(f64, @floatFromInt(held_total)),
    };
}

/// The pure finite-state controller baseline (report.md §1): no spiking network,
/// just the composed `solve`. Deterministic, so its "termination" is undefined.
fn controllerOnlyAccuracy(
    l: arithmetic.Layout,
    transitions: arithmetic.TransitionModel,
    seed: u64,
    cond: EvalCondition,
    base_episode: u32,
    csv: *std.Io.Writer,
) !AblationResult {
    var correct: u32 = 0;
    const held = heldOutExamples();
    var eval_index: u32 = 0;
    while (eval_index < eval_repeats * held.len) : (eval_index += 1) {
        const e = held[eval_index % held.len];
        const chosen: u8 = transitions.solve(e) orelse blk: {
            // No learned transition path: fall back to the same fixed prior the
            // pair-lookup baseline gets (deterministic answer 0).
            break :blk 0;
        };
        correct += @intFromBool(chosen == e.result());
        try csv.print("held_out_combination,{s},{d},{d},{d},{s},{d},{d},{d},{s},{d:.1},{s}\n", .{
            cond.name,
            seed,
            base_episode + eval_index,
            e.lhs,
            "add",
            e.rhs,
            e.result(),
            chosen,
            "none",
            @as(f32, if (chosen == e.result()) 1.0 else -1.0),
            "yes",
        });
    }
    _ = l;
    const held_total = eval_repeats * held.len;
    return .{
        .held_accuracy = @as(f64, @floatFromInt(correct)) / @as(f64, @floatFromInt(held_total)),
        .stable_termination_rate = 0.0,
    };
}

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
        const r = runEpisode(&s, l, ext, seed, episode, e, hint, full_controller, true);
        // This receives the same rewarded unit-transition feedback as the
        // action assembly. It cannot observe a multi-operand answer.
        if (stage == .increment_decrement and r.correct) _ = transitions.observeTransition(e);
        if (episode >= total_train - final_window) final_train_correct += @intFromBool(r.correct);
    }
    const train_accuracy = @as(f64, @floatFromInt(final_train_correct)) / @as(f64, @floatFromInt(final_window));

    const eval_span: u32 = eval_repeats * @as(u32, heldOutExamples().len);
    var conditions: [eval_conditions.len]AblationResult = undefined;
    for (eval_conditions, 0..) |cond, ci| {
        const base_episode = total_train + @as(u32, @intCast(ci)) * eval_span;
        if (!cond.use_snn) {
            conditions[ci] = try controllerOnlyAccuracy(l, transitions, seed, cond, base_episode, csv);
        } else if (cond.trained_readout) {
            conditions[ci] = try evalHeldOnSim(&s, l, ext, seed, transitions, cond, base_episode, csv);
        } else {
            // Frozen readout: a fresh, never-rewarded network with the same
            // graph seed, reusing the already-learned transition controller.
            var fresh = try sim.Sim.init(gpa, c);
            defer fresh.deinit(gpa);
            conditions[ci] = try evalHeldOnSim(&fresh, l, ext, seed, transitions, cond, base_episode, csv);
        }
    }

    return .{
        .train_accuracy = train_accuracy,
        .lookup_baseline = 1.0 / @as(f64, @floatFromInt(l.actionCount())),
        .conditions = conditions,
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
    try csv.writer.print("split,condition,seed,episode,lhs,operation,rhs,target,chosen,termination,reward,controller\n", .{});

    var train_sum: f64 = 0;
    var lookup_baseline: f64 = 0;
    var held_sum: [eval_conditions.len]f64 = [_]f64{0} ** eval_conditions.len;
    var stable_sum: [eval_conditions.len]f64 = [_]f64{0} ** eval_conditions.len;
    try out.print("\n-- Phase 8/9 arithmetic curriculum: controller-ablation matrix ---\n", .{});
    for (seeds) |seed| {
        const r = try trainOne(gpa, seed, &csv.writer);
        train_sum += r.train_accuracy;
        lookup_baseline = r.lookup_baseline;
        for (r.conditions, 0..) |cr, ci| {
            held_sum[ci] += cr.held_accuracy;
            stable_sum[ci] += cr.stable_termination_rate;
        }
        try out.print("  seed {d}: train {d:.3}\n", .{ seed, r.train_accuracy });
        for (eval_conditions, 0..) |cond, ci| {
            try out.print(
                "    {s:<18} held {d:.3}  stable-term {d:.3}\n",
                .{ cond.name, r.conditions[ci].held_accuracy, r.conditions[ci].stable_termination_rate },
            );
        }
    }
    try writeAtomic(io, "arithmetic.csv", csv.written());
    try provenance.write(io, gpa, "arithmetic.meta.json", "arithmetic_curriculum", .{
        .seeds = seeds,
        .config_template = baseConfig(seeds[0]),
        .increment_episodes = increment_episodes,
        .arithmetic_episodes = arithmetic_episodes,
        .eval_repeats = eval_repeats,
        .pass_accuracy = pass_accuracy,
        .pass_gain_over_lookup = pass_gain_over_lookup,
        .curriculum_teacher_current = curriculum_teacher_current,
        .transition_action_current = transition_action_current,
        .transition_competition_current = transition_competition_current,
        .transition_termination_vote = transition_termination_vote,
        .eval_conditions = eval_conditions,
        .honest_condition = eval_conditions[honest_condition_index].name,
    });

    const denom = @as(f64, @floatFromInt(seeds.len));
    const train_mean = train_sum / denom;
    try out.print(
        "\n  controlled split: nonzero additions with result 4 were never trained\n" ++
            "  train accuracy             {d:.3}\n" ++
            "  pair-lookup baseline       {d:.3}\n\n" ++
            "  condition            held-acc   gain-vs-lookup   stable-term\n" ++
            "  --------------------------------------------------------------\n",
        .{ train_mean, lookup_baseline },
    );
    for (eval_conditions, 0..) |cond, ci| {
        const held_mean = held_sum[ci] / denom;
        const stable_mean = stable_sum[ci] / denom;
        try out.print(
            "  {s:<18}   {d:.3}      {d:.3}          {d:.3}\n",
            .{ cond.name, held_mean, held_mean - lookup_baseline, stable_mean },
        );
    }

    // The honest scientific claim is the learned spiking readout with NO
    // controller aid (report.md §1). The verdict is judged on that condition,
    // not on the controller-assisted `full` number.
    const honest_held = held_sum[honest_condition_index] / denom;
    const honest_gain = honest_held - lookup_baseline;
    const pass = honest_held >= pass_accuracy and honest_gain >= pass_gain_over_lookup;
    try out.print(
        "\n  honest claim = condition '{s}' (learned readout, controller removed)\n" ++
            "    held-out accuracy {d:.3}  (need >= {d:.2}), gain {d:.3}  (need >= {d:.2})\n" ++
            "  VERDICT: {s}\n\n  wrote arithmetic.csv\n\n",
        .{
            eval_conditions[honest_condition_index].name,
            honest_held,
            pass_accuracy,
            honest_gain,
            pass_gain_over_lookup,
            if (pass)
                "PASS -- the learned spiking readout alone beats pair memorization."
            else
                "FAIL -- held-out generalization is controller-assisted, not learned readout (see matrix).",
        },
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
