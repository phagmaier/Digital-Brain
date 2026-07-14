//! Configuration. Everything that affects a run lives here, and the whole
//! struct is written into run metadata alongside the seed, the PRNG algorithm,
//! and the Zig version. A run you cannot reconstruct is a run you did not do.

const std = @import("std");
const builtin = @import("builtin");
const rng = @import("rng.zig");

pub const NeuronKind = enum(u8) {
    excitatory,
    inhibitory,

    /// d_i in the spec: the SIGN of this neuron's outgoing effect.
    /// Sign lives in the presynaptic neuron's type. Never in the weight.
    pub fn sign(self: NeuronKind) f32 {
        return switch (self) {
            .excitatory => 1.0,
            .inhibitory => -1.0,
        };
    }
};

pub const ResetRule = enum {
    /// u <- u - reset_decrement   (default, DEC-003)
    subtractive,
    /// u <- 0
    hard,
};

fn isProbability(value: f32) bool {
    return value >= 0.0 and value <= 1.0;
}

fn isDecay(value: f32) bool {
    return value >= 0.0 and value < 1.0;
}

fn isNonNegative(value: f32) bool {
    return value >= 0.0;
}

pub const Config = struct {
    // -- reproducibility --------------------------------------------------
    master_seed: u64 = 0xC0FFEE,

    // -- population -------------------------------------------------------
    n_neurons: u32 = 100,
    /// Fraction excitatory. 80/20 is an engineering starting point, not a law.
    excitatory_fraction: f32 = 0.8,

    // -- graph ------------------------------------------------------------
    /// Target connection density (fraction of all ordered pairs).
    connection_density: f32 = 0.08,
    /// Distance bias for initial connectivity:
    ///   P(i -> j) ~ exp(-||x_i - x_j||^2 / (2 sigma^2))
    /// Neuron positions are sampled in the unit square, so sigma is in those
    /// units. Large sigma -> effectively uniform random graph.
    spatial_sigma: f32 = 0.25,
    /// Set true to disable the spatial bias entirely (a useful Phase 1 ablation:
    /// does local structure matter for the dynamics, before it matters for growth?)
    uniform_graph: bool = false,
    no_self_connections: bool = true,

    // -- synapses ---------------------------------------------------------
    /// INVARIANT: min_delay >= 1. See DEC-001.
    /// A spike emitted at t must not affect its target before t+1. Zero-delay
    /// synapses would make behaviour depend on neuron iteration order, or would
    /// land in an already-processed ring bucket and arrive a full wrap-around
    /// later. One timestep is the smallest causal interval this model can
    /// represent.
    min_delay: u16 = 1,
    max_delay: u16 = 5,

    /// Initial weights, drawn uniformly. w >= 0 ALWAYS.
    w_exc_init_lo: f32 = 0.4,
    w_exc_init_hi: f32 = 1.0,
    w_inh_init_lo: f32 = 1.2,
    w_inh_init_hi: f32 = 2.4,

    /// Stochastic synaptic release. Independent of stochastic firing, so the
    /// two randomness sources can be ablated separately.
    release_probability: f32 = 0.5,

    // -- membrane (REST-RELATIVE, DEC-002) --------------------------------
    // u_i = V_i - V_rest, so rest is exactly u = 0 and the equation leaks
    // toward zero with no ambiguity about which fixed point it has.
    //   u(t+1) = lambda_u * u(t) + I(t) - a(t) - s(t) * reset_decrement
    membrane_leak: f32 = 0.9, // lambda_u in [0,1]
    threshold: f32 = 1.0, // theta, rest-relative
    reset_rule: ResetRule = .subtractive,
    /// DECOUPLED FROM THRESHOLD ON PURPOSE (DEC-003). With sigmoid firing there
    /// is no clean threshold crossing, so subtracting theta specifically is
    /// arbitrary and can drive u far below rest.
    reset_decrement: f32 = 1.0,

    // -- firing -----------------------------------------------------------
    //   P(fire) = sigmoid(beta * (u - theta))
    /// beta controls the SHARPNESS of the firing nonlinearity, i.e. how
    /// deterministic firing is near threshold. It does NOT control the
    /// operating point. Network quietness is set by threshold, input scale, and
    /// E/I balance. "Low beta for a quiet core" is a category error.
    beta: f32 = 4.0,
    refractory_steps: u16 = 2,

    // -- adaptation -------------------------------------------------------
    //   a(t+1) = lambda_a * a(t) + alpha_a * s(t),  subtracted from u
    adaptation_decay: f32 = 0.95, // lambda_a
    adaptation_increment: f32 = 0.12, // alpha_a
    adaptation_enabled: bool = true,
    /// Reset adaptation at episode boundaries? See DEC-005 -- this one is
    /// deliberately configurable. Resetting creates an artificial discontinuity
    /// if adaptation represents genuine spike-frequency adaptation; not
    /// resetting lets one episode influence the next.
    reset_adaptation_between_episodes: bool = true,

    // -- background drive -------------------------------------------------
    /// Constant current injected into every neuron. This is the knob that sets
    /// the operating point. It is NOT beta.
    background_current: f32 = 0.35,

    // -- homeostasis (Phase 2) --------------------------------------------
    // Two independent homeostats, each individually gated. Both are OFF by
    // default so the Phase 1 baseline run is unchanged; turn them on via config
    // (see configs/homeostasis.json) or the perturbation harness.
    //
    // 1. Adaptive thresholds (the primary homeostat). Each neuron nudges its own
    //    threshold to drive its recent rate toward target_rate:
    //      theta_i += eta_h * (rho_i - rho_target)
    //    rho_i is the per-neuron firing-rate EMA. Negative feedback: fire too
    //    much -> threshold rises -> fire less. This is the "recent firing-rate
    //    estimate" + "adaptive thresholds" build items.
    homeostasis_enabled: bool = false,
    rate_ema_decay: f32 = 0.99, // lambda_rho; time constant ~ 1/(1-lambda) steps
    target_rate: f32 = 0.02, // rho_target, spikes/neuron/step
    homeostasis_lr: f32 = 0.01, // eta_h
    threshold_min: f32 = 0.1,
    /// Ceiling on the adaptive threshold. This is a stability rail, not a tuning
    /// knob -- a threshold pinned here means the operating point is off (drive too
    /// high for the target). 12 gives the homeostat enough authority to reject a
    /// roughly doubled drive; keep it finite so runaway is still caught.
    threshold_max: f32 = 12.0,
    /// Cadence of the homeostatic update. TRUE (the continuous simulator)
    /// applies both homeostats every step inside Sim.step. FALSE hands cadence
    /// to the caller: the Phase 3 episode driver sets this false and calls
    /// Sim.applyHomeostasis() once per episode -- the doc's "Update homeostasis"
    /// step, per episode/window rather than per timestep.
    homeostasis_per_step: bool = true,

    // 2. Synaptic scaling (DEC-007, the "optional simple weight normalization").
    //    Postsynaptic, excitatory-input only, multiplicative and slow. Each
    //    EXCITATORY synapse is scaled by its TARGET neuron's rate error:
    //      w_ij *= 1 + eta_w * (rho_target - rho_j)
    //    A neuron firing above target scales its excitatory inputs down. This is
    //    Turrigiano-style synaptic scaling, not Hebbian learning: it is global
    //    and activity-driven, touches no eligibility trace, and leaves the
    //    plastic[] flag alone (that belongs to Phase 3). Inhibitory synapses are
    //    left untouched -- scaling them by the same rule would be the wrong sign.
    weight_normalization_enabled: bool = false,
    weight_norm_lr: f32 = 0.002, // eta_w
    /// Upper clamp for scaled weights. w >= 0 still holds (sign is the neuron's).
    weight_max: f32 = 8.0,

    // -- plasticity (Phase 3, DEC-009: three-factor reward-modulated Hebb) ---
    // The learning rule is three-factor: pre x post (Hebbian coincidence,
    // captured in an eligibility trace) x reward (a global scalar delivered at
    // the end of an episode). The eligibility trace is what bridges the gap
    // between the coincidence and the later reward -- it "tags" recently active
    // synapses so a reward can find them (Izhikevich 2007).
    //   pre_i (decays, bumped on i's spike), post_j (decays, bumped on j's spike)
    //   e_ij(t) = lambda_e * e_ij + [post spike now] * pre_i   (LTP; LTD optional)
    //   on reward R:  w_ij += eta_w * (R - baseline) * e_ij
    // Only synapses with plastic[k] = true are touched; in the task those are the
    // input->action synapses (see net.zig). Deterministic, no RNG.
    plasticity_enabled: bool = false,
    pre_trace_decay: f32 = 0.9, // lambda_pre
    post_trace_decay: f32 = 0.9, // lambda_post
    trace_increment: f32 = 1.0, // bump added to a trace on a spike
    eligibility_decay: f32 = 0.9, // lambda_e
    /// Enable the anti-Hebbian LTD term (post-before-pre depresses). Off by
    /// default: LTP-only reward modulation is more robust for the association task.
    ltd_enabled: bool = false,
    learning_rate: f32 = 0.05, // eta_w
    /// Reward baseline EMA decay. Subtracting a running reward baseline makes the
    /// update zero-mean as accuracy rises, which stops the plastic weights from
    /// drifting/saturating once the task is solved. This is the REINFORCE-with-
    /// baseline trick; it materially improves cross-seed reliability.
    reward_baseline_decay: f32 = 0.98,
    weight_max_plastic: f32 = 4.0, // clamp for plastic weights (w >= 0 still holds)

    // -- task: immediate two-choice association (Phase 3, DEC-008) ----------
    // Four disjoint groups are carved from the low excitatory IDs: input_a,
    // input_b, action_0, action_1 (task_group_size neurons each; see task.zig).
    // Stimulus A drives input_a, B drives input_b. The correct mapping is
    // A->action_0, B->action_1. When task_enabled, net.zig adds all-to-all
    // PLASTIC input->action synapses (a trainable readout on top of the fixed
    // random reservoir); reward learning sculpts input_a->action_0 up and
    // input_a->action_1 down (and symmetrically for B).
    task_enabled: bool = false,
    task_group_size: u32 = 8,
    task_input_current: f32 = 1.5, // stimulus drive into the active input group
    task_ia_weight_init: f32 = 0.2, // initial (symmetric) input->action weight
    task_ia_p_release: f32 = 1.0, // reliable readout pathway
    /// Phase 4 working memory. Fixed all-to-all self-excitation WITHIN each input
    /// group. A stimulus kicks its input assembly on; this recurrent excitation
    /// (bounded by the threshold homeostat) sustains the assembly's activity
    /// after the stimulus is removed, so the input->action readout still reflects
    /// the stimulus after a delay. 0 disables it (the Phase 3 immediate regime).
    task_recurrent_weight: f32 = 0.0,

    // -- Stage 2: context-dependent delayed mapping + reservoir plasticity --
    // Delayed XOR-style task (report.md Stage 2 / final.md §7):
    //   context X + cue A → 0,  X + B → 1,  Y + A → 1,  Y + B → 0
    // Context and cue are presented at separate times. Plastic stimulus→action
    // readout edges go from both context and cue assemblies (still not a linear
    // shortcut: the XOR mapping is not linearly separable in those four rates);
    // context assemblies also carry fixed self-excitation (`context_hold_weight`)
    // so the context can bridge the inter-stimulus delay. The flagship comparison
    // is fixed-reservoir + plastic readout vs locally plastic reservoir edges
    // (DEC-014). OFF by default so the Phase 1 baseline is unchanged.
    context_task_enabled: bool = false,
    context_task_group_size: u32 = 6,
    /// Fixed all-to-all self-excitation WITHIN each context group (not the cue
    /// groups). 0 disables the context hold.
    context_hold_weight: f32 = 0.0,
    /// When true (and plasticity_enabled), reservoir and grown edges are plastic:
    /// they accumulate eligibility and take the three-factor reward update. Task
    /// readout edges remain plastic independently. OFF by default (DEC-008's
    /// fixed-reservoir regime).
    reservoir_plasticity_enabled: bool = false,

    // -- structural plasticity (Phase 5, DEC-011) --------------------------
    // Local random connection search: the reservoir graph GROWS and PRUNES
    // connections over training instead of staying fixed. OFF by default so the
    // Phase 1 baseline run is byte-identical; enable via configs/structural.json
    // or the growth harness (zig build grow).
    //
    // Only RESERVOIR (and grown) edges are structural. The task readout and the
    // working-memory recurrent edges are never grown, decayed, or pruned -- they
    // carry Phase 3/4 function and must survive rewiring untouched.
    //
    // Mechanism (spec §8): every synapse carries a PERMANENCE q in [0,1] (its
    // resistance to pruning, distinct from the functional weight). Used synapses
    // consolidate (q -> 1); disused ones decay (q -> 0), their weight decays
    // faster the lower q is (§8.5), and once weak + low-permanence + old enough
    // they are pruned (§8.6), freeing a slot. Growth then samples a nearby target
    // (local random search, §8.1) and installs a weak tentative synapse in a free
    // slot. The per-neuron slot capacity IS the outgoing connection budget (§8.7).
    structural_plasticity_enabled: bool = false,

    /// Per-neuron outgoing slot capacity == the HARD connection budget (§8.7's
    /// "maximum outgoing synapses"). Each source neuron gets this many CSR slots;
    /// its initial live edges fill the front, the rest are free slots for growth.
    /// A neuron already above budget from the initial graph simply cannot grow
    /// until pruning frees a slot -- capacity is max(initial_out_degree, this).
    max_out_degree: u32 = 20,

    /// Target STRUCTURAL out-degree (§8.7's "target outgoing degree"). Growth stops
    /// adding to a source once it has this many live structural (reservoir/grown)
    /// out-edges, so the live population plateaus instead of accreting to the hard
    /// cap; pruning then frees room and growth refills it, which is what turns
    /// steady accretion into genuine TURNOVER around a set point. 0 disables the
    /// set-point (grow until the hard cap). Must be <= max_out_degree.
    target_out_degree: u32 = 10,

    // Permanence update (§8.4), simplified for reservoir edges (which carry no
    // eligibility trace): co-activity of the two endpoints STABILIZES a synapse,
    // disuse DECAYS it. Co-activity is read from the endpoints' rate EMAs,
    // normalized to target_rate (both firing at target -> coactivity 1).
    //   q += eta_a * coactivity  -  lambda_q  +  eta_q * max(0, baseline * e)
    permanence_activity_lr: f32 = 0.05, // eta_a: gain from endpoint co-activity
    permanence_disuse_decay: f32 = 0.05, // lambda_q: leak per structural event
    /// Ceiling on the (target-normalized) co-activity term, so an over-active
    /// endpoint pair cannot pin permanence at 1 for every synapse. With eta_a ==
    /// lambda_q, the break-even co-activity is 1 (both endpoints at target); more
    /// co-active than that consolidates, less decays toward prunable. The low
    /// clamp keeps even consolidated synapses off the hard ceiling so weight decay
    /// stays mildly active -- which is what keeps weak/grown synapses cycling out
    /// (real turnover) rather than every synapse pinning at permanence 1.
    coactivity_max: f32 = 1.5,

    /// Permanence-dependent weight decay (§8.5): w *= 1 - lambda_w*(1-q). High
    /// permanence -> ~no decay; low permanence -> fast decay toward prunable.
    weight_permanence_decay: f32 = 0.12, // lambda_w

    // Pruning (§8.6): remove a structural synapse when it is simultaneously
    // low-permanence, weak, AND past a minimum age (the grace period that stops a
    // tentative synapse being deleted before it can participate).
    prune_permanence_min: f32 = 0.12, // q_min
    prune_weight_min: f32 = 0.08, // w_min
    min_synapse_age: u32 = 2, // in structural events

    // Growth (§8.1, pure local random search -- heuristic #1, the baseline). Each
    // structural event, a source neuron with a free slot attempts (with prob
    // growth_probability) to sample a point near itself and connect to the nearest
    // legal neuron. New synapses are WEAK and TENTATIVE (low permanence) -- strong
    // new weights and no grace period are listed failure modes (§21).
    growth_probability: f32 = 0.10, // per source neuron per structural event
    /// Spatial spread of the local target sample. 0 => reuse spatial_sigma.
    growth_sigma: f32 = 0.0,
    grow_weight_init: f32 = 0.1, // weak (well below w_exc_init_lo)
    grow_permanence_init: f32 = 0.35, // tentative: above q_min, below established

    /// Cadence of structural events in the CONTINUOUS simulator, analogous to
    /// homeostasis_per_step. 0 (the default) hands cadence to the caller: the
    /// episode harness sets it 0 and calls Sim.applyStructuralPlasticity() once
    /// per growth window. > 0 makes Sim.step() run a structural event every this-
    /// many steps, so a free-running `zig build run -- configs/structural.json`
    /// visibly rewires. Keep it large -- growth is the slowest clock (§8.7).
    structural_interval_steps: u32 = 0,

    // -- consolidation (Phase 6, DEC-012) ---------------------------------
    // Separates the FAST weight timescale from the SLOW structure timescale so
    // that repeatedly-rewarded pathways SURVIVE while unused tentative ones decay.
    // OFF by default; requires structural_plasticity_enabled (it reuses that slow
    // clock for permanence decay + pruning). Enable via configs/consolidation.json
    // or the continual-learning harness (zig build continual).
    //
    // The bridge is reward-gated permanence on the PLASTIC readout synapses (the
    // ones that carry eligibility, unlike the reservoir): when a plastic synapse
    // is part of rewarded behaviour, applyReward nudges its permanence up (§8.4's
    // eta_q*max(0, r*e) term). Consolidated (high-q) pathways then barely feel the
    // permanence-dependent weight decay / pruning that erodes the rest. So:
    //   reservoir edges  -- permanence driven by ACTIVITY (Phase 5, DEC-011).
    //   plastic  edges   -- permanence driven by REWARD (this, DEC-012); §8.3's
    //                       "consolidated == repeatedly associated with rewarded
    //                       behaviour", NOT merely co-active.
    // With this on, plastic synapses also join the slow disuse-decay + prune loops,
    // so an unused, unconsolidated readout pathway forgets (decays and is pruned).
    consolidation_enabled: bool = false,
    /// eta_q: reward-gated permanence gain per rewarded episode on plastic
    /// synapses. Small on purpose -- permanence is the SLOW variable; it should
    /// take many rewarded episodes to consolidate, so noise doesn't stick.
    consolidation_lr: f32 = 0.02,
    /// Consolidation reward source (report.md §5 comparison). Default (false)
    /// uses RAW reward `r` as DEC-012 mandates: once the task is mastered the
    /// reward baseline -> +1 and a centered term would stop consolidating the
    /// reliably-correct pathways. Set true to consolidate on the baseline-
    /// subtracted modulator `(r - baseline)` instead -- the failure mode DEC-012
    /// warns about, exposed here so the continual harness can measure it directly.
    consolidation_use_centered_reward: bool = false,

    // -- arithmetic curriculum (Phase 8, DEC-013) ------------------------
    // A separate, bounded symbolic layout for the arithmetic harness.  Each
    // sequence position owns its own numeral assembly, which preserves operand
    // order; action assemblies cover non-negative answers 0..2*max_operand.
    // All task edges are deterministic all-to-all symbol->action readouts and
    // draw no RNG, preserving the reservoir stream (DEC-004).
    arithmetic_enabled: bool = false,
    arithmetic_group_size: u32 = 6,
    arithmetic_max_operand: u8 = 4,
    arithmetic_input_current: f32 = 1.5,
    arithmetic_readout_weight_init: f32 = 0.2,
    arithmetic_readout_p_release: f32 = 1.0,
    /// A neutral, fixed-duration answer clock applied equally to every action
    /// assembly. It removes learned termination without leaking the answer.
    arithmetic_answer_probe_current: f32 = 0.5,
    arithmetic_symbol_steps: u32 = 12,
    arithmetic_gap_steps: u32 = 3,
    arithmetic_settle_steps: u32 = 15,
    arithmetic_readout_steps: u32 = 20,

    // -- learned termination (Phase 9) -----------------------------------
    // The first answer interface deliberately uses a fixed window (Phase 8).
    // Once answer production is reliable, an opt-in stable-answer controller
    // ends the episode as soon as one *unique* answer stays dominant for this
    // many consecutive readout steps. This is answer-ID-neutral: it cannot
    // encode, or favour, a particular answer. A finite timeout remains a hard
    // safety rail and receives its own weaker terminal reward.
    termination_enabled: bool = false,
    termination_stable_steps: u32 = 4,
    termination_timeout_steps: u32 = 40,
    termination_timeout_reward: f32 = -0.2,

    // -- workspace-inspired broadcast (Phase 7) --------------------------
    // A deliberately small, task-scoped global-workspace metaphor. The two
    // input assemblies are the only candidate writers; their recent activity
    // competes for `workspace_capacity` slots. An assembly must cross the
    // ignition threshold before it is admitted. The admitted identity decays
    // unless its candidate activity refreshes it, then feeds current back to
    // that candidate and a weak common broadcast to the wider excitatory pool.
    // This is a bottlenecked broadcast, not unrestricted shared memory.
    // OFF by default so Phases 1--6 remain byte-identical.
    workspace_enabled: bool = false,
    /// Number of simultaneously broadcast candidate assemblies. There are
    /// currently two task input candidates, so this is in [1, 2].
    workspace_capacity: u32 = 1,
    /// Leaky evidence trace for each candidate assembly's mean spike activity.
    workspace_candidate_decay: f32 = 0.85,
    /// Minimum candidate evidence required for competitive admission.
    workspace_ignition_threshold: f32 = 0.75,
    /// Persistence of an admitted workspace item per simulation step.
    workspace_state_decay: f32 = 0.90,
    /// Identity-preserving feedback applied to neurons in the winning candidate
    /// assembly. This is what lets a selected representation remain available.
    workspace_feedback_current: f32 = 0.45,
    /// Small, identity-neutral current sent to every non-input excitatory neuron
    /// while any workspace item is active. It is the broad broadcast component;
    /// it cannot by itself encode the answer or bypass the plastic readout.
    workspace_broadcast_current: f32 = 0.03,

    // -- run --------------------------------------------------------------
    steps: u32 = 2000,

    /// Validate the invariants that must hold before a single step is taken.
    pub fn validate(self: Config) !void {
        // NaN silently makes ordinary comparisons false, while infinities can
        // poison dynamics and serialized provenance. Reject every float through
        // one centralized gate before applying field-specific domains below.
        inline for (std.meta.fields(Config)) |field| {
            if (comptime field.type == f32) {
                if (!std.math.isFinite(@field(self, field.name)))
                    return error.NonFiniteParameter;
            }
        }

        // DEC-001. This is the one that will silently corrupt a run if violated.
        if (self.min_delay < 1) return error.ZeroDelayForbidden;
        if (self.max_delay < self.min_delay) return error.BadDelayRange;

        if (self.n_neurons == 0) return error.NoNeurons;
        if (!isProbability(self.excitatory_fraction))
            return error.BadExcitatoryFraction;
        if (!isProbability(self.membrane_leak))
            return error.BadLeak;
        if (!isProbability(self.release_probability))
            return error.BadReleaseProbability;
        if (self.connection_density <= 0.0 or self.connection_density > 1.0)
            return error.BadDensity;
        if (!self.uniform_graph and self.spatial_sigma <= 0.0)
            return error.BadSpatialSigma;
        if (!isNonNegative(self.reset_decrement) or self.beta < 0.0)
            return error.BadMembraneParameter;
        if (!isDecay(self.adaptation_decay) or !isNonNegative(self.adaptation_increment))
            return error.BadAdaptationParameter;

        // w >= 0 always. Sign lives in the neuron type.
        if (!isNonNegative(self.w_exc_init_lo) or !isNonNegative(self.w_inh_init_lo))
            return error.NegativeWeight;
        if (self.w_exc_init_hi < self.w_exc_init_lo or self.w_inh_init_hi < self.w_inh_init_lo)
            return error.BadWeightRange;

        // Phase 2 homeostasis knobs.
        if (!isDecay(self.rate_ema_decay))
            return error.BadRateEmaDecay;
        if (!isProbability(self.target_rate))
            return error.BadTargetRate;
        if (!isNonNegative(self.homeostasis_lr) or !isNonNegative(self.weight_norm_lr))
            return error.BadHomeostasisLearningRate;
        if (!isNonNegative(self.threshold_min) or self.threshold_max < self.threshold_min)
            return error.BadThresholdRange;
        if (self.homeostasis_enabled and (self.threshold < self.threshold_min or self.threshold > self.threshold_max))
            return error.InitialThresholdOutsideHomeostaticRange;
        // DEC-007 scales excitatory inputs only, so this clamp need not cover
        // intentionally strong inhibitory weights used by E/I ablations.
        if (self.weight_max < self.w_exc_init_hi)
            return error.WeightMaxBelowInit;

        // Phase 3 plasticity/task knobs.
        if (!isDecay(self.pre_trace_decay) or !isDecay(self.post_trace_decay) or
            !isDecay(self.eligibility_decay) or !isDecay(self.reward_baseline_decay))
            return error.BadEligibilityDecay;
        if (!isNonNegative(self.trace_increment) or !isNonNegative(self.learning_rate))
            return error.BadPlasticityLearningRate;
        if (!isNonNegative(self.weight_max_plastic)) return error.NegativeWeight;
        if (!isProbability(self.task_ia_p_release))
            return error.BadTaskReleaseProbability;
        if (!isNonNegative(self.task_input_current) or !isNonNegative(self.task_ia_weight_init) or
            !isNonNegative(self.task_recurrent_weight))
            return error.NegativeTaskCurrentOrWeight;
        if (self.task_enabled and self.weight_max_plastic < self.task_ia_weight_init)
            return error.PlasticWeightMaxBelowInit;
        if ((self.task_enabled and self.arithmetic_enabled) or
            (self.task_enabled and self.context_task_enabled) or
            (self.arithmetic_enabled and self.context_task_enabled))
            return error.OverlappingTaskLayouts;
        if (self.task_enabled) {
            // The four task groups are carved from the excitatory population and
            // must fit inside it.
            const n_exc: u32 = @intFromFloat(@round(@as(f32, @floatFromInt(self.n_neurons)) * self.excitatory_fraction));
            if (self.task_group_size == 0) return error.EmptyTaskGroup;
            if (4 * self.task_group_size > n_exc) return error.TaskGroupsExceedExcitatory;
        }

        // Stage 2 context-dependent task: six groups from low excitatory IDs.
        if (self.context_task_enabled) {
            if (self.context_task_group_size == 0) return error.EmptyContextTaskGroup;
            if (6 * self.context_task_group_size > self.nExcitatory())
                return error.ContextTaskGroupsExceedExcitatory;
            if (!isNonNegative(self.context_hold_weight))
                return error.NegativeContextHoldWeight;
            if (self.weight_max_plastic < self.task_ia_weight_init)
                return error.PlasticWeightMaxBelowInit;
        }
        if (self.reservoir_plasticity_enabled and !self.plasticity_enabled)
            return error.ReservoirPlasticityNeedsPlasticity;

        // Phase 8 arithmetic layout. The position-bound symbol assemblies plus
        // answer assemblies are all carved from the low excitatory IDs; require
        // room before net.zig constructs a single edge. Limiting the tiny
        // curriculum's operand range also keeps the u8 answer interface exact.
        if (self.arithmetic_enabled) {
            if (self.arithmetic_group_size == 0) return error.EmptyArithmeticGroup;
            if (self.arithmetic_max_operand == 0 or self.arithmetic_max_operand > 32)
                return error.BadArithmeticOperandRange;
            if (!isProbability(self.arithmetic_readout_p_release))
                return error.BadArithmeticReleaseProbability;
            if (self.arithmetic_symbol_steps == 0 or self.arithmetic_readout_steps == 0)
                return error.ZeroArithmeticWindow;
            if (!isNonNegative(self.arithmetic_input_current) or !isNonNegative(self.arithmetic_answer_probe_current) or
                !isNonNegative(self.arithmetic_readout_weight_init))
                return error.NegativeArithmeticCurrentOrWeight;
            if (self.weight_max_plastic < self.arithmetic_readout_weight_init)
                return error.PlasticWeightMaxBelowInit;
            const max_operand: u32 = self.arithmetic_max_operand;
            const symbol_groups = 4 + 2 * (max_operand + 1); // START/END/+/- + operands
            const action_groups = 2 * max_operand + 1;
            const required = (symbol_groups + action_groups) * self.arithmetic_group_size;
            if (required > self.nExcitatory()) return error.ArithmeticGroupsExceedExcitatory;
        }

        // Phase 9 builds on the bounded arithmetic action interface. It is
        // explicitly opt-in, so the Phase 8 fixed-window protocol remains the
        // default and the Phase 1 baseline is byte-identical.
        if (self.termination_enabled) {
            if (!self.arithmetic_enabled) return error.TerminationNeedsArithmetic;
            if (self.termination_stable_steps == 0) return error.ZeroTerminationStableWindow;
            if (self.termination_timeout_steps < self.termination_stable_steps)
                return error.TerminationTimeoutTooShort;
            if (self.termination_timeout_reward >= 0.0) return error.BadTimeoutReward;
        }

        // Phase 5 structural plasticity knobs. Permanence lives in [0,1]; the
        // prune thresholds and the tentative init must sit inside it, and a
        // finite budget is required so growth has somewhere to write.
        if (self.structural_plasticity_enabled) {
            if (self.max_out_degree == 0) return error.ZeroConnectionBudget;
            if (self.target_out_degree > self.max_out_degree)
                return error.TargetDegreeExceedsBudget;
            if (!isProbability(self.prune_permanence_min))
                return error.BadPrunePermanence;
            if (!isProbability(self.grow_permanence_init))
                return error.BadGrowPermanence;
            if (!isProbability(self.growth_probability))
                return error.BadGrowthProbability;
            if (!isNonNegative(self.permanence_activity_lr) or
                !isProbability(self.permanence_disuse_decay) or self.coactivity_max <= 0.0 or
                !isProbability(self.weight_permanence_decay))
                return error.BadStructuralCoefficient;
            if (!isNonNegative(self.prune_weight_min) or !isNonNegative(self.grow_weight_init))
                return error.NegativeWeight;
            if (!isNonNegative(self.growth_sigma)) return error.BadGrowthSigma;
            // A tentative synapse must not be born already prunable, or it would
            // be deleted the moment its grace period expires with no chance to
            // consolidate -- defeating the point of growth.
            if (self.grow_permanence_init < self.prune_permanence_min)
                return error.TentativeBornPrunable;
        }

        // Phase 6 consolidation reuses the slow structural clock (permanence decay
        // + pruning), so it cannot run without it.
        if (self.consolidation_enabled) {
            if (!self.structural_plasticity_enabled)
                return error.ConsolidationNeedsStructuralPlasticity;
            if (!self.plasticity_enabled) return error.ConsolidationNeedsPlasticity;
            if (!self.task_enabled and !self.arithmetic_enabled and !self.context_task_enabled)
                return error.ConsolidationNeedsPlasticTask;
            if (!isNonNegative(self.consolidation_lr)) return error.BadConsolidationLr;
        }

        // Phase 7's candidate assemblies are the two task input groups. Keeping
        // the writer set fixed and tiny is the anti-unrestricted-memory rule.
        if (self.workspace_enabled) {
            if (!self.task_enabled) return error.WorkspaceNeedsTask;
            if (self.workspace_capacity == 0 or self.workspace_capacity > 2)
                return error.BadWorkspaceCapacity;
            if (!isDecay(self.workspace_candidate_decay))
                return error.BadWorkspaceCandidateDecay;
            if (!isDecay(self.workspace_state_decay))
                return error.BadWorkspaceStateDecay;
            if (self.workspace_ignition_threshold <= 0.0)
                return error.BadWorkspaceIgnitionThreshold;
            if (!isNonNegative(self.workspace_feedback_current) or !isNonNegative(self.workspace_broadcast_current))
                return error.NegativeWorkspaceCurrent;
        }
        if (self.steps == 0) return error.ZeroRunLength;
    }

    /// Number of excitatory neurons, by the same deterministic rule net.zig uses.
    pub fn nExcitatory(self: Config) u32 {
        return @intFromFloat(@round(@as(f32, @floatFromInt(self.n_neurons)) * self.excitatory_fraction));
    }

    /// Load a Config from a JSON file. Accepts EITHER a bare Config object or a
    /// full run_meta.json (which nests the config under "config"), so any past
    /// run can be replayed by pointing at the run_meta.json it emitted. Missing
    /// fields fall back to the struct defaults, so a config file need only list
    /// the knobs it overrides.
    pub fn loadFromFile(io: std.Io, gpa: std.mem.Allocator, path: []const u8) !Config {
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 20));
        defer gpa.free(bytes);
        return parseJson(gpa, bytes);
    }

    /// See `loadFromFile`. Split out so it can be unit-tested without touching
    /// the filesystem.
    pub fn parseJson(gpa: std.mem.Allocator, bytes: []const u8) !Config {
        // Try the run_meta.json shape first: { "config": { ... }, ... }.
        // ignore_unknown_fields lets us skip the metadata siblings (prng, zig
        // version). A bare Config document has no "config" key, so this parse
        // leaves `config` null and we fall through to the bare parse below.
        const Wrapper = struct { config: ?Config = null };
        if (std.json.parseFromSlice(Wrapper, gpa, bytes, .{ .ignore_unknown_fields = true })) |parsed| {
            defer parsed.deinit();
            if (parsed.value.config) |c| return c;
        } else |_| {}

        // Otherwise the whole document is a bare Config object.
        const parsed = try std.json.parseFromSlice(Config, gpa, bytes, .{ .ignore_unknown_fields = true });
        defer parsed.deinit();
        return parsed.value;
    }
};

/// Written next to every run. If you cannot reconstruct the run from this,
/// the run did not happen.
pub const RunMetadata = struct {
    config: Config,
    prng_algorithm: []const u8 = rng.prng_algorithm,
    prng_impl_version: u32 = rng.prng_impl_version,
    zig_version: []const u8 = builtin.zig_version_string,

    pub fn write(self: RunMetadata, w: *std.Io.Writer) !void {
        var s: std.json.Stringify = .{ .writer = w, .options = .{ .whitespace = .indent_2 } };
        try s.write(self);
    }
};

test "config: default passes validation" {
    const c = Config{};
    try c.validate();
}

test "config: invalid numeric and mechanism combinations are rejected" {
    const Case = struct {
        config: Config,
        expected: anyerror,
    };
    const cases = [_]Case{
        .{ .config = .{ .background_current = std.math.nan(f32) }, .expected = error.NonFiniteParameter },
        .{ .config = .{ .spatial_sigma = 0.0 }, .expected = error.BadSpatialSigma },
        .{ .config = .{ .task_ia_weight_init = -0.1 }, .expected = error.NegativeTaskCurrentOrWeight },
        .{ .config = .{ .task_recurrent_weight = -0.1 }, .expected = error.NegativeTaskCurrentOrWeight },
        .{ .config = .{ .pre_trace_decay = 1.0 }, .expected = error.BadEligibilityDecay },
        .{ .config = .{ .reward_baseline_decay = -0.1 }, .expected = error.BadEligibilityDecay },
        .{ .config = .{ .learning_rate = -0.1 }, .expected = error.BadPlasticityLearningRate },
        .{ .config = .{ .adaptation_decay = 1.0 }, .expected = error.BadAdaptationParameter },
        .{ .config = .{ .structural_plasticity_enabled = true, .weight_permanence_decay = 1.1 }, .expected = error.BadStructuralCoefficient },
        .{ .config = .{ .task_enabled = true, .arithmetic_enabled = true }, .expected = error.OverlappingTaskLayouts },
        .{ .config = .{ .task_enabled = true, .context_task_enabled = true }, .expected = error.OverlappingTaskLayouts },
        .{ .config = .{ .context_task_enabled = true, .context_task_group_size = 20 }, .expected = error.ContextTaskGroupsExceedExcitatory },
        .{ .config = .{ .reservoir_plasticity_enabled = true }, .expected = error.ReservoirPlasticityNeedsPlasticity },
        .{ .config = .{ .structural_plasticity_enabled = true, .consolidation_enabled = true, .task_enabled = true }, .expected = error.ConsolidationNeedsPlasticity },
        .{ .config = .{ .structural_plasticity_enabled = true, .consolidation_enabled = true, .plasticity_enabled = true }, .expected = error.ConsolidationNeedsPlasticTask },
        .{ .config = .{ .steps = 0 }, .expected = error.ZeroRunLength },
    };

    for (cases) |case| try std.testing.expectError(case.expected, case.config.validate());

    // A zero spatial sigma is irrelevant and therefore legal for a uniform graph.
    try (Config{ .uniform_graph = true, .spatial_sigma = 0.0 }).validate();
}

test "config: zero delay is rejected (DEC-001)" {
    var c = Config{};
    c.min_delay = 0;
    try std.testing.expectError(error.ZeroDelayForbidden, c.validate());
}

test "config: negative weights are rejected" {
    var c = Config{};
    c.w_exc_init_lo = -0.1;
    try std.testing.expectError(error.NegativeWeight, c.validate());
}

test "config: neuron sign convention" {
    try std.testing.expectEqual(@as(f32, 1.0), NeuronKind.excitatory.sign());
    try std.testing.expectEqual(@as(f32, -1.0), NeuronKind.inhibitory.sign());
}

test "config: arithmetic layout must fit the excitatory population" {
    var c = Config{ .arithmetic_enabled = true };
    try std.testing.expectError(error.ArithmeticGroupsExceedExcitatory, c.validate());
    c.n_neurons = 200;
    try c.validate();
}

test "config: learned termination needs a bounded action interface" {
    var c = Config{ .termination_enabled = true };
    try std.testing.expectError(error.TerminationNeedsArithmetic, c.validate());
    c.arithmetic_enabled = true;
    c.n_neurons = 200;
    c.termination_stable_steps = 0;
    try std.testing.expectError(error.ZeroTerminationStableWindow, c.validate());
    c.termination_stable_steps = 4;
    c.termination_timeout_steps = 3;
    try std.testing.expectError(error.TerminationTimeoutTooShort, c.validate());
    c.termination_timeout_steps = 4;
    c.termination_timeout_reward = 0.0;
    try std.testing.expectError(error.BadTimeoutReward, c.validate());
    c.termination_timeout_reward = -0.2;
    try c.validate();
}

test "config: workspace requires task candidates and a finite capacity" {
    var c = Config{ .workspace_enabled = true };
    try std.testing.expectError(error.WorkspaceNeedsTask, c.validate());
    c.task_enabled = true;
    c.workspace_capacity = 0;
    try std.testing.expectError(error.BadWorkspaceCapacity, c.validate());
    c.workspace_capacity = 1;
    try c.validate();
}

test "config: parseJson reads a bare Config object and applies defaults for the rest" {
    const c = try Config.parseJson(std.testing.allocator,
        \\{ "master_seed": 42, "n_neurons": 7, "reset_rule": "hard" }
    );
    try std.testing.expectEqual(@as(u64, 42), c.master_seed);
    try std.testing.expectEqual(@as(u32, 7), c.n_neurons);
    try std.testing.expectEqual(ResetRule.hard, c.reset_rule);
    // A field the document did not mention keeps its default.
    try std.testing.expectEqual(@as(f32, 0.8), c.excitatory_fraction);
}

test "config: parseJson round-trips through run metadata (replay a run_meta.json)" {
    // Serialize a config the way a real run does, then parse it back from the
    // run_meta.json shape. The recovered config must match, which is what makes
    // replay-from-metadata trustworthy.
    const original = Config{ .master_seed = 0xABCDEF, .n_neurons = 33, .steps = 123 };

    var out: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer out.deinit();
    try (RunMetadata{ .config = original }).write(&out.writer);

    const recovered = try Config.parseJson(std.testing.allocator, out.written());
    try std.testing.expectEqual(original.master_seed, recovered.master_seed);
    try std.testing.expectEqual(original.n_neurons, recovered.n_neurons);
    try std.testing.expectEqual(original.steps, recovered.steps);
}
