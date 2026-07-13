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

    // -- homeostasis (Phase 2 -- present but off) --------------------------
    homeostasis_enabled: bool = false,
    rate_ema_decay: f32 = 0.99, // lambda_rho
    target_rate: f32 = 0.02, // rho_target, spikes/neuron/step
    homeostasis_lr: f32 = 0.01, // eta_h
    threshold_min: f32 = 0.1,
    threshold_max: f32 = 5.0,

    // -- run --------------------------------------------------------------
    steps: u32 = 2000,

    /// Validate the invariants that must hold before a single step is taken.
    pub fn validate(self: Config) !void {
        // DEC-001. This is the one that will silently corrupt a run if violated.
        if (self.min_delay < 1) return error.ZeroDelayForbidden;
        if (self.max_delay < self.min_delay) return error.BadDelayRange;

        if (self.n_neurons == 0) return error.NoNeurons;
        if (self.excitatory_fraction < 0.0 or self.excitatory_fraction > 1.0)
            return error.BadExcitatoryFraction;
        if (self.membrane_leak < 0.0 or self.membrane_leak > 1.0)
            return error.BadLeak;
        if (self.release_probability < 0.0 or self.release_probability > 1.0)
            return error.BadReleaseProbability;
        if (self.connection_density <= 0.0 or self.connection_density > 1.0)
            return error.BadDensity;

        // w >= 0 always. Sign lives in the neuron type.
        if (self.w_exc_init_lo < 0.0 or self.w_inh_init_lo < 0.0)
            return error.NegativeWeight;
        if (self.w_exc_init_hi < self.w_exc_init_lo or self.w_inh_init_hi < self.w_inh_init_lo)
            return error.BadWeightRange;
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
