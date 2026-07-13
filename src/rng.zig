//! Randomness. See DEC-004 (Stateless derived RNG keys).
//!
//! The PRNG is VENDORED ON PURPOSE. Naming `std.Random.DefaultPrng` is not
//! sufficient for reproducibility: a stdlib change silently changes the stream,
//! and every run in the log book becomes unreproducible with no error message.
//! xoshiro256++ and splitmix64 are ~30 lines each. Owning them is cheaper than
//! owning the bug.
//!
//! Two classes of randomness, and the distinction is load-bearing:
//!
//!   DERIVED KEYS (stateless) -- init, task, action, growth.
//!     key = hash(master_seed, label, index)
//!     Episode 500 is DEFINITIONALLY the same task under every configuration,
//!     no matter what else changed in the code. This gives counterfactual
//!     control across ablations, which a running stream does not.
//!
//!   RUNNING STREAMS (stateful) -- firing, release.
//!     High-volume per-step draws. Cross-variant alignment is impossible here
//!     anyway once dynamics diverge, so a running stream is fine.

const std = @import("std");

/// Bump this if the algorithm below is ever changed. It goes in run metadata.
pub const prng_algorithm = "xoshiro256++";
pub const prng_impl_version = 1;

// ---------------------------------------------------------------------------
// splitmix64 -- used for seeding and for key derivation finalization.
// ---------------------------------------------------------------------------

pub const SplitMix64 = struct {
    state: u64,

    pub fn init(seed: u64) SplitMix64 {
        return .{ .state = seed };
    }

    pub fn next(self: *SplitMix64) u64 {
        self.state +%= 0x9E3779B97F4A7C15;
        var z = self.state;
        z = (z ^ (z >> 30)) *% 0xBF58476D1CE4E5B9;
        z = (z ^ (z >> 27)) *% 0x94D049BB133111EB;
        return z ^ (z >> 31);
    }
};

/// Bit-mixing finalizer (the splitmix64 avalanche step, standalone).
fn mix64(x: u64) u64 {
    var z = x;
    z = (z ^ (z >> 30)) *% 0xBF58476D1CE4E5B9;
    z = (z ^ (z >> 27)) *% 0x94D049BB133111EB;
    return z ^ (z >> 31);
}

// ---------------------------------------------------------------------------
// xoshiro256++
// ---------------------------------------------------------------------------

pub const Rng = struct {
    s: [4]u64,

    /// Seed all four words from splitmix64, as the reference implementation
    /// recommends. Never seed xoshiro directly from a small integer.
    pub fn init(seed: u64) Rng {
        var sm = SplitMix64.init(seed);
        return .{ .s = .{ sm.next(), sm.next(), sm.next(), sm.next() } };
    }

    pub fn next(self: *Rng) u64 {
        const result = std.math.rotl(u64, self.s[0] +% self.s[3], 23) +% self.s[0];
        const t = self.s[1] << 17;

        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = std.math.rotl(u64, self.s[3], 45);

        return result;
    }

    /// Uniform in [0, 1). 24-bit mantissa, so exactly representable in f32.
    pub fn float01(self: *Rng) f32 {
        const bits: u64 = self.next() >> 40; // top 24 bits
        return @as(f32, @floatFromInt(bits)) * 0x1.0p-24;
    }

    /// Uniform in [0, n). Rejection-sampled, so no modulo bias.
    /// n must be > 0.
    pub fn below(self: *Rng, n: u64) u64 {
        std.debug.assert(n > 0);
        if (n == 1) return 0;
        // Largest multiple of n that fits in u64; reject above it.
        const limit = std.math.maxInt(u64) - (std.math.maxInt(u64) % n) - (n - 1);
        while (true) {
            const r = self.next();
            if (r <= limit) return r % n;
        }
    }

    /// Uniform in [lo, hi).
    pub fn range(self: *Rng, lo: f32, hi: f32) f32 {
        return lo + (hi - lo) * self.float01();
    }

    /// Standard normal, Box-Muller. Only used for weight init; not on the hot path.
    pub fn normal(self: *Rng) f32 {
        // Guard against log(0).
        // Named sample_u / sample_v to avoid shadowing the primitive type `u1`.
        var sample_u = self.float01();
        if (sample_u < 1e-7) sample_u = 1e-7;
        const sample_v = self.float01();
        const r = @sqrt(-2.0 * @log(sample_u));
        return r * @cos(2.0 * std.math.pi * sample_v);
    }

    pub fn bernoulli(self: *Rng, p: f32) bool {
        return self.float01() < p;
    }
};

// ---------------------------------------------------------------------------
// Stateless key derivation (DEC-004)
// ---------------------------------------------------------------------------

/// Named RNG streams. Every derived key carries one of these labels so that
/// two different subsystems can never collide on the same (seed, index) pair.
pub const Stream = enum {
    init_graph,
    task,
    action,
    growth,

    fn label(self: Stream) u64 {
        // Fixed constants, not @intFromEnum -- reordering the enum must not
        // change any stream.
        return switch (self) {
            .init_graph => 0x1111_1111_1111_1111,
            .task => 0x2222_2222_2222_2222,
            .action => 0x3333_3333_3333_3333,
            .growth => 0x4444_4444_4444_4444,
        };
    }
};

/// key = hash(master_seed, stream_label, index)
///
/// This is the whole point of DEC-004: `taskRng(seed, 500)` is the same task in
/// every ablation variant, regardless of how many random draws any other
/// subsystem consumed. Reproducible AND counterfactually controlled.
pub fn derive(master_seed: u64, stream: Stream, index: u64) u64 {
    var h: u64 = 0xcbf2_9ce4_8422_2325; // FNV-1a 64 offset basis
    h = mix64(h ^ master_seed);
    h = mix64(h ^ stream.label());
    h = mix64(h ^ index);
    return h;
}

/// Convenience: a fresh, independent Rng for a given (stream, index).
pub fn derived(master_seed: u64, stream: Stream, index: u64) Rng {
    return Rng.init(derive(master_seed, stream, index));
}

// ===========================================================================
// Tests
// ===========================================================================

const testing = std.testing;

test "rng: same seed produces same stream" {
    var a = Rng.init(12345);
    var b = Rng.init(12345);
    for (0..1000) |_| {
        try testing.expectEqual(a.next(), b.next());
    }
}

test "rng: different seeds diverge" {
    var a = Rng.init(1);
    var b = Rng.init(2);
    var same: usize = 0;
    for (0..100) |_| {
        if (a.next() == b.next()) same += 1;
    }
    try testing.expect(same == 0);
}

test "rng: float01 stays in [0,1)" {
    var r = Rng.init(7);
    for (0..100_000) |_| {
        const x = r.float01();
        try testing.expect(x >= 0.0);
        try testing.expect(x < 1.0);
    }
}

test "rng: float01 mean is near 0.5" {
    var r = Rng.init(99);
    var sum: f64 = 0;
    const n = 200_000;
    for (0..n) |_| sum += r.float01();
    const mean = sum / @as(f64, n);
    try testing.expect(@abs(mean - 0.5) < 0.01);
}

test "rng: bernoulli is statistically correct" {
    // This is the spec's "release probability behaves statistically correctly"
    // check, at the RNG layer. The synapse-level version lives in sim.zig.
    var r = Rng.init(4242);
    const p: f32 = 0.3;
    const n: usize = 200_000;
    var hits: usize = 0;
    for (0..n) |_| {
        if (r.bernoulli(p)) hits += 1;
    }
    const observed = @as(f64, @floatFromInt(hits)) / @as(f64, @floatFromInt(n));
    // sigma = sqrt(p(1-p)/n) ~= 0.001. Five sigma is comfortably 0.005.
    try testing.expect(@abs(observed - 0.3) < 0.005);
}

test "rng: below() is unbiased over a small range" {
    var r = Rng.init(31337);
    var counts = [_]usize{0} ** 7;
    const n: usize = 140_000;
    for (0..n) |_| counts[r.below(7)] += 1;
    const expected = @as(f64, @floatFromInt(n)) / 7.0;
    for (counts) |c| {
        const dev = @abs(@as(f64, @floatFromInt(c)) - expected) / expected;
        try testing.expect(dev < 0.05);
    }
}

test "derive: task key is independent of other streams' draw counts" {
    // THE point of DEC-004. Two "variants" burn wildly different numbers of
    // draws from their firing streams; their task sequences must be identical.
    const seed: u64 = 0xDEAD_BEEF;

    var variant_a_tasks: [64]u64 = undefined;
    var variant_b_tasks: [64]u64 = undefined;

    var firing_a = Rng.init(1);
    var firing_b = Rng.init(1);

    for (0..64) |ep| {
        // Variant A burns 3 firing draws per episode.
        for (0..3) |_| _ = firing_a.next();
        var t = derived(seed, .task, ep);
        variant_a_tasks[ep] = t.next();

        // Variant B burns 977. (Say it grew some synapses.)
        for (0..977) |_| _ = firing_b.next();
        var t2 = derived(seed, .task, ep);
        variant_b_tasks[ep] = t2.next();
    }

    try testing.expectEqualSlices(u64, &variant_a_tasks, &variant_b_tasks);
}

test "derive: streams do not collide" {
    const seed: u64 = 777;
    const a = derive(seed, .task, 5);
    const b = derive(seed, .action, 5);
    const c = derive(seed, .growth, 5);
    const d = derive(seed, .init_graph, 5);
    try testing.expect(a != b);
    try testing.expect(a != c);
    try testing.expect(a != d);
    try testing.expect(b != c);
    try testing.expect(b != d);
    try testing.expect(c != d);
}

test "derive: adjacent indices are decorrelated" {
    const seed: u64 = 5;
    var prev = derive(seed, .task, 0);
    for (1..500) |i| {
        const cur = derive(seed, .task, i);
        try testing.expect(cur != prev);
        prev = cur;
    }
}
