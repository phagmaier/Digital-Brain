//! Phase 2 exit-criterion experiment: does the network return toward a target
//! activity range after a moderate perturbation?
//!
//! Design -- a controlled A/B, run on the same timeline:
//!   * settle window : no perturbation. With homeostasis on, the adaptive
//!     thresholds pull the population rate down to target_rate.
//!   * perturb window: a SUSTAINED moderate extra current is injected into
//!     every neuron. This is the strong form of the test -- the drive stays on,
//!     so a merely *stable* network would sit at an elevated rate forever. Only
//!     an actual homeostat returns the rate to the target band.
//!
//! Two conditions share the identical perturbation:
//!   ON  -- homeostasis enabled: should re-enter the band despite the drive.
//!   OFF -- the control: should stay pinned above the band.
//! The contrast between them is the proof; either alone is unconvincing.
//!
//! Writes perturb.csv (rate + mean threshold per step, both conditions) and
//! prints a PASS/FAIL verdict.
//!
//! Build/run:  zig build perturb

const std = @import("std");
const cfg = @import("config.zig");
const sim = @import("sim.zig");

const settle_steps: u32 = 4000;
const perturb_steps: u32 = 4000;
const total_steps: u32 = settle_steps + perturb_steps;

/// Sustained extra current added to every neuron during the perturb window.
/// "Moderate": comparable to background_current (0.35), enough to clearly push
/// the rate out of the band without saturating.
const perturb_current: f32 = 0.40;

/// The target band, as a multiplicative tolerance around target_rate.
const band_lo_frac: f32 = 0.5;
const band_hi_frac: f32 = 1.6;

fn baseConfig() cfg.Config {
    return .{
        .master_seed = 0xB0A710,
        .n_neurons = 100,
        .steps = total_steps,
        .target_rate = 0.05,
        // Faster than the 0.01 default so the demo converges within the window;
        // still slow relative to the 100-step rate-EMA time constant.
        .homeostasis_lr = 0.05,
        // threshold_max is left at the default (12), which is exactly what this
        // experiment needs: the default ceiling must give the homeostat enough
        // authority to reject a sustained doubling of drive. If it did not, this
        // run would FAIL -- which is the point of exercising the default here.
    };
}

const Trace = struct {
    rate: []f32, // mean_rate_ema per step
    thresh: []f32, // mean_threshold per step

    fn init(gpa: std.mem.Allocator) !Trace {
        return .{
            .rate = try gpa.alloc(f32, total_steps),
            .thresh = try gpa.alloc(f32, total_steps),
        };
    }
    fn deinit(self: *Trace, gpa: std.mem.Allocator) void {
        gpa.free(self.rate);
        gpa.free(self.thresh);
    }
};

/// Run one condition end to end, injecting the sustained perturbation during
/// the perturb window, and record the rate/threshold traces.
fn run(gpa: std.mem.Allocator, homeostasis: bool, ext: []const f32) !Trace {
    var c = baseConfig();
    c.homeostasis_enabled = homeostasis;

    var s = try sim.Sim.init(gpa, c);
    defer s.deinit(gpa);

    var trace = try Trace.init(gpa);
    for (0..total_steps) |t| {
        const external: ?[]const f32 = if (t >= settle_steps) ext else null;
        const m = s.step(external);
        trace.rate[t] = m.mean_rate_ema;
        trace.thresh[t] = m.mean_threshold;
    }
    return trace;
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    const ext = try gpa.alloc(f32, baseConfig().n_neurons);
    defer gpa.free(ext);
    @memset(ext, perturb_current);

    var on = try run(gpa, true, ext);
    defer on.deinit(gpa);
    var off = try run(gpa, false, ext);
    defer off.deinit(gpa);

    // ---- write the trace CSV --------------------------------------------
    var out: std.Io.Writer.Allocating = .init(gpa);
    defer out.deinit();
    const w = &out.writer;
    try w.print("t,phase,rate_on,thresh_on,rate_off,thresh_off\n", .{});
    for (0..total_steps) |t| {
        const phase = if (t < settle_steps) "settle" else "perturb";
        try w.print("{d},{s},{d:.6},{d:.5},{d:.6},{d:.5}\n", .{
            t, phase, on.rate[t], on.thresh[t], off.rate[t], off.thresh[t],
        });
    }
    try writeAtomic(io, "perturb.csv", out.written());

    // ---- verdict --------------------------------------------------------
    const c = baseConfig();
    const band_lo = c.target_rate * band_lo_frac;
    const band_hi = c.target_rate * band_hi_frac;

    // Averages over the last 10% of each window, to read the settled value
    // rather than a transient.
    const settle_tail = windowMean(on.rate, settle_steps - settle_steps / 10, settle_steps);
    const on_final = windowMean(on.rate, total_steps - total_steps / 20, total_steps);
    const off_final = windowMean(off.rate, total_steps - total_steps / 20, total_steps);
    const on_peak = windowMax(on.rate, settle_steps, settle_steps + perturb_steps / 10);

    const converged = settle_tail >= band_lo and settle_tail <= band_hi;
    const perturbed_out = on_peak > band_hi; // the perturbation actually kicked it out
    const recovered = on_final >= band_lo and on_final <= band_hi;
    const control_stays_out = off_final > band_hi;
    const pass = converged and perturbed_out and recovered and control_stays_out;

    var buf: [2048]u8 = undefined;
    var stdout = std.Io.File.stdout().writer(io, &buf);
    const o = &stdout.interface;
    try o.print(
        \\
        \\-- Phase 2 homeostasis: perturbation recovery -----------------
        \\  target_rate        {d:.4}
        \\  target band        [{d:.4}, {d:.4}]
        \\  perturb current    +{d:.3}  (sustained, steps {d}..{d})
        \\
        \\  settle tail (ON)   {d:.4}   {s}
        \\  perturb peak (ON)  {d:.4}   {s}
        \\  final rate  (ON)   {d:.4}   {s}
        \\  final rate  (OFF)  {d:.4}   {s}
        \\
    , .{
        c.target_rate,          band_lo,
        band_hi,                perturb_current,
        settle_steps,           total_steps,
        settle_tail,            checkStr(converged, "converged to band", "did NOT converge"),
        on_peak,                checkStr(perturbed_out, "left band under drive", "perturbation too weak"),
        on_final,               checkStr(recovered, "returned to band", "did NOT recover"),
        off_final,              checkStr(control_stays_out, "stayed above band (as expected)", "control did not stay out -- test is not isolating homeostasis"),
    });
    try o.print("  VERDICT: {s}\n\n", .{if (pass) "PASS -- homeostasis regulates back to target." else "FAIL -- see failing line(s) above."});
    try o.print("  wrote perturb.csv\n\n", .{});
    try o.flush();
}

fn checkStr(ok: bool, yes: []const u8, no: []const u8) []const u8 {
    return if (ok) yes else no;
}

fn windowMean(xs: []const f32, lo: u32, hi: u32) f32 {
    var sum: f32 = 0;
    for (lo..hi) |i| sum += xs[i];
    return sum / @as(f32, @floatFromInt(hi - lo));
}

fn windowMax(xs: []const f32, lo: u32, hi: u32) f32 {
    var mx: f32 = 0;
    for (lo..hi) |i| mx = @max(mx, xs[i]);
    return mx;
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
