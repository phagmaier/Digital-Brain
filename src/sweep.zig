//! Parameter sweep harness. Runs the simulator across a grid of configs and
//! writes one summary row per run to `sweep.csv`. This is the tool EXP-001
//! actually uses: "find the parameter regime where the network is alive and
//! sparse" is a search, and this runs the search reproducibly.
//!
//! Each grid point uses its own (seed, background_current, w_inh) config, runs
//! a full simulation, and records the same Summary numbers the single-run
//! binary prints -- so a promising row here can be reproduced exactly by
//! feeding the same config to `brain`.
//!
//! Build/run:  zig build sweep
//! Edit the grids below to change what is explored; the file is meant to be
//! hand-edited, the same way main.zig's config literal is.

const std = @import("std");
const cfg = @import("config.zig");
const sim = @import("sim.zig");
const log = @import("log.zig");
const provenance = @import("provenance.zig");

// ---- the grid -------------------------------------------------------------
// Kept as plain arrays so the sweep is obvious and diffable. The Cartesian
// product of these is what gets run.

const seeds = [_]u64{ 1, 2, 3 };
const background_currents = [_]f32{ 0.20, 0.35, 0.50 };
const w_inh_los = [_]f32{ 0.5, 1.2, 3.0, 6.0 };

const steps: u32 = 2000;

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    var out: std.Io.Writer.Allocating = .init(gpa);
    defer out.deinit();
    const w = &out.writer;

    try w.print(
        "seed,background_current,w_inh_lo,n_synapses,mean_firing_rate,spikes_per_step,silent_fraction,mean_u,ei_current_ratio,verdict\n",
        .{},
    );

    var stdout_buf: [4096]u8 = undefined;
    var stdout = std.Io.File.stdout().writer(io, &stdout_buf);

    var n_runs: u32 = 0;
    for (seeds) |seed| {
        for (background_currents) |bg| {
            for (w_inh_los) |w_inh_lo| {
                const config = cfg.Config{
                    .master_seed = seed,
                    .steps = steps,
                    .background_current = bg,
                    .w_inh_init_lo = w_inh_lo,
                    // Keep the hi/lo ratio fixed so only the level moves.
                    .w_inh_init_hi = w_inh_lo * 2.0,
                };
                try config.validate();

                const summary = try runOne(gpa, config);
                try w.print("{d},{d:.3},{d:.3},{d},{d:.6},{d:.4},{d:.4},{d:.4},{d:.4},{s}\n", .{
                    seed,
                    bg,
                    w_inh_lo,
                    summary.n_synapses,
                    summary.mean_firing_rate,
                    summary.spikes_per_step,
                    summary.silent_fraction,
                    summary.mean_u,
                    summary.ei_current_ratio,
                    verdict(summary.mean_firing_rate),
                });
                n_runs += 1;
            }
        }
    }

    try writeAtomic(io, "sweep.csv", out.written());
    try provenance.write(io, gpa, "sweep.meta.json", "parameter_sweep", .{
        .seeds = seeds,
        .background_currents = background_currents,
        .w_inh_los = w_inh_los,
        .steps = steps,
        .base_config = cfg.Config{},
    });

    try stdout.interface.print("swept {d} configs -> sweep.csv\n", .{n_runs});
    try stdout.interface.flush();
}

/// Run one config to completion and return its post-burn-in Summary. Mirrors
/// the loop in main.zig, minus the per-artefact CSV writes.
fn runOne(gpa: std.mem.Allocator, config: cfg.Config) !log.Summary {
    var s = try sim.Sim.init(gpa, config);
    defer s.deinit(gpa);

    var logger = try log.Logger.init(gpa, config.steps);
    defer logger.deinit(gpa);

    for (0..config.steps) |_| {
        const m = s.step(null);
        try logger.record(gpa, &s, m);
    }

    const burn_in: u32 = config.steps / 10;
    return log.Summary.compute(&s, logger, burn_in);
}

/// Same thresholds as log.Summary.print, as a short tag for the CSV column.
fn verdict(rate: f64) []const u8 {
    if (rate < 0.001) return "dead";
    if (rate > 0.20) return "saturated";
    return "alive";
}

/// Atomic write, same pattern as main.zig.
fn writeAtomic(io: std.Io, path: []const u8, bytes: []const u8) !void {
    var atomic = try std.Io.Dir.cwd().createFileAtomic(io, path, .{ .replace = true });
    defer atomic.deinit(io);

    var buf: [4096]u8 = undefined;
    var file_writer = atomic.file.writer(io, &buf);
    try file_writer.interface.writeAll(bytes);
    try file_writer.interface.flush();
    try atomic.replace(io);
}
