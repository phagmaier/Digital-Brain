//! Brain-Inspired Local Learning System -- Phase 0/1 entry point.
//!
//! Phase 1 target, verbatim from the spec:
//!
//!   "A reproducible simulator of 100 fixed-graph stochastic spiking neurons --
//!    E/I types, rest-relative leaky membrane, refractory periods, probabilistic
//!    firing, probabilistic synaptic release, delay >= 1 event ring buffer,
//!    derived-key RNG, raster logging."
//!
//! Later-phase mechanisms, including the Phase 7 workspace broadcast, remain
//! opt-in through Config; the built-in default is still the Phase 1 baseline.

const std = @import("std");
const cfg = @import("config.zig");
const sim = @import("sim.zig");
const log = @import("log.zig");
const rng = @import("rng.zig");

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    // Optional first argument: a path to a JSON config (a bare Config object,
    // or a run_meta.json to replay). Whatever we use gets written back into
    // run_meta.json below, so every run stays reconstructible.
    const argv = try init.minimal.args.toSlice(init.arena.allocator());
    const config_path: ?[]const u8 = if (argv.len > 1) argv[1] else null;

    const config = if (config_path) |path|
        cfg.Config.loadFromFile(io, gpa, path) catch |err| {
            var ebuf: [512]u8 = undefined;
            var estderr = std.Io.File.stderr().writer(io, &ebuf);
            try estderr.interface.print(
                "error: could not load config '{s}': {s}\n",
                .{ path, @errorName(err) },
            );
            try estderr.interface.flush();
            return err;
        }
    else
        // Phase 1 defaults, used when no config file is given.
        cfg.Config{
            .master_seed = 0xC0FFEE,
            .n_neurons = 100,
            .steps = 3000,
        };

    try config.validate();

    var s = try sim.Sim.init(gpa, config);
    defer s.deinit(gpa);

    var logger = try log.Logger.init(gpa, config.steps);
    defer logger.deinit(gpa);

    for (0..config.steps) |_| {
        const m = s.step(null); // no external input in Phase 1
        try logger.record(gpa, &s, m);
    }

    // ---- artefacts ------------------------------------------------------

    try writeFile(io, gpa, "raster.csv", logger, &s, .raster);
    try writeFile(io, gpa, "metrics.csv", logger, &s, .metrics);
    try writeFile(io, gpa, "neurons.csv", logger, &s, .neurons);
    try writeFile(io, gpa, "synapses.csv", logger, &s, .synapses);
    try writeMeta(io, gpa, config);

    // ---- summary --------------------------------------------------------

    const burn_in: u32 = config.steps / 10;
    const summary = log.Summary.compute(&s, logger, burn_in);

    var stdout_buf: [4096]u8 = undefined;
    var stdout = std.Io.File.stdout().writer(io, &stdout_buf);
    try summary.print(&stdout.interface);
    try stdout.interface.print(
        "  config             {s}\n",
        .{config_path orelse "built-in defaults"},
    );
    try stdout.interface.print(
        "  wrote raster.csv, metrics.csv, neurons.csv, synapses.csv, run_meta.json\n\n",
        .{},
    );
    try stdout.interface.flush();
}

const Artefact = enum { raster, metrics, neurons, synapses };

fn writeFile(
    io: std.Io,
    gpa: std.mem.Allocator,
    path: []const u8,
    logger: log.Logger,
    s: *const sim.Sim,
    which: Artefact,
) !void {
    var out: std.Io.Writer.Allocating = .init(gpa);
    defer out.deinit();

    switch (which) {
        .raster => try logger.writeRaster(&out.writer),
        .metrics => try logger.writeMetrics(&out.writer),
        .neurons => try log.Logger.writeNeurons(s, &out.writer),
        .synapses => try log.Logger.writeSynapses(s, &out.writer),
    }

    try writeAtomic(io, path, out.written());
}

fn writeMeta(io: std.Io, gpa: std.mem.Allocator, config: cfg.Config) !void {
    const meta = cfg.RunMetadata{ .config = config };

    var out: std.Io.Writer.Allocating = .init(gpa);
    defer out.deinit();
    try meta.write(&out.writer);

    try writeAtomic(io, "run_meta.json", out.written());
}

/// Atomically write `bytes` to `path` under the process cwd (Zig 0.16 Io API).
fn writeAtomic(io: std.Io, path: []const u8, bytes: []const u8) !void {
    var atomic = try std.Io.Dir.cwd().createFileAtomic(io, path, .{ .replace = true });
    defer atomic.deinit(io);

    var buf: [4096]u8 = undefined;
    var file_writer = atomic.file.writer(io, &buf);
    try file_writer.interface.writeAll(bytes);
    try file_writer.interface.flush();
    try atomic.replace(io);
}

// Pull every module's tests into `zig build test`.
test {
    std.testing.refAllDecls(@This());
    _ = rng;
    _ = cfg;
    _ = @import("net.zig");
    _ = @import("task.zig");
    _ = @import("arithmetic.zig");
    _ = sim;
    _ = log;
}
