//! Machine-readable provenance sidecars for experiment harnesses.

const std = @import("std");
const builtin = @import("builtin");
const rng = @import("rng.zig");

/// Write an experiment-specific JSON document atomically. `protocol` is an
/// anonymous struct supplied by the harness and should contain every constant,
/// seed list, condition, and config template needed to interpret its CSV.
pub fn write(
    io: std.Io,
    gpa: std.mem.Allocator,
    path: []const u8,
    experiment: []const u8,
    protocol: anytype,
) !void {
    const document = .{
        .schema_version = @as(u32, 1),
        .experiment = experiment,
        .prng_algorithm = rng.prng_algorithm,
        .prng_impl_version = rng.prng_impl_version,
        .zig_version = builtin.zig_version_string,
        .protocol = protocol,
    };

    var out: std.Io.Writer.Allocating = .init(gpa);
    defer out.deinit();
    var stringify: std.json.Stringify = .{
        .writer = &out.writer,
        .options = .{ .whitespace = .indent_2 },
    };
    try stringify.write(document);

    var atomic = try std.Io.Dir.cwd().createFileAtomic(io, path, .{ .replace = true });
    defer atomic.deinit(io);
    var buf: [4096]u8 = undefined;
    var file_writer = atomic.file.writer(io, &buf);
    try file_writer.interface.writeAll(out.written());
    try file_writer.interface.flush();
    try atomic.replace(io);
}

test "provenance: schema serializes PRNG, toolchain, and protocol" {
    var out: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer out.deinit();
    const document = .{
        .prng_algorithm = rng.prng_algorithm,
        .prng_impl_version = rng.prng_impl_version,
        .zig_version = builtin.zig_version_string,
        .protocol = .{ .seeds = [_]u64{ 1, 2 } },
    };
    var stringify: std.json.Stringify = .{ .writer = &out.writer };
    try stringify.write(document);
    try std.testing.expect(std.mem.indexOf(u8, out.written(), "\"prng_impl_version\":2") != null);
    try std.testing.expect(std.mem.indexOf(u8, out.written(), "\"seeds\":[1,2]") != null);
}
