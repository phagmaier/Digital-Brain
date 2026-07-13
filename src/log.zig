//! Logging. Raster events, per-step metrics, and run metadata.
//!
//! The rule from the spec: "Config + seed + toolchain version saved with every
//! run." A run you cannot reconstruct is a run you did not do.

const std = @import("std");
const cfg = @import("config.zig");
const sim = @import("sim.zig");
const net = @import("net.zig");

const Allocator = std.mem.Allocator;

pub const SpikeEvent = struct {
    t: u32,
    neuron: net.NeuronId,
    kind: cfg.NeuronKind,
};

pub const Logger = struct {
    spikes: std.ArrayList(SpikeEvent),
    steps: std.ArrayList(sim.StepMetrics),

    pub fn init(gpa: Allocator, expected_steps: u32) !Logger {
        return .{
            .spikes = try std.ArrayList(SpikeEvent).initCapacity(gpa, expected_steps * 4),
            .steps = try std.ArrayList(sim.StepMetrics).initCapacity(gpa, expected_steps),
        };
    }

    pub fn deinit(self: *Logger, gpa: Allocator) void {
        self.spikes.deinit(gpa);
        self.steps.deinit(gpa);
    }

    pub fn record(self: *Logger, gpa: Allocator, s: *const sim.Sim, m: sim.StepMetrics) !void {
        try self.steps.append(gpa, m);
        const nrn = &s.network.neurons;
        for (0..nrn.n) |i| {
            if (nrn.fired[i]) {
                try self.spikes.append(gpa, .{
                    .t = m.t,
                    .neuron = @intCast(i),
                    .kind = nrn.kind[i],
                });
            }
        }
    }

    /// Raster: one row per spike. This is the primary Phase 1 artefact -- the
    /// thing you actually look at to decide whether the dynamics are healthy.
    pub fn writeRaster(self: Logger, w: *std.Io.Writer) !void {
        try w.print("t,neuron,kind\n", .{});
        for (self.spikes.items) |e| {
            try w.print("{d},{d},{s}\n", .{
                e.t,
                e.neuron,
                @tagName(e.kind),
            });
        }
    }

    /// Per-step metrics. The E/I current columns are the ones that tell you WHY
    /// the network died or exploded, rather than merely that it did.
    pub fn writeMetrics(self: Logger, w: *std.Io.Writer) !void {
        try w.print("t,spikes,exc_spikes,inh_spikes,mean_u,exc_current,inh_current,scheduled_events\n", .{});
        for (self.steps.items) |m| {
            try w.print("{d},{d},{d},{d},{d:.5},{d:.5},{d:.5},{d}\n", .{
                m.t,
                m.spikes,
                m.exc_spikes,
                m.inh_spikes,
                m.mean_u,
                m.exc_current,
                m.inh_current,
                m.scheduled_events,
            });
        }
    }

    /// Neuron positions, so the network view (Part VI) has coordinates to plot.
    pub fn writeNeurons(s: *const sim.Sim, w: *std.Io.Writer) !void {
        try w.print("id,kind,pos_x,pos_y,threshold\n", .{});
        const nrn = &s.network.neurons;
        for (0..nrn.n) |i| {
            try w.print("{d},{s},{d:.5},{d:.5},{d:.5}\n", .{
                i,
                @tagName(nrn.kind[i]),
                nrn.pos_x[i],
                nrn.pos_y[i],
                nrn.threshold[i],
            });
        }
    }

    pub fn writeSynapses(s: *const sim.Sim, w: *std.Io.Writer) !void {
        try w.print("id,source,target,weight,delay,p_release\n", .{});
        const syn = &s.network.synapses;
        for (0..syn.n) |k| {
            try w.print("{d},{d},{d},{d:.5},{d},{d:.5}\n", .{
                k,
                syn.source[k],
                syn.target[k],
                syn.weight[k],
                syn.delay[k],
                syn.p_release[k],
            });
        }
    }
};

/// Run-level summary. These are the numbers that decide whether EXP-001 passes.
pub const Summary = struct {
    steps: u32,
    n_neurons: u32,
    n_synapses: u32,
    mean_firing_rate: f64, // spikes / neuron / step
    spikes_per_step: f64,
    silent_fraction: f64, // neurons that never fired
    mean_u: f64,
    ei_current_ratio: f64, // |I_exc| / |I_inh|

    pub fn compute(s: *const sim.Sim, lg: Logger, burn_in: u32) Summary {
        const n = s.network.neurons.n;

        var counted: u64 = 0;
        var spike_total: u64 = 0;
        var u_total: f64 = 0;
        var exc_total: f64 = 0;
        var inh_total: f64 = 0;

        for (lg.steps.items) |m| {
            if (m.t < burn_in) continue;
            counted += 1;
            spike_total += m.spikes;
            u_total += m.mean_u;
            exc_total += m.exc_current;
            inh_total += m.inh_current;
        }

        var ever_fired = std.mem.zeroes([4096]bool);
        for (lg.spikes.items) |e| {
            if (e.t >= burn_in and e.neuron < 4096) ever_fired[e.neuron] = true;
        }
        var silent: u32 = 0;
        for (0..n) |i| {
            if (!ever_fired[i]) silent += 1;
        }

        const cf = @as(f64, @floatFromInt(@max(counted, 1)));
        const nf = @as(f64, @floatFromInt(n));

        return .{
            .steps = @intCast(lg.steps.items.len),
            .n_neurons = n,
            .n_synapses = s.network.synapses.n,
            .mean_firing_rate = @as(f64, @floatFromInt(spike_total)) / (cf * nf),
            .spikes_per_step = @as(f64, @floatFromInt(spike_total)) / cf,
            .silent_fraction = @as(f64, @floatFromInt(silent)) / nf,
            .mean_u = u_total / cf,
            .ei_current_ratio = exc_total / @max(inh_total, 1e-9),
        };
    }

    pub fn print(self: Summary, w: *std.Io.Writer) !void {
        try w.print(
            \\
            \\-- run summary (post burn-in) ------------------------------
            \\  neurons            {d}
            \\  synapses           {d}
            \\  steps              {d}
            \\  mean firing rate   {d:.5}  (spikes / neuron / step)
            \\  spikes per step    {d:.3}
            \\  silent fraction    {d:.3}
            \\  mean u             {d:.4}
            \\  E/I current ratio  {d:.3}
            \\
        , .{
            self.n_neurons,
            self.n_synapses,
            self.steps,
            self.mean_firing_rate,
            self.spikes_per_step,
            self.silent_fraction,
            self.mean_u,
            self.ei_current_ratio,
        });

        // The Phase 1 exit criterion, stated out loud rather than left for you
        // to eyeball at 2am.
        if (self.mean_firing_rate < 0.001) {
            try w.print("  VERDICT: DEAD. See 'Everything goes silent' in Failure Modes.\n", .{});
        } else if (self.mean_firing_rate > 0.20) {
            try w.print("  VERDICT: SATURATED. See 'Everything fires continuously'.\n", .{});
        } else {
            try w.print("  VERDICT: alive and sparse. EXP-001 smoke test passes.\n", .{});
        }
    }
};
