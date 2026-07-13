# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A reproducible simulator of fixed-graph stochastic spiking neurons, written in Zig (0.16.0, per `build.zig.zon` `minimum_zig_version`). This is **Phase 1**: 100 E/I neurons, rest-relative leaky membrane, refractory periods, probabilistic firing, probabilistic synaptic release, delayed delivery. **No learning, no growth, no workspace** — those later phases are scaffolded (fields allocated, config knobs present, code paths stubbed with `// Phase N` comments) but deliberately inert until Phase 1's exit criteria pass.

## Commands

```sh
zig build              # compile to zig-out/bin/brain (and zig-out/bin/sweep)
zig build run          # build + run; writes CSV artefacts to cwd, prints run summary
zig build run -- cfg.json   # run with a JSON config (bare Config, or a run_meta.json to replay)
zig build test         # run all tests (both test executables, in parallel)
zig build sweep        # run the parameter-sweep harness -> sweep.csv

./scripts/check-determinism.sh [cfg.json]   # runs the binary twice, fails if artefacts differ
uv run scripts/plot_raster.py               # render raster.png from raster.csv + metrics.csv
```

Run a single test by name filter (Zig has no per-file test target here; both test binaries honor the filter):

```sh
zig build test --                              # (no direct name flag via build; use the compiler)
zig test src/sim.zig --test-filter "refractory"   # run one test from one file directly
```

`zig build run` overwrites `raster.csv`, `metrics.csv`, `neurons.csv`, `synapses.csv`, and `run_meta.json` in the working directory (these are committed as sample outputs, not ignored). With no argument the binary uses the default `Config{}` literal in `main.zig`; with a JSON path argument it loads that config (`Config.loadFromFile`, which accepts either a bare Config object or a `run_meta.json` to replay a prior run). Missing JSON fields fall back to struct defaults, so a config file need only list the knobs it overrides.

The `sweep` executable (`src/sweep.zig`, its own build root) runs a Cartesian grid of configs and writes one `Summary` row per run to `sweep.csv`; edit the grid arrays at the top of that file. The two helper scripts live in `scripts/`: `check-determinism.sh` is the Phase-0 exit criterion as a guard (byte-compares artefacts across two runs of the built binary), and `plot_raster.py` uses `uv`'s inline PEP-723 dependencies so it needs no venv setup.

## Architecture

Six source files under `src/`, all pure Zig, no dependencies:

- **`config.zig`** — `Config` (every knob that affects a run), `NeuronKind` (E/I + its `sign()`), `ResetRule`, and `RunMetadata` (serialized to JSON alongside every run). `Config.validate()` enforces the hard invariants before any step runs.
- **`rng.zig`** — vendored `xoshiro256++` + `splitmix64` and the stateless key-derivation scheme. See DEC-004 below.
- **`net.zig`** — `Neurons` and `Synapses` (structure-of-arrays), `Network.build()` which constructs the fixed sparse graph.
- **`sim.zig`** — `EventQueue` (delay ring buffer), `StepMetrics`, and `Sim` with the normative `step()` ordering.
- **`log.zig`** — `Logger` (raster + per-step metrics), CSV writers, and `Summary` (the run verdict: DEAD / SATURATED / alive-and-sparse).
- **`main.zig`** — wires it together; runs the loop, writes artefacts atomically, prints the summary.
- **`root.zig`** — the library module root (currently only a placeholder `add`); exists because `build.zig` builds *two* test executables — one rooted at `root.zig` (the `brain` module) and one at `main.zig` (which `refAllDecls` + explicit `_ = @import(...)` to pull in every module's tests).

### The load-bearing design decisions (DEC-xxx)

The code is written around a set of numbered design decisions referenced by ID in comments. **These are invariants, not preferences — do not "simplify" them away.** When touching relevant code, preserve them and keep the DEC references:

- **DEC-001 — delay ≥ 1, always.** A spike emitted at `t` must not affect its target before `t+1`. Zero delay would make behavior depend on iteration order or land in an already-drained ring bucket. Asserted at graph construction (`net.zig`) *and* at `EventQueue.schedule` (`sim.zig`), and rejected by `Config.validate()`.
- **DEC-002 — rest-relative membrane.** `u = V - V_rest`, so rest is exactly `u = 0` and leak decays toward zero. There is only one coordinate system; never reintroduce an absolute `V_rest`.
- **DEC-003 — reset decrement decoupled from threshold.** Subtractive reset subtracts `reset_decrement`, not `theta`. With sigmoid firing there is no clean threshold crossing, so subtracting `theta` specifically is arbitrary.
- **DEC-004 — two classes of randomness.** *Derived keys* (`rng.derive`/`derived`, streams `init_graph`/`task`/`action`/`growth`) are stateless: `key = hash(seed, stream_label, index)`, so "episode 500" is the same task under every ablation regardless of how many draws other subsystems burned. *Running streams* (firing, release in `Sim`) are stateful and high-volume. Sign matters: stream labels are fixed constants, **not** `@intFromEnum` — reordering the enum must not change any stream.
- **DEC-005 — episode-boundary reset table.** `Sim.resetEpisode` / `Neurons.resetFast` clear *fast* state (membrane, refractory, event queue, traces) and preserve *slow* state (weights, thresholds, rate EMA, running RNG streams). Adaptation reset is deliberately configurable. Getting this wrong produces fake "learning" from cross-episode leakage.

### Reproducibility is the central constraint

Two runs with the same `master_seed` must produce byte-identical spike history. This is enforced structurally, and any change that breaks it is a bug:

- **Structure-of-arrays + CSR adjacency, ascending-ID traversal.** Synapses are sorted by source; `out_start[i]..out_start[i+1]` is neuron `i`'s outgoing slice. The RNG draw sequence depends only on stable array order, never on hash-map iteration.
- **`EventQueue` accumulates `f32` currents into future buckets** rather than queueing event records — float addition isn't associative, so fixing the accumulation order (by traversal order) removes a reproducibility hazard.
- **The PRNG is vendored on purpose.** Naming `std.Random.DefaultPrng` isn't enough: a stdlib change would silently alter the stream. If you ever change `rng.zig`'s algorithm, bump `prng_impl_version` (it's written into `run_meta.json`).
- **`Sim.step()` ordering is normative** (spec §9: deliver → external → background → membrane/adaptation/refractory/fire → schedule → homeostasis). Reordering changes dynamics. Phase-N stages (workspace broadcast, traces/eligibility, competition) are intentionally absent and marked in place.

### Tests are the Phase-1 completion checklist

The tests in `sim.zig` and `net.zig` aren't incidental coverage — they encode the spec's exit criteria (activity neither dies nor explodes at defaults; inhibition manipulation visibly changes dynamics; release probability is statistically correct; same seed → identical history; episode reset preserves weights). Treat a failure in these as "the model is wrong" rather than "the test is flaky," and keep them passing when changing dynamics.

## Conventions

- **Sign lives in the presynaptic neuron's type, never in the weight.** `weight >= 0` always; the effect's sign is `NeuronKind.sign()`. Validation rejects negative weights.
- Allocator is threaded explicitly (`gpa`); every `init` has a matching `deinit`, and `errdefer` cleanup is paired at each allocation site.
- Uses the Zig 0.16 `std.Io` API (`std.Io.Writer`, `createFileAtomic`, `init: std.process.Init` in `main`) — not the older `std.fs`/`std.io` shapes.

## ZIG 

- This project is written in zig version 0.16.0
- This is a fairly new version so if you need to know specific zig api calls or standard library functions/structs please use context7
