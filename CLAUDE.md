# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A reproducible simulator of fixed-graph stochastic spiking neurons, written in Zig (0.16.0, per `build.zig.zon` `minimum_zig_version`). Phases 1–3 are complete:

- **P1 — fixed recurrent spiking**: 100 E/I neurons, leaky membrane, refractory, stochastic firing/release, delayed delivery.
- **P2 — homeostasis**: per-neuron firing-rate estimate (`rate_ema`), adaptive thresholds, optional synaptic scaling (DEC-007). The update is factored into `Sim.applyHomeostasis()`: the continuous simulator calls it every step (`homeostasis_per_step = true`); the training driver sets that false and calls it once per episode (the doc's "Update homeostasis" cadence). `threshold_max` defaults to 12 — a stability rail, not a tuning knob.
- **P3 — local reward learning**: pre/post traces, per-synapse eligibility traces, and a three-factor reward-modulated weight update (DEC-009), learning a two-choice immediate association (DEC-008 task) above chance across seeds.

Every Phase 2/3 mechanism is **off by default** so the Phase 1 baseline run is unchanged; they are enabled via config (`configs/homeostasis.json`) or the experiment harnesses (`zig build perturb`, `zig build train`). **No structural growth, no workspace** — those later phases are scaffolded (fields allocated, config knobs present, code paths stubbed with `// Phase N` comments) but deliberately inert.

## Commands

```sh
zig build              # compile to zig-out/bin/brain (and zig-out/bin/sweep)
zig build run          # build + run; writes CSV artefacts to cwd, prints run summary
zig build run -- cfg.json   # run with a JSON config (bare Config, or a run_meta.json to replay)
zig build test         # run all tests (both test executables, in parallel)
zig build sweep        # run the parameter-sweep harness -> sweep.csv
zig build perturb      # Phase 2 homeostasis perturbation experiment -> perturb.csv
zig build train        # Phase 3 two-choice association training -> train.csv
zig build run -- configs/homeostasis.json   # a run with the threshold homeostat on

./scripts/check-determinism.sh [cfg.json]   # runs the binary twice, fails if artefacts differ
uv run scripts/plot_raster.py               # render raster.png from raster.csv + metrics.csv
uv run scripts/plot_homeostasis.py          # render perturb.png from perturb.csv
uv run scripts/plot_learning.py             # render train.png from train.csv
```

The experiment harnesses (`sweep`, `perturb`, `train`) are compute-heavy; run them with `-Doptimize=ReleaseFast` (e.g. `zig build train -Doptimize=ReleaseFast`).

Run a single test by name filter (Zig has no per-file test target here; both test binaries honor the filter):

```sh
zig build test --                              # (no direct name flag via build; use the compiler)
zig test src/sim.zig --test-filter "refractory"   # run one test from one file directly
```

`zig build run` overwrites `raster.csv`, `metrics.csv`, `neurons.csv`, `synapses.csv`, and `run_meta.json` in the working directory (these are committed as sample outputs, not ignored). With no argument the binary uses the default `Config{}` literal in `main.zig`; with a JSON path argument it loads that config (`Config.loadFromFile`, which accepts either a bare Config object or a `run_meta.json` to replay a prior run). Missing JSON fields fall back to struct defaults, so a config file need only list the knobs it overrides.

Three auxiliary executables sit alongside `brain`, each its own build root (added in `build.zig`): `src/sweep.zig` (`zig build sweep`) runs a Cartesian grid of configs → one `Summary` row per run in `sweep.csv`; `src/perturb.zig` (`zig build perturb`) runs the Phase 2 exit-criterion experiment — an A/B of homeostasis ON vs OFF under an identical sustained perturbation — → `perturb.csv`; `src/train.zig` (`zig build train`) runs the Phase 3 exit-criterion experiment — the two-choice association training loop across seeds — → `train.csv`. Each prints a PASS/FAIL verdict; edit the experiment constants at the top of each file. The `scripts/` helpers use `uv`'s inline PEP-723 dependencies (no venv setup): `check-determinism.sh` (Phase-0 guard, byte-compares two runs), `plot_raster.py` (→ `raster.png`), `plot_homeostasis.py` (→ `perturb.png`), and `plot_learning.py` (→ `train.png`). Example configs live in `configs/`.

## Architecture

Six source files under `src/`, all pure Zig, no dependencies:

- **`config.zig`** — `Config` (every knob that affects a run), `NeuronKind` (E/I + its `sign()`), `ResetRule`, and `RunMetadata` (serialized to JSON alongside every run). `Config.validate()` enforces the hard invariants before any step runs.
- **`rng.zig`** — vendored `xoshiro256++` + `splitmix64` and the stateless key-derivation scheme. See DEC-004 below.
- **`net.zig`** — `Neurons` and `Synapses` (structure-of-arrays), `Network.build()` which constructs the fixed sparse graph.
- **`sim.zig`** — `EventQueue` (delay ring buffer), `StepMetrics`, and `Sim` with the normative `step()` ordering, plus `Sim.applyHomeostasis()` (per-step-or-per-episode homeostatic update), `Sim.updateEligibility()` (pre/post traces + eligibility, called from `step()` when plasticity is on), and `Sim.applyReward()` (the three-factor weight update, called once per episode).
- **`task.zig`** — the Phase 3 immediate-association task (DEC-008): the four-group `Layout` (input_a/input_b/action_0/action_1 carved from low excitatory IDs), stimulus injection, and the correct mapping. Dependency-light (config only) so `net.zig` can use it at build time without an import cycle.
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
- **DEC-007 — synaptic scaling is homeostatic, not Hebbian.** The optional weight-normalization homeostat (`weight_normalization_enabled`) scales *excitatory* inputs multiplicatively by the *postsynaptic* neuron's rate error (`w *= 1 + eta_w*(target - rho_j)`). It is global and activity-driven — it touches no eligibility trace and leaves the `plastic[]` flag alone (Hebbian plasticity is Phase 3). Inhibitory synapses are left untouched (scaling them by the same rule would be the wrong sign). It runs in stable CSR order with no RNG, so it does not affect reproducibility.
- **DEC-008 — the task is a plastic readout on a fixed reservoir.** When `task_enabled`, `net.zig` adds all-to-all **plastic** input→action synapses on top of the random recurrent graph; the random reservoir stays fixed (never plastic). So learning is a trainable readout over a fixed spiking reservoir — reliable across seeds, and the credit assignment is obvious. The task synapses draw **no RNG** and are interleaved in source order during construction, so (a) CSR still falls out with no sort and (b) the reservoir's RNG stream is byte-for-byte identical to a non-task build (the `net: enabling the task does not perturb the reservoir` test guards this). Groups are carved deterministically from low excitatory IDs (`task.zig`); the correct mapping A→0, B→1 is fixed, and the initial input→action weights are symmetric, so there is no trivial solution — the network must break symmetry from reward alone.
- **DEC-009 — learning is a three-factor rule: pre × post × reward.** Pre/post traces (decay + bump on spike) feed a per-synapse eligibility trace that tags recent pre→post coincidences (`updateEligibility()` in `step()`, LTP-only by default, causal pre-before-post since delays ≥ 1). The eligibility bridges the within-episode coincidence to the end-of-episode reward; `applyReward(R)` then does `w += eta * (R - baseline) * eligibility` on plastic synapses only. The **reward baseline** (an EMA of reward) makes the update zero-mean as accuracy rises, which stops plastic weights from drifting/saturating and materially improves cross-seed reliability. Eligibility is per-episode fast state (reset in `resetEpisode`); the reward baseline is slow state (persists). All deterministic, stable traversal, no RNG — reproducibility holds. Weights only change during stepping when a Phase 2/3 mechanism is explicitly enabled.

### Reproducibility is the central constraint

Two runs with the same `master_seed` must produce byte-identical spike history. This is enforced structurally, and any change that breaks it is a bug:

- **Structure-of-arrays + CSR adjacency, ascending-ID traversal.** Synapses are sorted by source; `out_start[i]..out_start[i+1]` is neuron `i`'s outgoing slice. The RNG draw sequence depends only on stable array order, never on hash-map iteration.
- **`EventQueue` accumulates `f32` currents into future buckets** rather than queueing event records — float addition isn't associative, so fixing the accumulation order (by traversal order) removes a reproducibility hazard.
- **The PRNG is vendored on purpose.** Naming `std.Random.DefaultPrng` isn't enough: a stdlib change would silently alter the stream. If you ever change `rng.zig`'s algorithm, bump `prng_impl_version` (it's written into `run_meta.json`).
- **`Sim.step()` ordering is normative** (spec §9: deliver → external → background → membrane/adaptation/refractory/fire → schedule → homeostasis). Reordering changes dynamics. Phase-N stages (workspace broadcast, traces/eligibility, competition) are intentionally absent and marked in place.

### Tests are the phase completion checklist

The tests in `sim.zig` and `net.zig` aren't incidental coverage — they encode each phase's exit criteria. Phase 1: activity neither dies nor explodes at defaults; inhibition manipulation visibly changes dynamics; release probability is statistically correct; same seed → identical history; episode reset preserves weights. Phase 2: adaptive thresholds pull an over-active network toward target; **the network returns to the target band after a sustained perturbation** (the exit criterion, as an A/B against a no-homeostasis control); synaptic scaling shrinks excitatory inputs to over-active neurons and leaves inhibition alone; both homeostats off ⇒ weights unchanged. Phase 3: eligibility tags a co-active synapse and reward moves its weight by the reward sign; plasticity disabled ⇒ task synapses fixed; **the two-choice association is learned above chance across seeds** (the exit criterion — 4 seeds × 1200 episodes, deterministic, so the accuracies are exact). Treat a failure in these as "the model is wrong" rather than "the test is flaky," and keep them passing when changing dynamics.

## Conventions

- **Sign lives in the presynaptic neuron's type, never in the weight.** `weight >= 0` always; the effect's sign is `NeuronKind.sign()`. Validation rejects negative weights.
- Allocator is threaded explicitly (`gpa`); every `init` has a matching `deinit`, and `errdefer` cleanup is paired at each allocation site.
- Uses the Zig 0.16 `std.Io` API (`std.Io.Writer`, `createFileAtomic`, `init: std.process.Init` in `main`) — not the older `std.fs`/`std.io` shapes.

## ZIG 

- This project is written in zig version 0.16.0
- This is a fairly new version so if you need to know specific zig api calls or standard library functions/structs please use context7

## Additional information if needed

Note that if you are ever confused about terms or about what we are building and the specifications of what we are building there is a very large detailed obsidian note that explains the project and every phase. Note that it is very long and detailed though only query/search inside it when you are confused and need more information. Here is the path to that file: 
/home/phagmaier/Documents/Obsidian/"Digital Brain Project"/"Brain Inspired Local Learning System.md"
