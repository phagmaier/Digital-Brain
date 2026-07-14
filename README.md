# 🧠 Brain — A Brain-Inspired Local Learning System

A reproducible, incrementally-built simulator of stochastic spiking neurons with
local, reward-modulated learning — no backpropagation. Built in pure Zig.

**Central question: Can organized computation emerge from local reinforcement,
stochastic exploration, and selective stabilization of connections?**

The answer so far is partial and honest: local rules can adapt and stabilize
interfaces around a designed dynamical substrate, and interacting local
mechanisms can produce useful memory behavior. But organized computation did
not spontaneously emerge in the recurrent network — it was largely scaffolded at
the readout and controller interface. This project explores *which* ingredients
earn their keep, one carefully-controlled mechanism at a time.

---

## What this is (and isn't)

**It is:**
- A research sandbox for local, delayed, structural learning
- A platform for mechanism science — what is necessary, what couples, what fails
- A reproducible experimental apparatus (~6.5k LOC of pure Zig)
- Nine incrementally-tested phases plus three stages of characterization, each
  with its own experiment harness

**It is not:**
- A brain simulation or claim about biological fidelity
- A novel general learning algorithm (the constituent mechanisms are established;
  the interaction findings are the contribution)
- A system that competes with backpropagation on benchmarks
- Proof that local rules spontaneously invent arithmetic

---

## Phase ladder

Each phase adds one mechanism. All later-phase features are **off by default** —
the Phase 1 baseline remains byte-identical across the entire stack.

| Phase | Mechanism | Exit criterion | Status |
|:-----:|-----------|---------------|:------:|
| 0 | Reproducibility architecture | Same seed → identical spikes | ✅ |
| 1 | Fixed recurrent spiking (100 E/I neurons) | Alive, sparse, not dead or saturated | ✅ |
| 2 | Homeostasis (adaptive thresholds, synaptic scaling) | Return to target rate after perturbation | ✅ |
| 3 | Local reward learning (pre × post × reward eligibility) | Two-choice association above chance, 8/8 seeds | ✅ |
| 4 | Delayed learning (self-exciting working memory assembly) | Retain association across long delay | ✅ |
| 5 | Structural plasticity (grow/prune reservoir edges) | Connections change; task survives; activity stable | ✅ |
| 6 | Consolidation (reward-gated permanence on readout) | Useful pathways survive disuse | ✅ |
| 7 | Workspace broadcast (capacity-limited competition) | Causal long-delay accuracy gain over ablation | ✅ |
| 8 | Arithmetic curriculum (transition composition) | Held-out `a+b=4` beats pair-lookup; BUT: controller-driven, not learned | ⚠️¹ |
| 9 | Evidence-triggered termination | Stable unique answer ends episodes; timeout uses distinct penalty | ✅² |

*¹ The held-out generalization is carried by a finite-state controller, not the spiking readout. The honest `learned_readout` condition FAILs by design — see findings.md.*
*² Termination is evidence-triggered over controller-assisted answers — see findings.md.*

---

## Characterization stages

| Stage | Focus | Key result | Status |
|:-----:|-------|-----------|:------:|
| 1 | Instrumentation & baselines | Cost ratio 0.336 local/dense; BPTT/ESN beat on accuracy; SNN niches: sparsity, structural forgetting | ✅ |
| 2 | Recurrent plasticity (context-XOR) | Local plasticity in reservoir does not yet beat fixed-reservoir readout | ❌³ |
| 3A | Stochasticity factorial + WTA credit | Credit assignment, not exploration, is the bottleneck; WTA halved time-to-mastery | ✅ |

*³ Honest FAIL of the Stage 2 exit criterion — the open problem stands. See findings.md.*

---

## Quick start

**Prerequisites:** [Zig 0.16.0](https://ziglang.org/download/)

```sh
git clone https://github.com/YOUR_USER/brain.git
cd brain

# Build all executables
zig build

# Run the default simulation (writes CSV artefacts to cwd, prints summary)
zig build run

# Run with a config override
zig build run -- configs/structural.json

# Run all tests
zig build test
```

### Experiment harnesses

Each harness is a separate executable. Run compute-heavy ones with `-Doptimize=ReleaseFast`:

```sh
# Phase experiments
zig build train -Doptimize=ReleaseFast       # P3: two-choice association
zig build delay -Doptimize=ReleaseFast       # P4: delayed association
zig build grow -Doptimize=ReleaseFast        # P5: structural rewiring
zig build continual -Doptimize=ReleaseFast   # P6: continual learning + lesion
zig build workspace -Doptimize=ReleaseFast   # P7: workspace broadcast
zig build arithmetic -Doptimize=ReleaseFast  # P8/P9: arithmetic curriculum + termination

# Stage experiments
zig build instrument -Doptimize=ReleaseFast  # Stage 1: cost/sparsity/forgetting/shift
uv run scripts/baselines.py                  # Stage 1: tabular/ESN/BPTT external baselines
zig build recurrent -Doptimize=ReleaseFast   # Stage 2: context-XOR recurrent plasticity
zig build stochastic -Doptimize=ReleaseFast  # Stage 3A: stochasticity factorial + WTA credit

# Parameter sweeps and diagnostics
zig build sweep -Doptimize=ReleaseFast       # grid search
zig build perturb -Doptimize=ReleaseFast     # homeostasis A/B
```

Each harness prints `PASS`/`FAIL` and writes a CSV. Edit its top-level constants to
reconfigure.

### Plotting

Plot scripts use `uv`'s inline PEP-723 dependencies (no venv setup):

```sh
uv run scripts/plot_raster.py          # → raster.png
uv run scripts/plot_learning.py        # learning curves
uv run scripts/plot_delay.py           # delay vs accuracy
uv run scripts/plot_structural.py      # structural change over time
uv run scripts/plot_continual.py       # consolidation survival
uv run scripts/plot_homeostasis.py     # perturbation recovery
uv run scripts/plot_instrument.py      # cost/sparsity/forgetting/shift
uv run scripts/plot_baselines.py       # external baseline comparisons
```

### Determinism check

```sh
./scripts/check-determinism.sh         # runs binary twice, fails if artefacts differ
./scripts/check-determinism.sh configs/structural.json
./scripts/check-golden.sh              # checks default output against the committed baseline
```

---

## Architecture

Core modules under `src/` — pure Zig, no external dependencies:

| Module | Purpose |
|--------|---------|
| `config.zig` | `Config` (all knobs), `NeuronKind` (E/I), validation |
| `rng.zig` | Vendored xoshiro256++ + stateless key derivation (DEC-004) |
| `net.zig` | `Neurons`/`Synapses` (SoA), sparse CSR graph construction, structural/plastic edge flags |
| `sim.zig` | `EventQueue`, `step()` ordering, plasticity, homeostasis, structural updates, WTA credit |
| `task.zig` | Two-choice association task layout and stimulus injection |
| `context_task.zig` | Stage 2 context-XOR layout: six groups, delayed cross-coupling |
| `arithmetic.zig` | Symbolic layout: operand assemblies, transition controller |
| `arithmetic_curriculum.zig` | P8/P9 harness: unit transitions, composition, held-out evaluation, ablation matrix |
| `termination.zig` | Stable-answer detector and terminal reward mapping |
| `log.zig` | CSV writers, `Logger`, `Summary` verdict |
| `main.zig` | Run loop, artefact output, CLI |

Experiment roots: `train.zig`, `delay.zig`, `grow.zig`, `continual.zig`,
`workspace.zig`, `arithmetic_curriculum.zig`, `sweep.zig`, `perturb.zig`,
`instrument.zig`, `recurrent.zig`, `stochastic.zig`.

External baseline (Python): `scripts/baselines.py`.

---

## Key design principles

1. **Reproducibility is a hard constraint.** Two runs with the same `master_seed`
   produce byte-identical spike history. Enforced by vendored PRNG, stateless
   derived key streams (DEC-004), structure-of-arrays with stable CSR traversal,
   and fixed-order event accumulation.

2. **Default-off mechanisms.** Every later-phase mechanism activates only via
   config flags. Phase 1 baseline stays byte-identical as features accumulate.

3. **Tests encode invariants, not just coverage.** Unit tests guard mechanism
   prerequisites; harnesses encode multi-seed behavioural criteria. Neither is
   flaky — thresholds sit safely below observed results.

4. **Honest mechanism science.** Ablations, baselines, and explicit negative
   results. Probes that failed for real reasons were deleted rather than papered
   over (e.g., the fresh-network working memory specificity test).

The numbered design decisions (DEC-001 through DEC-014) are documented in
[AGENTS.md](AGENTS.md) and are invariants, not preferences.

---

## Key empirical results

### Interaction findings (strongest results)

- **Working memory emerged from an interaction, not a mechanism.**
  Stimulus-specific persistence required self-excitation × homeostasis ×
  network operating point — not recurrent gain alone. A fresh network with
  self-excitation saturates globally; selectivity only appears after homeostatic
  tuning. This is probably the strongest conceptual finding.

- **Fast learning and slow consolidation want different reward signals.**
  Weight updates need baseline-subtracted reward to prevent drift. But
  permanence consolidation needs *raw* reward — using the baseline kills
  consolidation on the seeds that learn fastest. A signal appropriate for
  optimizing a fast variable is not necessarily appropriate for stabilizing a
  slow variable.

- **Structural plasticity and homeostasis form a stable coupling.**
  Random structural growth (~790→~940 live edges) did not cause catastrophic
  disruption because homeostasis compensates for changing connectivity.
  Regime is growth-dominated with a trickle of pruning; a target out-degree
  set-point prevents unbounded accretion.

### Capability findings

- **Three forms of temporal retention were cleanly separated.** Reservoir
  fading memory (~5–10 steps), self-exciting assembly persistence (0.996 at
  delay 20), and capacity-one workspace broadcast (0.724 vs 0.530 ablation at
  delay 40) form distinct delay regimes with one-flag causal ablations.

- **Credit assignment, not exploration, is the learning bottleneck.**
  WTA credit + forced exploration halved the episodes-to-mastery (875→500).
  Noise alone doesn't speed learning — the system needs to know *which*
  synapses to credit, not just explore more actions.

- **Consolidation produces causal pathway survival, not just correlation.**
  After training A (to mastery) then B, lesioning the consolidated A pathway
  drops retest by 0.94 vs 0.18 with consolidation off. The pathway the
  behaviour depends on is the one consolidation preserved.

### Honest negatives

- **Recurrent plasticity does not yet beat the fixed reservoir (Stage 2).**
  Local three-factor learning inside the reservoir sits near chance on a
  context-XOR task — the fixed-reservoir readout cannot solve it, but local
  recurrence doesn't yet rescue it. This is the clearest open problem.

- **Arithmetic generalization was scaffolded, not learned (Phase 8).**
  The transition controller composes the answer; the spiking readout alone
  sits at the 1/9 prior. The ablation matrix isolates this cleanly.

- **External baselines beat the SNN on raw accuracy (Stage 1).**
  A 32-unit BPTT RNN and 64-unit ESN solve the same tasks at ceiling. The
  SNN's plausible niches are sparse spiking activity, local update cost
  (~58k vs ~133k accounted ops), and structural forgetting dynamics — not
  sample efficiency or asymptotic accuracy.

### Characterization results (Stage 3A)

- **Firing and release noise are largely redundant** for the two-choice task.
  Fully deterministic firing+release still masters the task (0.95) and
  matches the stochastic ceiling with WTA + exploration, at faster takeoff.
- **WTA credit alone cuts time-to-mastery by ~17%**; combined with forced
  exploration it nearly halves it.

For detailed findings see [findings.md](findings.md). The numbered design
decisions are in [AGENTS.md](AGENTS.md). For a complete project assessment
see [final.md](final.md).

---

## Generated artefacts

`zig build run` overwrites these in the current directory:

```
raster.csv      metrics.csv     neurons.csv     synapses.csv     run_meta.json
```

Experiment harnesses each write their own CSV (e.g. `train.csv`, `continual.csv`,
`stochastic.csv`) plus a `*.meta.json` provenance manifest. All CSV/PNG artefacts
are gitignored (regenerable). Config presets live in `configs/`.

---

## License

All rights reserved. Source available for review and experimentation.
