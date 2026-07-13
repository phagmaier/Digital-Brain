# 🧠 Brain — A Brain-Inspired Local Learning System

A reproducible, incrementally-built simulator of stochastic spiking neurons with
local, reward-modulated learning — no backpropagation. Built in pure Zig.

**Central question: Can organized computation emerge from local reinforcement,
stochastic exploration, and selective stabilization of connections?**

The answer so far: local rules can adapt and stabilize interfaces around a
designed dynamical substrate, and interacting local mechanisms can produce
useful memory behavior. But organized computation did not spontaneously emerge
in the recurrent network — it was largely scaffolded at the readout and
controller interface. This project explores *which* ingredients earn their
keep, one carefully-controlled mechanism at a time.

---

## What this is (and isn't)

**It is:**
- A research sandbox for local, delayed, structural learning
- A platform for mechanism science — what is necessary, what couples, what fails
- A reproducible experimental apparatus (~6.5k LOC of pure Zig)
- Nine incrementally-tested phases, each with its own experiment harness

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
| 8 | Arithmetic curriculum (transition composition) | Held-out `a+b=4` beats pair-lookup baseline | ✅ |
| 9 | Evidence-triggered termination | Stable unique answer ends episodes; timeout uses distinct penalty | ✅ |

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
zig build train -Doptimize=ReleaseFast       # P3: two-choice association
zig build delay -Doptimize=ReleaseFast       # P4: delayed association
zig build grow -Doptimize=ReleaseFast        # P5: structural rewiring
zig build continual -Doptimize=ReleaseFast   # P6: continual learning
zig build workspace -Doptimize=ReleaseFast   # P7: workspace broadcast
zig build arithmetic -Doptimize=ReleaseFast  # P8/P9: arithmetic curriculum

# Parameter sweeps and perturbations
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
```

### Determinism check

```sh
./scripts/check-determinism.sh         # runs binary twice, fails if artefacts differ
./scripts/check-determinism.sh configs/structural.json
```

---

## Architecture

Core modules under `src/` — pure Zig, no external dependencies:

| Module | Purpose |
|--------|---------|
| `config.zig` | `Config` (all knobs), `NeuronKind` (E/I), validation |
| `rng.zig` | Vendored xoshiro256++ + stateless key derivation (DEC-004) |
| `net.zig` | `Neurons`/`Synapses` (SoA), sparse CSR graph construction |
| `sim.zig` | `EventQueue`, `step()` ordering, plasticity, homeostasis, structural updates |
| `task.zig` | Two-choice association task layout and stimulus injection |
| `log.zig` | CSV writers, `Logger`, `Summary` verdict |
| `main.zig` | Run loop, artefact output, CLI |
| `arithmetic.zig` | Symbolic layout: operand assemblies, transition controller |
| `arithmetic_curriculum.zig` | P8/P9 harness: unit transitions, composition, held-out evaluation |
| `termination.zig` | Stable-answer detector and terminal reward mapping |
| `workspace.zig` | P7 harness: workspace-on/off delayed task ablation |

Experiment roots: `train.zig`, `delay.zig`, `grow.zig`, `continual.zig`, `sweep.zig`,
`perturb.zig`, `workspace.zig`, `arithmetic_curriculum.zig`.

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

The numbered design decisions (DEC-001 through DEC-013) are documented in
[AGENTS.md](AGENTS.md) and are invariants, not preferences.

---

## Key empirical results

- **Working memory emerged from an interaction, not a mechanism.**
  Stimulus-specific persistence required self-excitation × homeostasis × network
  operating point — not recurrent gain alone. A fresh network with self-excitation
  saturates globally; selectivity only appears after homeostatic tuning. This is
  probably the strongest conceptual finding.

- **Fast learning and slow consolidation want different reward signals.**
  Weight updates need baseline-subtracted reward to prevent drift. But
  permanence consolidation needs *raw* reward — using the baseline kills
  consolidation on the seeds that learn fastest. A signal appropriate for
  optimizing a fast variable is not necessarily appropriate for stabilizing a
  slow variable.

- **Three forms of temporal retention were cleanly separated.** Reservoir
  fading memory (~5–10 steps), self-exciting assembly persistence (0.996 at
  delay 20), and capacity-one workspace broadcast (0.705 vs 0.491 ablation at
  delay 40) form distinct delay regimes with a one-flag causal ablation.

- **Learning is exploration-limited, not rate-limited.** A ~600-episode plateau
  then sharp takeoff; tripling the learning rate barely moves it. The bottleneck
  appears to be breaking initial symmetry via stochastic exploration.

- **Structural plasticity pairs with homeostasis.** Rewiring grows from ~790
  to ~940 live edges without destroying task performance or activity stability.
  The regime is growth-dominated — disuse pruning is naturally weak when
  homeostasis keeps most neurons active.

- **Transition composition generalizes beyond memorization, but the composition
  procedure is scaffolded.** Held-out `a+b=4` accuracy 1.000 vs pair-lookup
  prior 0.111. However: the system already contains the algorithm (begin at
  left operand, apply successor/predecessor, repeat); the learned part is the
  unit transitions, not the composition procedure itself.

For detailed findings see [findings.md](findings.md). For a full assessment see
[final.md](final.md). The original specification is in
[Brain Inspired Local Learning System.md](Brain%20Inspired%20Local%20Learning%20System.md).

---

## Generated artefacts

`zig build run` overwrites these in the current directory:

```
raster.csv      metrics.csv     neurons.csv     synapses.csv     run_meta.json
```

All CSV/PNG artefacts are gitignored (regenerable). Config presets live in
`configs/`.

---

## License

All rights reserved. Source available for review and experimentation.
