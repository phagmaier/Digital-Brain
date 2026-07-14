## Stage 1 progress

### ✅ Item 1 — Controller-ablation matrix (report §1, the "immediate priority")
Separated the two conflated controller effects (`Controller` struct) and drove five reported conditions. The result dismantles the confound exactly as the report predicted:

| condition                           | held-acc  | gain vs lookup | stable-term |
| ----------------------------------- | --------- | -------------- | ----------- |
| `full` (original)                   | 1.000     | +0.889         | 1.000       |
| `learned_readout` (honest verdict)  | **0.118** | +0.007         | 0.000       |
| `frozen_controller` (untrained net) | 1.000     | +0.889         | 1.000       |
| `controller_only` (no SNN)          | 1.000     | +0.889         | 0.000       |

An **untrained** network + controller still scores 1.000; the learned readout alone is at the 1/9 prior. The verdict now **FAILs by design** on the honest condition. `controller_only` doubles as Stage-1 item 4's finite-state control for this task.

### ✅ Item 2 — Strengthened consolidation protocol (report §5)
Reworked `continual.zig` with all four requested improvements, plus a new config knob (`consolidation_use_centered_reward`) in `config.zig`/`sim.zig`:

- **Mastery gate** (0.90 rolling acc) before block B; **3/20 non-mastered seeds excluded** from the verdict
- **20 paired seeds** with 95% CIs; verdict judged on the CI **lower bound**
- **Causal pathway lesion**: zeroing `input_a→action_0` collapses raw-consolidation retest by **0.943 ± 0.112** (vs 0.183 off) — the consolidated pathway is causally load-bearing, not just correlated
- **Raw-vs-centered** comparison: centered produces ~0 fully-consolidated synapses but is retest-indistinguishable (both at ceiling) — an honest nuance I recorded

Verdict: **PASS** (survival 0.951 ± 0.045; less-forgetting 0.282 ± 0.119).

### ✅ Item 3 — Expand workspace to 20 paired seeds + CIs
Expanded `workspace.zig` from 4 seeds to **20 paired seeds** with the same stats protocol as `continual.zig`:

- Same seed under both `workspace` / `ablated` conditions (paired gap)
- Mean ± 95% normal CI over the seed sample
- Verdict judged on the CI **lower bound** for both faces of the criterion

Measured result (`zig build workspace -Doptimize=ReleaseFast`):

| metric | mean ± 95% CI half | lower bound | threshold |
| ------ | ------------------ | ----------- | --------- |
| workspace accuracy | **0.724 ± 0.072** | 0.652 | ≥ 0.65 |
| ablated accuracy | 0.530 ± 0.022 | — | (near chance) |
| paired causal gain | **0.194 ± 0.072** | 0.122 | ≥ 0.10 |
| delay workspace state (on / off) | 0.957 ± 0.018 / 0.000 | — | — |

Verdict: **PASS**. Docs updated (`findings.md` Phase 7, `AGENTS.md` P7).

### ✅ Item 5 — Instrumentation: cost, sparsity, forgetting, distribution-shift
New harness `src/instrument.zig` / `zig build instrument` (8 seeds, ReleaseFast):

| track | artefact | headline (mean ± 95% CI) |
| ----- | -------- | ------------------------ |
| Online-update cost | `instrument_cost.csv` | local/dense ops ratio **0.336 ± 0.007** (256 plastic vs ~1049 live) |
| Sparsity | same | firing rate **0.098 ± 0.008**; active (≥½ target) **0.96 ± 0.03** |
| Forgetting curves | `instrument_forgetting.csv` | A-retest after B-disuse: cons ON **1.000**, OFF **0.816 ± 0.167**, gap **0.184 ± 0.167** |
| Distribution shift | `instrument_shift.csv` | pre **0.930** → drop **0.590** → post **0.935**; re-adapts after A↔B flip |
| Lesion resistance | *(continual.zig)* | not re-run; Phase 6 pathway lesion remains the causal probe |

Plot: `uv run scripts/plot_instrument.py` → `instrument.png`. Docs: `findings.md` Stage 1 section, `AGENTS.md` commands/architecture.

Note on forgetting config: both arms keep `consolidation_enabled=true` (so plastic edges join the slow prune clock, DEC-012); OFF only zeros `consolidation_lr` — matching `continual.zig`.

### Remaining Stage 1
- **#4b** Small BPTT RNN baseline (large new implementation)
- *(#4a finite-state control for arithmetic already covered by `controller_only`)*

I did not commit. Want me to continue with #4b, or commit this batch first?
