## Stage 1 — COMPLETE

### ✅ Item 1 — Controller-ablation matrix (report §1)
Five conditions; honest `learned_readout` FAILs by design; `controller_only` = finite-state control.

### ✅ Item 2 — Strengthened consolidation protocol (report §5)
Mastery gate, 20 paired seeds + CIs, pathway lesion, raw-vs-centered. Verdict **PASS**.

### ✅ Item 3 — Workspace 20 paired seeds + CIs
Accuracy 0.724 ± 0.072, gain 0.194 ± 0.072. Verdict **PASS**.

### ✅ Item 5 — Instrumentation
`zig build instrument`: cost ratio 0.336, sparsity, forgetting curves, distribution shift.

### ✅ Item 4 — Conventional baselines
- **#4a** Finite-state/tabular for arithmetic: `controller_only` + pair-lookup (already in arithmetic harness).
- **#4b** External baselines: `uv run scripts/baselines.py`
  - tabular / ESN / BPTT Elman RNN on two-choice immediate, delay-20, B-disuse forgetting, label-overwrite, mapping-flip shift
  - artefacts: `baseline.csv`, `baseline_curves.csv`, `baseline.meta.json`, `baseline.png`
  - Headline: BPTT and ESN beat/match the SNN on raw accuracy and shift recovery; SNN niches are **sparse spiking activity**, **local update cost (~58k vs BPTT ~133k accounted ops)**, and **structural forgetting/consolidation** (B-disuse can prune unused plastic pathways; orthogonal-input BPTT does not forget under pure disuse but fully overwrites on same-input label flip).

---

## Stage 2 — COMPLETE (as runnable experiment; scientific criterion open)

`zig build recurrent -Doptimize=ReleaseFast` — context-XOR with readout / structural / recurrent / consol / lesion + representation probe. Honest FAIL of the causal-advantage bars is expected until recurrent credit assignment clears them (see `findings.md`).

---

## Stage 3 — Mechanism-science tracks (report.md / final.md Track A–D)

### ✅ Track A (this slice) — Firing×release factorial + forced exploration + WTA credit
- Config: `stochastic_firing`, `stochastic_release` (default true = Phase 1).
- Det firing: hard `u >= threshold`. Det release: mean-preserving `w * p_release`.
- `Sim.maskEligibilityToTargets` for winner-take-all credit.
- Harness: `zig build stochastic -Doptimize=ReleaseFast` → `stochastic.csv`.

### ☐ Remaining Stage 3 tracks
1. Working-memory phase diagram (excitation × homeostasis × inhibition × delay × refractory × size)
2. Structural growth controls (local vs global, coactivity-biased, reward/error-biased; transfer + lesions)
3. Workspace stress (distractors, replacement, ordered recall, capacity 1/2/4)
4. Robustness (neuron/synapse lesions, input noise, parameter drift, post-train distribution shift)
