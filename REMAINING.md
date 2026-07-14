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

## Stage 1 is done. Natural next work (Stage 2+)

From `report.md` / `final.md` (not started):

1. **Stage 2 flagship** — context-dependent delayed task (XOR-style) with recurrent plasticity ablations
2. Mechanism-science tracks (stochasticity factorial, forced exploration, …)
3. Optional: BPTT seq model on the arithmetic held-out split (only if revisiting composition in the substrate)

I did not commit. Say if you want a Stage 1 commit message drafted or a Stage 2 kickoff.
