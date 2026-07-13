Both Stage 1 items I've taken on are complete, tested, and documented. Here's the summary.

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

### Bookkeeping
- Docs updated to match (`findings.md` Phase 6 + 8/9, `AGENTS.md` P6/P8/P9, DEC-012/013, test-checklist) so the intentional arithmetic FAIL isn't mistaken for a regression.
- Dynamics baseline **byte-identical** (raster/metrics/neurons/synapses unchanged); only `run_meta.json`'s hash moved from the new config field. Refreshed the golden manifest via `update-golden.sh` and restored its provenance comment. **Golden guard PASS.**
- `zig fmt` clean, full test suite passes.

### Remaining Stage 1 (not started)
- **#3** Expand *workspace* to 20–50 seeds + CIs (continual is already done at 20)
- **#4b** Small BPTT RNN baseline (large new implementation)
- **#5** Instrumentation: online-update cost, sparsity, forgetting curves, distribution-shift

I did not commit — your working tree is staged-clean with the changes above. Want me to continue with one of the remaining items, or commit this batch first?
