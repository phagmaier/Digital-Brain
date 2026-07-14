# Findings — Phases 1–8

Observations worth remembering from building the brain-inspired local-learning
system. Not a spec (see the Obsidian note for that) and not a changelog (see git
+ the DEC comments in code) — just the non-obvious things we learned by running
it. Numbers are from the default 100-neuron network unless noted; every result
here is reproducible (same seed → same numbers).

---

## Stage 0 hardening — v1 freeze

**The bounded integer sampler changed intentionally.** The original rejection
limit accepted a domain whose size was generally not divisible by the requested
bound. The practical bias was negligible at `u64` scale, but the implementation
did not satisfy its stated invariant. `Rng.below()` now rejects exactly
`2^64 mod n` words, an exhaustive `u8` reference test proves equal residue
counts for every possible bound, and `prng_impl_version` is now 2 (DEC-004).

**Reproducibility is now guarded both within and across versions.** The repeated-
run determinism check includes `run_meta.json`; `scripts/check-golden.sh` compares
all default artefacts and the human-readable summary against a committed golden.
Every experiment harness also emits a `*.meta.json` protocol sidecar with its
seeds, config templates, thresholds, PRNG version, and Zig version.

**Configuration and allocator failures now fail closed.** Validation rejects
non-finite floats centrally, negative task weights/currents and learning rates,
invalid decay/structural coefficients, overlapping task/arithmetic layouts, and
consolidation without both reward plasticity and a plastic task. Multi-allocation
network initializers clean up every partial construction; exhaustive failing-
allocator tests cover the full `Network.build()` path.

---

## Phase 1 — fixed recurrent spiking

**The default network runs "warm," ~0.14 spikes/neuron/step.** Alive and sparse
enough to pass the smoke test (dead < 0.001, saturated > 0.20), but on the high
side of "sparse." Not a problem — it just means there's plenty of room for the
Phase 2 homeostat to pull the rate down to any target. Worth knowing before
reading too much into the raw default rate.

**The reproducibility architecture holds end-to-end.** Same seed + config →
byte-identical `raster.csv`/`metrics.csv` across separate process runs, verified
by `scripts/check-determinism.sh`. The load-bearing pieces (each a DEC): derived
RNG keys, a vendored PRNG, structure-of-arrays with CSR adjacency traversed in
ascending-ID order, and an event queue that accumulates `f32` current into future
buckets (not event records — float addition isn't associative, so fixing the
accumulation order matters). None of this is free; all of it earns its keep.

---

## Phase 2 — homeostasis

**Threshold homeostasis has bounded authority, set by `threshold_max`.** Under a
sustained doubling of background drive, the adaptive threshold saturated at the
default ceiling of 5.0 and the rate pinned at ~0.15 — it *could not* regulate back
to target. Raising the ceiling to 12 fixed it. The lesson we baked into the code:
**a threshold pinned at the ceiling is a signal that the operating point is wrong
(drive too high for the target), not that the rail is too low.** If you ever need
a huge `threshold_max`, lower `background_current` or engage synaptic scaling
instead — threshold and drive are coupled, and fighting drive with a runaway
threshold treats the symptom.

**Set-point regulation is real and clean.** With homeostasis on, a network
settles to `target_rate`; hit it with a sustained perturbation and it climbs back
into the band while a no-homeostasis control stays pinned above it (0.05 → 0.16
peak → 0.05 recovered, vs control stuck at 0.25). The mean-threshold trace ramping
up to reject the perturbation is the controller's output made visible.

**Cadence matters and is now a knob.** The training loop wants homeostasis updated
*per episode*, not per timestep. We factored the update into `Sim.applyHomeostasis()`
with a `homeostasis_per_step` flag — the continuous simulator self-regulates every
step; the episode driver sets the flag false and calls it once per episode. This
seam, built in Phase 2, is exactly what Phases 3–4 needed.

---

## Phase 3 — local reward learning

**Learning is exploration-limited, not rate-limited.** The learning curve is a
flat ~600-episode plateau near chance, then a sharp rise to ceiling — and raising
the learning rate 3× barely moved the takeoff point (and slightly *lowered* final
accuracy). The bottleneck is breaking the initial symmetry via stochastic
exploration, not the step size. If we ever want faster learning, the lever is
better credit assignment (winner-take-all / lateral inhibition between action
groups, or crediting only the chosen action), not a bigger `learning_rate`.

**The reward baseline is load-bearing for cross-seed reliability.** Subtracting a
running EMA of reward (REINFORCE-with-baseline) makes the weight update zero-mean
as accuracy rises, which stops the plastic weights from drifting/saturating once
the task is solved. Without it, learning is far less reliable across seeds. Cheap
trick, big effect.

**Plastic-readout-on-a-fixed-reservoir is the reliable architecture (a liquid
state machine).** We keep the random recurrent reservoir *fixed* and make only the
input→action synapses plastic. Result: 8/8 seeds reach 100% on the two-choice
task, with obvious credit assignment. Training the whole recurrent net would be a
much harder, less reliable problem — and isn't needed for immediate association.

**Adding task structure doesn't perturb the reservoir.** Task synapses draw no RNG
and are interleaved in source order at build time, so the reservoir's RNG stream
(and therefore its exact wiring/weights) is byte-identical with or without the
task. This is what lets the Phase 1/2 reservoir stay a fixed, reproducible
substrate under Phase 3+.

---

## Phase 4 — delayed learning / working memory

**There are two retention mechanisms, and separating them is the real result.**
1. The fixed reservoir has its own **fading memory** (echo-state property): it
   retains a stimulus for ~5–10 steps for free, then decays to chance by ~delay 20.
2. A **self-exciting input assembly** (working memory) holds the stimulus across
   *long* delays.

   The accuracy-vs-delay curves for the two conditions diverge exactly where the
   reservoir runs out — at delay 20, working memory sits at **0.996** while the
   reservoir-only control has decayed to **0.565** (near chance). So the memory
   mechanism is specifically what buys you *long*-delay retention; short delays
   are handled by the substrate alone.

**Stimulus-specific persistence is emergent from self-excitation *plus*
homeostasis — not from self-excitation alone.** This surprised us and is worth
internalizing: a *fresh* network with self-excitation turned on is globally
saturated — kick assembly A and assembly B lights up too (both fire ~equally),
because reservoir feedback plus recurrent gain drives everything. Specificity (the
*other* assembly staying silent) only appears after the homeostatic threshold
tuning that training installs finds the per-neuron operating point where "stimulus
+ self-excitation" crosses threshold but "reservoir echo alone" does not. Working
memory here is a **system property** of self-excitation + homeostasis + learning,
not a single knob. (This is also why Phase 4's mechanism test is behavioural/end-
to-end rather than a fresh-network probe — the fresh-network probe fails for a
*real* reason, and we deleted it rather than paper over it.)

**The working memory is a decaying persistent state, not a perfect attractor.**
The recurrent-state trace shows the stimulated assembly firing in bursts through
the whole trial but with a gradually shrinking envelope (own-assembly rate ~0.66
early → ~0.15 by the end of a 40-step delay), while the *other* assembly sits at
essentially zero throughout. It holds long enough for these delays but would
eventually fade — arguably more biologically honest than a perfect latch. A true
fixed-point attractor would need more tuning or explicit bistability.

**The persisting assembly oscillates (~2–3 step period).** Visible as bursts in
the retention trace — a consequence of the refractory period (2 steps) combined
with the delay ≥ 1 on the self-excitation, not a bug.

---

## Phase 5 — structural plasticity

**A padded CSR makes growth free.** The whole reproducibility architecture rests
on a fixed, source-sorted CSR adjacency — which naïvely fights adding and removing
edges. The fix is to *never resize it*: over-allocate each source neuron a fixed
slot budget (`max_out_degree`), pack its live edges at the front, and leave the
rest as dead free slots. Pruning frees a slot; growth fills a free slot **inside
the same neuron's range**. `out_start` never moves, traversal stays in stable
array order, and the connection budget falls out for free (it's the slot count).
With the feature off, capacity = live count, so the build is byte-identical to
Phase 4. This one layout decision is what let Phase 5 be an *addition*, not a
rewrite — and the reservoir's RNG stream is provably unperturbed (padding draws
no RNG), the same guarantee the task and self-excitation already had.

**In a homeostatic, task-active network, disuse-pruning is *rare* — and that's
correct, not a bug.** Our first instinct was "connections change" ⇒ lots of
churn. But homeostasis keeps essentially every neuron near `target_rate`, so
essentially every synapse is "used," its permanence pins high, and it is *not*
prunable. Aggressive pruning would delete useful connectivity. So Phase 5's real
regime is **growth-dominated exploration with a trickle of pruning**: the graph
enriches (here ~790 → ~940 live reservoir edges, ~20% more, different topology)
while consolidation protects what works. This is the tentative/established/
consolidated story (§8.3) playing out honestly.

**Accretion needs a set-point, or it fills the budget.** Left alone, growth just
climbs to the hard slot cap and stops — a denser fixed graph, not turnover. The
doc's own connection-budget list has the fix: a **target out-degree** (§8.7).
Growth stops adding to a neuron once it hits the target; pruning frees room and
growth refills it, so the live population plateaus around a set point instead of
accreting. Same trick as the homeostatic *rate* set-point, one level up — at the
level of *structure* rather than *activity*.

**The exit criterion is a three-way AND, and stability is relative.** "Connections
change while activity remains stable and performance is not destroyed" only means
something as a conjunction, so the `grow` harness ANDs all three: churn > 0,
firing rate in a healthy band, and task accuracy still above chance (vs a
structural-off control). One subtlety we got wrong first: the *absolute* firing
rate peaks ~0.2 during stimulus injection — with **or without** structural
plasticity. An absolute ceiling was testing the task, not the mechanism. The
honest stability test is **relative**: rewiring must not push the peak materially
above the no-rewiring control's peak (and must not go dead). With that, all four
seeds pass at 100% accuracy — identical to the control — while the reservoir
rewires underneath. Homeostasis is doing the real work of keeping it stable; the
pairing (rewire + regulate) *is* the result.

## Phase 6 — consolidation

**Consolidation must use RAW reward, not the baseline-subtracted modulator — this
was the whole ballgame.** The weight update (DEC-009) subtracts a running reward
baseline so it goes zero-mean once the task is solved; that is load-bearing for
weight *stability*. But we first wired the permanence-consolidation term to the
same modulator, and it silently failed on exactly the seeds that learned fastest:
once mastered, baseline → +1, modulator → 0, so nothing consolidated (one seed
finished block A with *zero* consolidated synapses despite 100% accuracy). The fix
is what §8.4 actually says — `η_q·max(0, r·e)` uses the reward `r`, not `r −
baseline`. Raw reward keeps consolidating a correct pathway for as long as it stays
correct. The lesson: the baseline that makes *learning* stable is poison for
*consolidation*; they want opposite signals from the same reward.

**Directly measured, the raw-vs-centered story is subtler than "centered fails"
(report.md §5).** The harness now runs a third `centered` condition
(`consolidation_use_centered_reward = true`) alongside `raw` and `off`, so the two
consolidation signals are compared head-to-head rather than argued from first
principles. The structural signature is exactly as predicted — centered
consolidation produces **~0 fully-consolidated-band synapses** (permanence never
reaches q ≥ 0.6, because the modulator collapses to 0 once mastered) where raw
produces ~150–250. But **A-retest accuracy is statistically identical** (raw −
centered = 0.001 ± 0.001), because *both sit at the 1.000 retest ceiling* on this
two-choice task: the permanence centered accrues *early* in block A, before the
baseline saturates, is already enough to carry the pathway through block B's
disuse. So on this task the raw-vs-centered difference is real in the permanence
bands and invisible at the behavioural readout — the retest metric is
ceiling-limited and cannot separate them. A task with a longer or more
interfering block B, where partial permanence is not enough to survive, is what
would turn the band-level difference into a behavioural one.

**Reward-gated for plastic, activity-gated for reservoir — and the distinction is
principled, not incidental.** §8.3 defines consolidated as "repeatedly associated
with *rewarded behaviour*," and §8.4 warns against consolidating a synapse merely
because it changes/fires a lot. So plastic readout edges consolidate on reward
(they carry eligibility); reservoir edges consolidate on co-activity (they don't).
Concretely, under consolidation the slow loop gives plastic edges disuse-decay
*only* — their positive drive is the reward bump in `applyReward`, never mere
co-activity. Without this split, an actively-firing-but-useless readout synapse
would consolidate, which is exactly the failure the doc flags.

**"Tentative / established / consolidated" really are just permanence bands, and
that makes the exit criterion trivial to measure.** §8.3 says the state "need not
be an enum; it can emerge from thresholds over permanence." Taking that literally:
after block A we tag each plastic synapse consolidated (q ≥ 0.6) or tentative
(q ≤ 0.4), then after block B (disuse) count how many of each are still alive. The
result is clean and stark — over 17 mastered seeds the consolidated − tentative
survival gap is **0.951 ± 0.045** (95% CI, raw consolidation), i.e. consolidated
survival ≈ 0.95 against tentative ≈ 0.00. Trying instead to measure survival by
*weight* was noisy and misleading: a
functionally-intact pathway (retest accuracy 1.0) can sit at a tiny absolute weight
(0.08 is enough to win a readout), so weight magnitude hides the survival signal
that the alive/pruned distinction shows plainly.

**The clean control is the reward term, not the whole mechanism.** To show
consolidation is *what* preserves the pathway, the OFF condition keeps the entire
forgetting apparatus (plastic edges still decay and prune) and only zeroes
`consolidation_lr`. Then the two conditions share identical decay dynamics and
differ solely in whether reward consolidates permanence — so the A-retest gap
(raw − off = **0.282 ± 0.119**, 95% CI over 17 mastered seeds) is attributable to
consolidation alone, not to some difference in how aggressively the two conditions
forget.

**The protocol was hardened per report.md §5, and the causal claim is now a lesion,
not just a survival correlation.** Four changes: (1) block A trains to a *mastery
criterion* (0.90 rolling accuracy) rather than a fixed length, so a weak block-A
fit cannot masquerade as forgetting — the earlier fixed-length run left two seeds
at 0.47/0.74 accuracy; (2) seeds that never master block A within the cap are
reported separately and **excluded from the verdict** (3/20 here: seeds 1, 3, 14);
(3) the sample is **20 paired seeds** with 95% confidence intervals, and the
verdict is judged on the CI *lower bound*, not the point estimate; (4) a
**pathway-specific lesion** — after retest, the plastic `input_a → action_0`
readout weights are zeroed and A is retested again. This is the causal test the
survival contrast alone could not give: permanence is *definitionally* what
resists pruning, so "high-permanence synapses survive" is partly built into the
rule. The lesion asks the functional question directly — is the retained A
behaviour actually carried by that specific consolidated pathway? It is: zeroing
it drops raw-consolidation retest by **0.943 ± 0.112** (retest ≈ 0.94 → ≈ 0.00),
while under OFF the same lesion costs only 0.183 because the pathway had already
decayed and the residual A-retest was near chance. Consolidation does not merely
correlate with a surviving band; it builds the pathway the behaviour depends on.

## Phase 7 — workspace broadcast

**A capacity-one workspace produces a causal long-delay benefit, not just an
extra activity trace.** The controlled `workspace` harness removes the Phase 4
self-exciting memory assembly, holds the stimulus over a 40-step delay, and
changes only `workspace_enabled`. Across **20 paired seeds** (Stage 1 / report.md),
the enabled condition reached **0.724 ± 0.072** mean accuracy versus **0.530 ± 0.022**
for the otherwise-identical ablation: a **+0.194 ± 0.072** paired causal gain
(95% normal CIs; criterion judged on the CI *lower* bound: enabled ≥ 0.65 and
gain ≥ 0.10 — both clear). The ablated workspace state was exactly 0 by
construction; the enabled delay state averaged **0.957 ± 0.018**. This is the
important result: a bottlenecked winner can retain and broadcast task-relevant
state after the stimulus is gone.

**The workspace is intentionally a competition, not a second unconstrained
reservoir.** Candidate evidence is collected from the task assemblies, an
ignition threshold admits a winner, `workspace_capacity = 1` bounds access, and
decay prevents it becoming a permanent latch. The weak feedback/broadcast is
then a common bias plus winner-specific feedback. When editing it, preserve the
on/off ablation as a one-flag comparison; adding a second persistence mechanism
would make the causal interpretation ambiguous.

**Mean performance hides seed variance, so retain both the per-seed CSV and the
CI lower-bound criterion.** Enabled seeds still span ~0.48–1.00 while ablated
stays near chance (~0.47–0.62). A few seeds (e.g. 2, 10, 16) show essentially
no gain; the pass criterion therefore uses the multi-seed mean *and* an explicit
paired ablation gap, each judged on the 95% CI lower bound — not a claim that
every seed solves the task perfectly.

## Phase 8 — symbolic arithmetic curriculum

**Ordered symbol binding is required before subtraction can be meaningful.** A
numeral is represented by a position-bound one-hot assembly: left-operand `2`
and right-operand `2` are distinct populations. The serial sequence is
`START, lhs, operator, rhs, END`; separate `+`/`−` assemblies and a bounded set
of non-negative answer assemblies complete the representation. Without the
left/right distinction, `a − b` and `b − a` collapse to the same final input set.
All arithmetic readout edges are deterministic source-ordered symbol→answer
edges, so enabling arithmetic does not consume or perturb reservoir RNG draws.

**The curriculum's generalization mechanism is transition composition, not a
hidden operand-pair table.** Rewarded `n + 1` and `n − 1` trials populate only
successor/predecessor action transitions. One-operation evaluation obtains an
answer by repeatedly applying the appropriate unit transition from `lhs`, once
per `rhs`; the controller has no `(lhs, rhs) → answer` storage. The reached state
then participates in the fixed-duration answer-assembly readout and competition.
This is a deliberately structured controller coupled to spiking action
assemblies — **not** evidence that the unconstrained reservoir spontaneously
invented symbolic arithmetic. Preserve that distinction when interpreting or
extending Phase 8.

## Phase 9 — learned termination

**Termination is controlled by stable evidence, not answer identity.** The
Phase 9 interface observes the same accumulated spike-count evidence used by
the answer reader and terminates only when one uniquely dominant answer persists
for the configured stable window. Silence and ties reset the window, preventing
a low-index tie from becoming an implicit answer. This keeps timing separate
from answer representation while letting the rewarded readout learn to make
reliable answers available early.

**Timeout is a safety rail with a distinct terminal scalar.** A finite maximum
readout duration prevents a non-committing episode from running indefinitely.
Timely correct and incorrect commitments use `+1` and `-1`; timeouts use the
smaller configured penalty (default `-0.2`). The scalar is sent through the
existing eligibility-trace `applyReward()` path; Phase 9 adds no second learning
rule and consumes no RNG, so reproducibility is preserved.

**The held-out result is controller-driven, not learned-readout generalization
— the ablation matrix isolates this.** Training excludes every nonzero addition
with result four (`1+3`, `2+2`, `3+1`) while retaining the needed unit transitions
and examples that produce answer four. The original harness reported **1.000**
mean held-out accuracy, but that number combined three effects that report.md §1
flagged as conflated: the trained spiking readout, an answer-specific current
injected into the composed answer assembly (`transition_action_current = 100`),
and a *direct vote* the controller adds straight into the spike-count readout
(`transition_termination_vote`). The harness now reports the five-condition
ablation matrix (`zig build arithmetic`, 4 seeds × 240 held trials each):

| condition | held-acc | gain vs lookup (0.111) | stable-term |
|---|---|---|---|
| `full` (original) | 1.000 | +0.889 | 1.000 |
| `current_no_vote` (controller current into SNN, spike counts only) | 0.116 | +0.005 | 0.000 |
| `learned_readout` (trained readout, controller removed) | 0.118 | +0.007 | 0.000 |
| `frozen_controller` (untrained readout, full controller) | 1.000 | +0.889 | 1.000 |
| `controller_only` (finite-state `solve`, no SNN) | 1.000 | +0.889 | 0.000 |

The reading is unambiguous. The pure finite-state controller scores 1.000 with
no network at all, and an **untrained** network plus the controller also scores
1.000 — so the composed answer, not any learned spiking readout, produces the
result. The learned readout *on its own* sits at the 1/9 lookup prior (0.118 vs
0.111), and injecting the controller's current into the readout without the direct
vote does not lift it (0.116). "Stable termination" likewise depends entirely on
the direct vote: it is 1.000 with the vote and 0.000 without it, so it is
**evidence-triggered termination over controller-assisted answers, not learned
termination** (report.md §1). The harness verdict is now judged on the honest
condition (`learned_readout`) and therefore **FAILs** by design — this failure is
the corrected scientific claim, not a regression. The genuine, defensible result
is the `controller_only` line: bounded single-operation arithmetic is solved by
composing rewarded unit transitions, beating an exact-pair lookup on the
controlled split. The spiking substrate does not yet carry that computation.

**Fixed-duration readout is an important boundary condition.** Each answer is
read from a preconfigured final window rather than a learned stop signal. During
curriculum training a local teaching current makes the rewarded answer assembly
co-active with the symbols so existing pre×post eligibility can tag the plastic
readout. At evaluation, weights and the transition model are frozen; the neutral
probe and the composed state vote select the answer. Do not turn this into a
variable-duration protocol without defining a new termination/control
experiment.

## Stage 1 instrumentation — cost, sparsity, forgetting, shift

Report.md Stage 1 asked to measure the niches where a local three-factor system
might compete (not only final accuracy). The `instrument` harness
(`zig build instrument -Doptimize=ReleaseFast`) runs four tracks on the
two-choice association over 8 seeds and writes `instrument_cost.csv`,
`instrument_forgetting.csv`, `instrument_shift.csv` (+ `instrument.meta.json`).
Lesion resistance remains the load-bearing probe in `continual.zig` (Phase 6).

**Online-update cost (accounting model).** Eligibility + reward touch the 256
plastic readout edges; a dense baseline touches every live synapse each step
(~1050). Measured local/dense ops ratio: **0.336 ± 0.007**. This is an O(·)
accounting model (not a wall-clock microbenchmark), but it makes the claimed
local-update niche concrete: the three-factor rule is substantially cheaper than
full-graph-per-step credit assignment on this topology.

**Sparsity.** Final-window mean firing rate **0.098 ± 0.008** spikes/neuron/step
(soft alive band [0.005, 0.20]); fraction of neurons at ≥½ target rate
**0.96 ± 0.03**. Activity is sparse relative to a dense continuous code, though
nearly the whole population participates weakly under the task stimulus. Reference
final accuracy under the same protocol: **0.980 ± 0.023**.

**Forgetting curves.** Train A (full task) then force only B, probing frozen A
accuracy every 50 block-B episodes. Consolidation uses the continual protocol
knob (`consolidation_enabled=true` always so plastic edges join the slow clock;
OFF zeros `consolidation_lr` only). End-of-disuse A-retest: consolidation ON
**1.000 ± 0.000**, OFF **0.816 ± 0.167**, gap **0.184 ± 0.167**. Seed variance
under OFF is large (some seeds stay near ceiling, others fall toward chance) — the
time series in `instrument_forgetting.csv` is the artefact, not a single number.

**Distribution-shift adaptation.** Mid-run reward mapping flip (A→0/B→1 →
A→1/B→0) at episode 1000. Pre-shift block accuracy **0.930 ± 0.084**, first
post-shift block **0.590 ± 0.088**, final post-shift block **0.935 ± 0.086**.
Recovery to ≥0.70 typically within 50–350 post-shift episodes (one seed slower at
~700). Online three-factor learning re-adapts after the flip without a separate
optimizer.

**What this does *not* claim.** No external BPTT/ESN baseline is compared here
(Stage 1 #4b). Cost is FLOP-accounting, not wall time. Forgetting is a curve
under B-only disuse, not a full continual-learning survival/lesion battery
(that remains Phase 6).

## Cross-cutting engineering notes

- **Every phase's mechanism is off by default.** The Phase 1 baseline run is
  byte-identical across all eight phases of development — new machinery only
  activates when its config flag is set. This is what keeps the determinism guard
  and the Phase 1 regression baseline meaningful as the system grows.
- **Focused tests and experiment harnesses are the exit criteria.** Unit tests
  guard invariants and mechanism prerequisites; deterministic multi-seed
  harnesses (`train` through `arithmetic`) establish behavioural criteria and
  write the evidence CSV. Neither category is flaky: same seed/config gives the
  same output, so thresholds should sit safely below the observed result. Treat
  a failure as a model regression, not noise to be ignored.
- **Experiment harnesses are separate executables** (`sweep`/`perturb`/`train`/
  `delay`/`grow`/`continual`/`workspace`/`arithmetic`/`instrument`), each with
  tunable top-level constants and a PASS/FAIL (or instrumentation-complete)
  verdict. They emit CSV artefacts; plotting scripts currently cover raster,
  homeostasis, learning, delay, structural, continual, and instrument
  experiments. Run compute-heavy harnesses with `-Doptimize=ReleaseFast`.

## Open threads for later phases

- Faster learning via better credit assignment (see Phase 3 note) — only if we
  care about episodes-to-criterion.
- A true attractor working memory (bistability) if delays need to be effectively
  unbounded (see Phase 4 note).
- The recurrent reservoir's *weights* are still never trained — Phase 5 changes
  its *topology* (grow/prune) but grown edges are non-plastic and fixed-weight.
  Reward-gated permanence exists (Phase 6, DEC-012) but only on the *plastic*
  readout edges; making *reservoir* edges eligible so reward can consolidate them
  too is the natural next lever if reservoir credit assignment matters.
- Phase 5's pruning is disuse-driven and therefore quiet in a homeostatic network.
  If a later phase wants *visible* turnover (e.g. to study forgetting), the lever
  is an activity-biased or error-biased growth heuristic (§8.8 #2/#4) plus a
  co-activity signal measured during a quiet baseline, not the stimulus-inflated
  rate EMA we read at the growth window.
- Phase 7 currently establishes only a capacity-one, two-choice delayed-task
  broadcast result. Multi-item access, interference, and a learned admission
  policy remain open experiments; preserve the one-flag ablation before adding
  any of them.
- Phase 8's controller ablation matrix is now the honest record: bounded
  transition composition (the `controller_only` finite-state baseline) beats an
  exact-pair lookup, but the spiking readout carries none of that computation —
  isolated, it sits at the 1/9 prior, and the frozen-network condition scores
  1.000 on the controller alone. The open task is making the *spiking substrate*
  perform the composition: a plastic recurrent/readout path that generalizes on
  the held split with the controller removed, over larger ranges and
  independently held results/operators. This is the entry point to Stage 2's
  context-dependent delayed task.
