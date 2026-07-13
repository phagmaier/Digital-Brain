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
result is clean and stark — consolidated survival ~0.97, tentative survival ~0.00.
Trying instead to measure survival by *weight* was noisy and misleading: a
functionally-intact pathway (retest accuracy 1.0) can sit at a tiny absolute weight
(0.08 is enough to win a readout), so weight magnitude hides the survival signal
that the alive/pruned distinction shows plainly.

**The clean control is the reward term, not the whole mechanism.** To show
consolidation is *what* preserves the pathway, the OFF condition keeps the entire
forgetting apparatus (plastic edges still decay and prune) and only zeroes
`consolidation_lr`. Then the two conditions share identical decay dynamics and
differ solely in whether reward consolidates permanence — so the A-retest gap
(on 1.00 vs off 0.86) is attributable to consolidation alone, not to some
difference in how aggressively the two conditions forget.

## Phase 7 — workspace broadcast

**A capacity-one workspace produces a causal long-delay benefit, not just an
extra activity trace.** The controlled `workspace` harness removes the Phase 4
self-exciting memory assembly, holds the stimulus over a 40-step delay, and
changes only `workspace_enabled`. Across seeds 1–4, the enabled condition reached
**0.705** mean accuracy versus **0.491** for the otherwise-identical ablation: a
**+0.214** causal gain (criterion: enabled ≥ 0.65 and gain ≥ 0.10). The ablated
workspace state was exactly 0 by construction; the enabled delay state averaged
~0.94–0.99. This is the important result: a bottlenecked winner can retain and
broadcast task-relevant state after the stimulus is gone.

**The workspace is intentionally a competition, not a second unconstrained
reservoir.** Candidate evidence is collected from the task assemblies, an
ignition threshold admits a winner, `workspace_capacity = 1` bounds access, and
decay prevents it becoming a permanent latch. The weak feedback/broadcast is
then a common bias plus winner-specific feedback. When editing it, preserve the
on/off ablation as a one-flag comparison; adding a second persistence mechanism
would make the causal interpretation ambiguous.

**Mean performance hides seed variance, so retain both the per-seed CSV and the
mean/gap criterion.** The enabled seeds were 0.484–1.000 while the ablated
condition stayed near chance (0.472–0.532). The pass criterion therefore uses a
multi-seed mean and an explicit ablation gap, not a claim that every seed must
solve the task perfectly.

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

**The held-out result decisively exceeds exact-pair memorization on the stated
split.** Training excludes every nonzero addition with result four
(`1+3`, `2+2`, `3+1`) while retaining the needed unit transitions and examples
that produce answer four. Frozen evaluation over 4 seeds × 240 held trials
reported **1.000** mean accuracy. An exact ordered-pair lookup has no entry for
any held example and its fixed 9-answer prior is **1/9 = 0.111**; the measured
gain was **+0.889** (criteria: held accuracy ≥ 0.30 and gain ≥ 0.18). This is a
controlled composition result, not a random train/test split whose answers could
be memorized.

**Fixed-duration readout is an important boundary condition.** Each answer is
read from a preconfigured final window rather than a learned stop signal. During
curriculum training a local teaching current makes the rewarded answer assembly
co-active with the symbols so existing pre×post eligibility can tag the plastic
readout. At evaluation, weights and the transition model are frozen; the neutral
probe and the composed state vote select the answer. Do not turn this into a
variable-duration protocol without defining a new termination/control
experiment.

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
  `delay`/`grow`/`continual`/`workspace`/`arithmetic`), each with tunable
  top-level constants and a PASS/FAIL verdict. They emit CSV artefacts; plotting
  scripts currently cover raster, homeostasis, learning, delay, structural, and
  continual experiments. Run compute-heavy harnesses with
  `-Doptimize=ReleaseFast`.

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
- Phase 8 proves bounded transition composition against an exact-pair baseline,
  not open-ended arithmetic or an emergent reservoir algorithm. Stronger claims
  need larger ranges, independently held results/operators, and ablations of the
  transition controller and teaching current.
