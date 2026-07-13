# Findings — Phases 0–4

Observations worth remembering from building the brain-inspired local-learning
system. Not a spec (see the Obsidian note for that) and not a changelog (see git
+ the DEC comments in code) — just the non-obvious things we learned by running
it. Numbers are from the default 100-neuron network unless noted; every result
here is reproducible (same seed → same numbers).

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

## Cross-cutting engineering notes

- **Every phase's mechanism is off by default.** The Phase 1 baseline run is
  byte-identical across all four phases of development — new machinery only
  activates when its config flag is set. This is what keeps the determinism guard
  and the Phase 1 regression baseline meaningful as the system grows.
- **The tests *are* the exit criteria.** Each phase's exit criterion is encoded as
  a deterministic test (e.g. "learns above chance across 4 seeds × 1200 episodes").
  Because the sim is deterministic, these aren't flaky — the accuracies are exact,
  so the bounds sit safely below observed values but far above chance.
- **Experiment harnesses are separate executables** (`sweep`/`perturb`/`train`/
  `delay`), each with tunable constants at the top and a PASS/FAIL verdict, and
  each emitting a CSV that a `uv`-run Python script turns into a figure. Run them
  with `-Doptimize=ReleaseFast`.

## Open threads for later phases

- Faster learning via better credit assignment (see Phase 3 note) — only if we
  care about episodes-to-criterion.
- A true attractor working memory (bistability) if delays need to be effectively
  unbounded (see Phase 4 note).
- The recurrent reservoir's *weights* are still never trained — Phase 5 changes
  its *topology* (grow/prune) but grown edges are non-plastic and fixed-weight.
  Making structural edges eligible (reward-gated permanence, the `permanence_reward_lr`
  term we left at 0) is the natural next lever if reservoir credit assignment
  matters.
- Phase 5's pruning is disuse-driven and therefore quiet in a homeostatic network.
  If a later phase wants *visible* turnover (e.g. to study forgetting), the lever
  is an activity-biased or error-biased growth heuristic (§8.8 #2/#4) plus a
  co-activity signal measured during a quiet baseline, not the stimulus-inflated
  rate EMA we read at the growth window.
