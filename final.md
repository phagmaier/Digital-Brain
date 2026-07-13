# Honest Assessment — Brain-Inspired Local Learning System

**Date:** 2026-07-13  
**Sources:** project plan (`Brain Inspired Local Learning System.md`), `AGENTS.md`, `findings.md`, experiment CSVs, and the Zig implementation (~6.5k LOC under `src/`).

---

## 1. Executive verdict

**You built something genuinely good and worth taking seriously.** It is far beyond a toy "I made some neurons spike" project. You designed a reproducible experimental platform, introduced mechanisms incrementally, built causal ablations, noticed when your own proposed tests were misleading, and extracted several non-obvious interaction effects.

But the conclusion needs to be sharp:

> **The project is scientifically interesting, but it has not yet demonstrated a novel general learning algorithm.**

Its strongest contribution is currently **a carefully engineered experimental synthesis of known ideas that produced several interesting system-level findings**. Its weakest point is that much of the actual task learning remains concentrated in the readout or in explicitly designed controllers rather than emerging through plasticity inside the recurrent network.

That distinction does not diminish the project. It tells you exactly where the exciting next experiment lies.

**Is it worth continuing?** Yes — but not as "add more mechanisms until it becomes a brain." The platform is already a serious experimental apparatus. The high-value next work is *harder questions with cleaner controls*, not feature accretion. The project already produced a **complete vertical slice** of the original plan — that is a success even as a closed chapter.

---

## 2. What was planned vs what was built

### 2.1 The original research question

> **Can organized computation emerge from local reinforcement, stochastic exploration, and selective stabilization of connections?**

The plan was unusually disciplined: one mechanism at a time, ablations, baselines, and explicit "negative results count." Success was defined as a ladder of increasing capabilities (plan §23), not as beating gradient methods on arithmetic.

The answer the project actually supports is narrower than the original question:

> Local rules can adapt and stabilize interfaces around a designed dynamical substrate, and interacting local mechanisms can produce useful memory behavior. But organized computation did not spontaneously *emerge* in the recurrent network — it was largely scaffolded at the readout and controller interface.

### 2.2 Delivery against the phase ladder

| Phase | Plan exit criterion | Status | Measured evidence (defaults / harnesses) |
|------:|---------------------|--------|------------------------------------------|
| 0 | Same seed → identical results | **Met** | Vendored PRNG, derived streams, CSR order, `check-determinism.sh` |
| 1 | Stable sparse recurrent spiking | **Met** | ~0.14 spikes/neuron/step; alive, not saturated |
| 2 | Return to target after perturbation | **Met** | Homeostasis recovers ~0.05→0.16→0.05; control stuck ~0.25 |
| 3 | Immediate two-choice above chance, multi-seed | **Met** | 8/8 seeds → 100% final accuracy |
| 4 | Delayed association above chance | **Met** | Working memory @ delay 20: **0.996** vs reservoir-only **0.565** |
| 5 | Connections change; activity stable; task survives | **Met** | Live edges ~790→~940; accuracy still 100% with rewiring |
| 6 | Useful pathways survive disuse better | **Met** | Consolidated survival ~**0.97** vs tentative ~**0.00** |
| 7 | Workspace causally helps a delayed task | **Met** | Mean **0.705** vs ablated **0.491** at delay 40 (+0.214) |
| 8 | Arithmetic beats memorization on a controlled split | **Met** | Held-out `a+b=4`: **1.000** vs pair-prior **0.111** (+0.889) |
| 9 | Evidence-triggered adaptive termination | **Met** | Stable unique answer ends episodes; timeout uses distinct −0.2 |

### 2.3 What the plan asked for that was *not* done

These gaps matter for honesty, not as failures of the completed ladder:

| Planned item | Status | Comment |
|--------------|--------|---------|
| Stage D — context-dependent mapping | **Not done** | Would stress multi-signal integration beyond two-choice |
| Stage G — sequential multi-op arithmetic | **Not done** | Explicitly deferred; still the next hard curriculum step |
| Conventional baselines (MLP / RNN / equal params) | **Not done** | Only internal ablations and a pair-lookup arithmetic baseline. This is the largest scientific gap. |
| Full ablation matrix (plan §14 A–I, stochasticity splits) | **Partial** | Strong A/B harnesses exist; not every combinatorial ablation |
| Noise / damage robustness batteries | **Not done** | Planned for generalization characterization |
| Reservoir *weight* learning | **Deliberately avoided** | DEC-008: plastic readout on fixed reservoir (reliability) |

The unfinished items are mostly *extensions* and *comparisons*, not missing prerequisites for claiming the phase ladder — except for external baselines, which are now the most urgent gap.

### 2.4 Engineering achievement

The scientific claims rest on unusually strong engineering for an experimental sim:

- **~6.5k lines of pure Zig**, no external runtime deps, Zig 0.16.
- **Default-off mechanisms**: Phase 1 baseline stays byte-identical as features accumulate.
- **Reproducibility as a hard constraint**: same seed → same spikes; structural evolution independent of firing-draw counts (DEC-004).
- **Padded CSR** for grow/prune without breaking stable traversal (DEC-011) — the decision that made structural plasticity an addition, not a rewrite. Potentially novel as an engineering technique in this context.
- **Experiment harnesses as first-class products**: `train`, `delay`, `grow`, `continual`, `workspace`, `arithmetic`, each with PASS/FAIL and CSV evidence.
- **Tests encode invariants**, not just coverage; harness thresholds sit under deterministic observed results.

This is the difference between a notebook demo and a system you can still trust after nine layers of mechanism.

---

## 3. What the results actually showed

### 3.1 The most interesting findings

#### A. Working memory emerged from an interaction, not from one mechanism

This is probably the strongest conceptual finding.

Self-excitation alone did not yield selective memory. In a fresh network, recurrent gain caused broad activation. Selective persistence appeared only once homeostasis placed individual neurons into an operating regime where stimulus + self-excitation crossed threshold but incidental reservoir feedback and the competing assembly did not.

That means the memory was not simply "stored in a recurrent loop." It was produced by the interaction:

\[
\text{self-excitation}
\times
\text{homeostatic threshold adaptation}
\times
\text{network operating point}.
\]

That is exactly the sort of result that mechanism-oriented computational neuroscience should pursue. The individual ingredients are established, but their interaction in your system produced a nontrivial empirical result. Working memory here is a **system property, not a single knob** — and proving that required the deleted fresh-network probe, which is itself an honest methodological choice.

#### B. Three forms of temporal retention were cleanly separated

You distinguish:

1. **Short fading memory** in the reservoir (echo-state, ~5–10 steps, ~chance by delay 20).
2. **Longer persistence** in a self-exciting assembly (delay 20: 0.996 vs 0.565).
3. **Capacity-limited retention** through a workspace (delay 40: 0.705 vs 0.491 ablation).

That separation is valuable because many projects add several recurrent mechanisms and then merely report better accuracy. You instead showed approximately where the ordinary reservoir stopped carrying sufficient information and where the added mechanisms began earning their keep. The workspace result is especially useful because you removed the self-exciting mechanism and changed workspace with one flag — the causal interpretation is much cleaner than an end-to-end architecture comparison.

#### C. Fast learning and slow consolidation wanted different reward signals

This is a genuinely good design finding.

For weight updates, the reward baseline `r − r̄` was critical — it prevented continuing drift once performance reached ceiling. But using that same baseline-subtracted signal for consolidation caused successful pathways to stop consolidating precisely when the system became consistently correct. The slow permanence process instead needed positive raw reward: `max(0, r·e)`.

The broader concept of multiple timescales and reward-gated consolidation is not new. But this particular failure mode — rapid learners failing to consolidate because the reward prediction baseline eliminates the slow positive signal — is crisp, explainable, and potentially worth documenting formally.

> **Lesson:** A signal appropriate for optimizing a fast variable is not necessarily appropriate for stabilizing a slow variable.

#### D. Structural plasticity and homeostasis formed another meaningful coupling

Random structural growth did not cause catastrophic disruption because homeostasis compensated for changing connectivity. Disuse pruning was naturally weak — homeostasis keeps nearly every neuron and connection active. This led to two useful conclusions:

- Structural plasticity in this regime is **growth-dominated**, not high-churn.
- A **target out-degree** is required for structural equilibrium rather than eventual saturation of all available slots.

These principles exist in prior structural-plasticity literature. Your particular contribution is a strong implementation — the padded CSR engineering — and clean empirical demonstration that the coupling works.

#### E. Learning was exploration-limited rather than learning-rate-limited

The ~600-episode plateau followed by abrupt takeoff is interesting. Tripling the learning rate barely changed the escape point, supporting the hypothesis that the bottleneck is behavioral symmetry breaking or successful action exploration rather than update magnitude.

This should be phrased as a **strong hypothesis supported by your intervention**, not fully proven. Other tests could distinguish it more cleanly: force balanced action exploration; credit only the selected action; introduce lateral inhibition; initialize a tiny controlled asymmetry; compare distributions of time-to-first-successful-streak. If those interventions collapse the plateau while learning-rate changes do not, the exploration explanation becomes much stronger.

#### F. Local three-factor learning works — when credit is short and the substrate is fixed

Immediate association is solved cleanly (8/8 seeds at ceiling) with pre × post × reward eligibility on **plastic readout synapses over a fixed reservoir**. That architecture choice (liquid-state / reservoir + trainable readout) is load-bearing: it makes credit assignment obvious and seed reliability high.

**Implication:** The *where* of plasticity matters as much as the rule. Training the full recurrent net remains a harder, separate problem.

#### G. Arithmetic generalization here is transition composition, not emergent algorithm discovery

Phase 8 binds left/right numerals separately, trains only `n±1` transitions, composes them for one operation, and freezes evaluation on held-out pairs that an exact-pair table cannot answer. Result: **1.000** held-out accuracy vs **1/9** pair-prior.

But the system already contains the algorithm: begin at left operand, apply successor or predecessor, repeat according to right operand. The system has learned or populated local transitions, but the composition procedure is externally supplied. Therefore the result demonstrates:

> A structured transition representation can compose learned unit mappings.

It does **not** demonstrate that the brain-inspired system learned arithmetic composition from reward. The `findings.md` itself handles this honestly; the caveat should be prominent in any public account.

#### H. Phase 9 termination is evidence-triggered, not fully learned

The stopping criterion is designed: a uniquely dominant answer must persist for a fixed stability window. Learning can make an answer become available sooner or more reliably, but the policy for recognizing commitment is hand-specified. A more accurate description is **evidence-triggered adaptive termination** rather than fully learned termination.

### 3.2 Answers to the plan's open questions (partial)

| Open question (plan §22) | Status after Phases 1–9 |
|--------------------------|-------------------------|
| Can adaptive thresholds preserve sparse activity? | **Yes**, with bounded rails and sensible drive |
| Do separate weight & permanence reduce forgetting? | **Yes** (continual A→B→A survival contrast) |
| Does a workspace bottleneck help delayed tasks? | **Yes**, causal ablation at long delay |
| Does structural plasticity preserve useful pathways while changing connections? | **Yes**, with homeostasis + budgets |
| Can intermediate arithmetic state be represented/composed? | **Yes under scaffolding** (transition controller + assemblies) |
| Stochastic firing vs release contribution | **Not cleanly ablated yet** |
| Local spatial rewiring vs global random | **Local implemented; global control not run** |
| Full recurrent plasticity / no labeled outputs | **Out of scope so far** |
| Length generalization (multi-op) | **Not attempted** |

---

## 4. How novel is it?

### 4.1 The individual mechanisms are mostly not novel

The broad ingredients all have substantial precedents:

- Reward-modulated STDP and three-factor rules;
- Eligibility traces for delayed credit;
- Liquid-state machines with fixed recurrent reservoirs and trained readouts;
- Homeostatic intrinsic plasticity;
- Structural growth and pruning;
- Slow synaptic consolidation;
- Stochastic neural and synaptic dynamics;
- Recurrent persistent assemblies;
- Bottlenecked broadcast.

Your reliable Phase 3 architecture is recognizably a liquid-state machine: a fixed recurrent spiking "liquid" plus a trainable readout. Current reservoir-computing work explicitly describes fading memory as a central property and limitation of these systems.

So you should not claim: "I discovered a new biologically plausible learning method." At least not yet.

### 4.2 The combination may be unusual, but combination alone is not enough

I did not find an obvious exact match for your full stack: stochastic spiking reservoir + homeostatic thresholds + local rewarded readout + assembly persistence + permanence-based structural plasticity + reward consolidation + capacity-one workspace + symbolic transition composition. That exact arrangement may well be uncommon.

But almost every sufficiently detailed experimental architecture has a unique combination. Scientific novelty requires more than having a combination nobody happened to implement identically. A stronger novelty claim would be one of these:

1. **A previously undocumented interaction:** mechanism A systematically changes mechanism B in a way that matters across tasks and parameter ranges.
2. **A new algorithmic rule:** your update or architecture solves a recognized limitation better than relevant alternatives.
3. **A new empirical capability:** your system succeeds on an evaluation where closely matched established systems fail.
4. **A new explanatory result:** you identify necessary and sufficient ingredients for a phenomenon through controlled ablations.
5. **A reusable methodological contribution:** your deterministic experimental framework makes structural SNN experiments substantially easier or more reliable.

You may already have early versions of #1 (WM × homeostasis interaction, raw-vs-baseline reward finding) and #5 (padded CSR, derived RNG streams, deterministic harness architecture). You do not yet have convincing evidence for #2 or #3.

### 4.3 Novelty estimate

| Aspect | Assessment |
|--------|------------|
| Spiking, local reward, eligibility | Established |
| Fixed reservoir plus plastic readout | Established |
| Homeostatic selective persistence | Interesting interaction; likely related precedents, but your result is worth probing |
| Raw reward for consolidation versus centered reward for learning | Very good design finding; potentially publishable as part of a broader study |
| Rewiring under homeostatic control | Established direction; strong implementation |
| Padded deterministic CSR rewiring | Potentially novel engineering technique in this context |
| Capacity-one workspace experiment | Interesting integration; not yet a new workspace algorithm |
| Arithmetic result | Valid software result, but heavily scaffolded and not evidence of emergent arithmetic |
| Entire platform | Unusually coherent and rigorous for an independent project |

Relative to the research literature: **moderate originality of integration and experimentation; low novelty of the constituent algorithms; unresolved novelty of the observed interactions.**

---

## 5. What is interesting / where the promise is

### 5.1 Strong promise

1. **Mechanism interaction science.** The best results are not single knobs — they are *couplings*:
   - self-excitation × homeostasis → selective persistence
   - rewiring × homeostasis → stable structural exploration
   - baseline for weights × raw reward for permanence → two timescales of memory
   - eligibility × terminal reward × timeout scalar → one learning path

2. **A reproducible "local learning lab."** Deterministic multi-seed ablations, default-off composition, and byte-identical replays across structural rewiring. That is a platform advantage over typical notebook demos.

3. **Honest intermediate claims.** The project repeatedly refused fake mechanism tests (deleted the fresh-network WM specificity probe when it failed for a real reason). That culture is why the findings are trustworthy.

4. **Continual learning as structure, not just weight regularization.** Permanence bands + pruning give a discrete, measurable survival story that weight L2 cannot.

### 5.2 Moderate promise (conditional)

1. **Workspace as algorithmic primitive.** Capacity-one bottleneck helps long delay; multi-item interference, learned admission, and context tasks remain open.
2. **Structural search for reusable subcircuits.** Topology changes while the task holds — but no demonstration yet that grown edges form *task-specific reusable motifs* rather than mild enrichment.
3. **Neuromorphic / online / damage-robust niches.** The sparse local-update design is well-aligned with these evaluation axes but untested.

### 5.3 Weak / not yet promised

1. **Competing with backprop on arithmetic or language.** Not the goal; not close; wrong comparison.
2. **Emergent algorithm discovery in the reservoir.** Learning still lives mostly at the readout / controller interface.
3. **Biological realism as validation.** Metaphors are labeled as such; that discipline should continue.

---

## 6. Honest limitations

1. **Plasticity is mostly at the interface.** Reservoir weights are fixed; grown reservoir edges are non-plastic. The recurrent substrate is largely a fixed dynamical feature map plus structural turnover.

2. **Arithmetic is scaffolded.** Position-bound encodings, teaching currents during training, and an explicit transition controller do real work. It is good engineering and good curriculum design — it is not pure emergence.

3. **Tasks are small.** Two-choice, small numeral ranges, one operator, fixed or lightly variable episode structure. Ceiling effects (many 100% accuracies) mean future work needs harder tasks or harder splits.

4. **External baselines missing.** Without MLP/RNN/equal-parameter comparisons, claims about sample efficiency, robustness, or "interesting niches" remain qualitative. This is the largest scientific gap. A backprop-trained RNN will probably win on raw sample efficiency and accuracy — the interesting question is whether your system occupies a different Pareto point on forgetting, sparsity, online adaptation, or lesion resistance.

5. **Workspace variance.** Mean causal benefit coexists with seeds near chance. Scientifically fine; a reliability gap if workspace becomes a default substrate.

6. **Scale.** 100 neurons, fixed max out-degree, hand-tuned curricula. Scaling laws are unknown.

7. **Stochasticity not fully characterized.** Firing stochasticity vs release stochasticity was not cleanly ablated — they may be redundant, or may serve different roles (exploration vs regularization).

---

## 7. The single most important next experiment

Do **not** add another brain-inspired feature.

Instead, answer this:

> **Can plasticity inside the recurrent reservoir become necessary for a task that the fixed-reservoir/readout system cannot solve?**

The natural next version is not "more brain modules." It is a controlled recurrent-credit-assignment experiment.

### Proposed experiment

Use a **context-dependent delayed mapping**:

```
context X, cue A → action 1
context X, cue B → action 2
context Y, cue A → action 2
context Y, cue B → action 1
```

Present context and cue at separate times, with a delay before the answer. Design it so that a direct cue→action readout cannot solve the task without access to a context-dependent recurrent state.

Compare:

1. Fixed reservoir + plastic readout (current architecture)
2. Structural reservoir changes only
3. Plastic recurrent edges using the local three-factor rule
4. Recurrent plasticity with reward consolidation
5. Same models with context removed or temporal order shuffled

The success criterion should not merely be that recurrent plasticity performs above chance. It should be:

- The readout-only system reliably fails or reaches a substantially lower ceiling
- Recurrent plasticity succeeds across many seeds
- Freezing or ablating the learned recurrent changes destroys the advantage
- Representational probes show the reservoir carries separable context-dependent states
- The result survives matched parameter counts and training budgets

That would move the project from "a strong synthesis of established mechanisms" toward "evidence that this particular local recurrent learning scheme creates a capability unavailable to the standard fixed-reservoir architecture." That is the threshold where the work becomes much more potentially novel.

---

## 8. Other high-value research tracks

### Track A: Characterize what stochasticity is doing

Run a factorial: deterministic/stochastic firing × deterministic/stochastic release. Measure success rate across 20–50 seeds, median episodes-to-90%, variance in takeoff time, and robustness after learning. Then add forced exploration or winner-take-all action competition. This will tell you whether stochasticity is essential exploration, merely injecting delay, serving regularization, or redundant between neuron and synapse levels.

### Track B: Map the working-memory phase diagram

Vary self-excitation strength, homeostasis rate, target firing rate, inhibitory strength, refractory duration, recurrent delay, stimulus duration, and network size. Map regions: no persistence, selective decaying persistence, nonselective saturation, oscillatory persistence, stable attractor, chaotic/unstable. The exciting result would be showing that homeostasis reliably moves initially unstable assemblies into a narrow selective-memory regime.

### Track C: Determine whether structural growth learns anything task-specific

Test local distance-biased, global random, co-activity-biased, and reward/error-biased growth with matched final degree distributions. Measure whether grown edges preferentially connect task-relevant assemblies, reduce learning time on a second related task, survive consolidation, improve transfer, or become causally necessary when lesioned. The decisive experiment is **transfer**, not mere rewiring.

### Track D: Stress the workspace rather than enlarging it

Test dual delayed cues, distractors, cue replacement, order-sensitive recall, conflicting candidates, and capacity 1 vs 2 vs 4. A non-monotonic capacity result — too little loses information, too much produces interference — would be substantially more interesting than simply increasing average accuracy.

### Track E: External baselines

For each important task, compare against a conventional echo-state machine with linear readout, a tiny RNN trained by BPTT, and a simple tabular or finite-state baseline. Evaluate not just final accuracy but online adaptation, catastrophic forgetting, lesion resistance, sparse activity, local update cost, and adaptation under distribution shift. Your plausible advantages are in these niches, not in raw sample efficiency.

---

## 9. What this project is, finally

It is **not**:

- a brain simulation;
- a new brain learning algorithm;
- a general learning algorithm ready for large-scale ML;
- emergent arithmetic;
- general intelligence;
- or proof that local rules alone invent computation.

It **is**:

- a complete, reproducible implementation of a carefully chosen stack of brain-*inspired* mechanisms;
- a sequence of causal experiments that largely validate the plan's mechanism ladder;
- a body of non-obvious empirical lessons: working memory as a system property (homeostasis × self-excitation interaction); raw reward vs baseline-subtracted reward for consolidation; exploration-limited learning; growth-dominated structural regimes; scaffolded composition;
- a foundation that supports either a published experimental narrative or a second research phase with harder questions;
- and an unusually coherent and rigorous platform for an independent project.

---

## 10. Bottom line

| Dimension | Assessment |
|-----------|------------|
| Plan completion | **Excellent** — Phases 0–9 exit criteria met |
| Scientific honesty | **Excellent** — ablations, deleted false probes, scaffolded claims labeled |
| Engineering quality | **Excellent** — determinism, modularity, harnesses |
| Mechanism novelty | **Low** — individual mechanisms are established |
| Interaction novelty | **Moderate** — WM × homeostasis, raw-vs-baseline reward are genuine findings |
| Platform novelty | **Moderate** — padded CSR, deterministic replay architecture may be novel engineering |
| Immediate ML competitiveness | **Low** (and not the point) |
| Promise if continued carefully | **High** for mechanism science; **moderate** for niche continual/robust/online strengths |
| Worth continuing? | **Yes** — as focused research tracks, not feature stacking |

---

**Recommendation:** freeze v1 as a success. Write the scientific story around the delayed-memory dissociation, consolidation's raw-reward requirement, structural rewiring under homeostasis, workspace causality, and compositional arithmetic under controlled splits. Then open exactly one track — recurrent plasticity on a context-dependent task — with the same phase discipline that made this work.

> **Build one mechanism, prove that it behaves correctly, measure its effect, and only then add the next mechanism.**  
> That mantra already paid off. Keep it.
