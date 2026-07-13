# Final Assessment — Brain-Inspired Local Learning System

**Date:** 2026-07-13  
**Sources:** project plan (`Brain Inspired Local Learning System.md`), `AGENTS.md`, `findings.md`, experiment CSVs, and the Zig implementation (~6.5k LOC under `src/`).

---

## 1. Executive verdict

**This project achieved what it set out to do.** Phases 0–9 of the plan were implemented, each with mechanism tests and multi-seed behavioural harnesses, and every stated exit criterion was met under reproducible conditions.

**Is it worth continuing?** Yes — but not as “add more mechanisms until it becomes a brain.” The platform is already a serious experimental apparatus. The high-value next work is *harder questions with cleaner controls*, not feature accretion.

**Is there real promise?** Yes, of a specific kind. This is not a competitor to backprop on arithmetic benchmarks, and it does not show that unconstrained local rules spontaneously invent algorithms. It *does* show that a small set of biologically motivated mechanisms — sparse stochastic dynamics, homeostasis, three-factor eligibility, working-memory assemblies, structural permanence, reward-gated consolidation, and capacity-limited broadcast — can be composed into a system that:

1. stays alive and sparse,
2. learns from delayed scalar reward,
3. retains information over long delays,
4. rewires without destroying performance,
5. protects useful pathways under disuse,
6. gains causally from a workspace bottleneck, and
7. composes unit transitions into held-out arithmetic better than pair memorization.

Those are substantive scientific and engineering results for a project of this scale.

---

## 2. What was planned vs what was built

### 2.1 The original research question

> **Can organized computation emerge from local reinforcement, stochastic exploration, and selective stabilization of connections?**

The plan was unusually disciplined: one mechanism at a time, ablations, baselines, and explicit “negative results count.” Success was defined as a ladder of increasing capabilities (plan §23), not as beating gradient methods on arithmetic.

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
| 7 | Workspace causally helps a delayed/context task | **Met** | Mean **0.705** vs ablated **0.491** at delay 40 (+0.214) |
| 8 | Arithmetic beats memorization on a controlled split | **Met** | Held-out `a+b=4`: **1.000** vs pair-prior **0.111** (+0.889) |
| 9 | Learned / evidence-based termination | **Met** | Stable unique answer ends episodes; timeout uses distinct −0.2 |

### 2.3 What the plan asked for that was *not* done

These gaps matter for honesty, not as failures of the completed ladder:

| Planned item | Status | Comment |
|--------------|--------|---------|
| Stage D — context-dependent mapping | **Not done** | Would stress multi-signal integration beyond two-choice |
| Stage G — sequential multi-op arithmetic | **Not done** | Explicitly deferred; still the next hard curriculum step |
| Conventional baselines (MLP / RNN / equal params) | **Not done** | Only internal ablations and a pair-lookup arithmetic baseline |
| Full ablation matrix (plan §14 A–I, stochasticity splits) | **Partial** | Strong A/B harnesses exist; not every combinatorial ablation |
| Noise / damage robustness batteries | **Not done** | Planned for generalization characterization |
| Interactive network visualization | **Partial** | Raster + experiment plots; not the 2D spatial network view |
| Reservoir *weight* learning | **Deliberately avoided** | DEC-008: plastic readout on fixed reservoir (reliability) |

The unfinished items are mostly *extensions* and *comparisons*, not missing prerequisites for claiming the phase ladder.

### 2.4 Engineering achievement (often under-rated)

The scientific claims rest on unusually strong engineering for an experimental sim:

- **~6.5k lines of pure Zig**, no external runtime deps, Zig 0.16.
- **Default-off mechanisms**: Phase 1 baseline stays byte-identical as features accumulate.
- **Reproducibility as a hard constraint**: same seed → same spikes; structural evolution independent of firing-draw counts (DEC-004).
- **Padded CSR** for grow/prune without breaking stable traversal (DEC-011) — the decision that made structural plasticity an addition, not a rewrite.
- **Experiment harnesses as first-class products**: `train`, `delay`, `grow`, `continual`, `workspace`, `arithmetic`, each with PASS/FAIL and CSV evidence.
- **Tests encode invariants**, not just coverage; harness thresholds sit under deterministic observed results.

This is the difference between a notebook demo and a system you can still trust after nine layers of mechanism.

---

## 3. What the results actually showed

### 3.1 Core scientific conclusions

#### A. Local three-factor learning works — when credit is short and the substrate is fixed

Immediate association is solved cleanly (8/8 seeds at ceiling) with pre × post × reward eligibility on **plastic readout synapses over a fixed reservoir**. That architecture choice (liquid-state / reservoir + trainable readout) is load-bearing: it makes credit assignment obvious and seed reliability high.

**Implication:** “Local reinforcement can organize useful computation” is true here, but the *where* of plasticity matters as much as the rule. Training the full recurrent net was correctly treated as a harder, separate problem.

#### B. Learning is exploration-limited, not rate-limited

~600-episode near-chance plateau, then sharp takeoff. Tripling learning rate barely moves takeoff and can slightly hurt final accuracy. The bottleneck is **breaking initial symmetry via stochastic exploration**, not gradient step size.

**Implication:** If episodes-to-criterion matters later, invest in action competition / better credit (chosen-action only, lateral inhibition), not larger `η`.

#### C. Homeostasis is not optional infrastructure — it is a computational partner

- It recovers a target activity band after sustained drive.
- Threshold rails have bounded authority; a pinned `threshold_max` means the *operating point* is wrong, not that the rail should keep rising.
- Working-memory *specificity* (A stays on, B stays off) is **emergent from self-excitation + homeostasis trained into place** — not from recurrent gain alone. A fresh network with self-excitation saturates globally.

**Implication:** “Working memory” here is a **system property**, not a single knob. That is one of the project’s best conceptual results.

#### D. There are two memory timescales in the substrate, and they must not be conflated

| Mechanism | What it buys | Measured signature |
|-----------|--------------|--------------------|
| Reservoir fading memory (echo-state) | Short delays “for free” | Useful ~5–10 steps; ~chance by delay ~20 |
| Self-exciting input assembly | Long-delay retention | Delay 20: **0.996** vs reservoir-only **0.565** |
| Capacity-one workspace (no self-excitation) | Causal long-delay gain via bottleneck + broadcast | Delay 40: **0.705** vs **0.491** ablation |

The delay curves *diverge where the reservoir runs out*. That is the cleanest controlled result in the project: memory mechanisms earn their keep specifically for **long** delays.

#### E. Structural plasticity can rewire without destroying the task — if paired with homeostasis and budgets

- Graph enriches (~20% more live reservoir edges, different topology).
- Disuse pruning is **rare** in a homeostatic, task-active network — and that is correct: nearly every synapse is “used.”
- Without a **target out-degree**, growth accretes to the hard slot cap; with it, structure can turn over around a set-point.
- Exit criterion is a **three-way AND**: churn > 0, relative activity stability, task accuracy preserved.

**Implication:** “Connections change” is not the same as “aggressive pruning.” The honest regime is **growth-dominated exploration with consolidation protecting what works**.

#### F. Consolidation is real — and raw reward vs baseline is a critical distinction

- Weight learning needs **baseline-subtracted** reward (zero-mean updates once solved).
- Permanence consolidation needs **raw** reward (`η_q·max(0, r·e)`). Using the baseline kills consolidation exactly on the seeds that learn fastest (baseline → +1, modulator → 0).
- Consolidated vs tentative survival after A→B disuse: **~0.97 vs ~0.00**.
- Measuring survival by weight magnitude is misleading; small weights can still win readouts.

**Implication:** Fast learning stability and slow structural memory want **opposite signals from the same reward**. That is a genuine design lesson, not a tuning anecdote.

#### G. Workspace broadcast can be causal — with seed variance

Removing self-excitation and enabling only a capacity-one workspace yields a multi-seed mean gain (+0.214) on a 40-step delay task. Ablated workspace state is exactly 0; enabled delay state sits ~0.94–0.99.

Caveat: enabled seeds range **0.484–1.000**. The claim is a **mean causal benefit**, not universal seed perfection. That honesty in the criterion design is correct science.

#### H. Arithmetic generalization here is transition composition, not spontaneous symbol invention

Phase 8 deliberately:

- binds left/right numerals separately,
- trains only `n±1` transitions,
- composes them for one operation,
- freezes evaluation on held-out pairs that an exact-pair table cannot answer.

Result: **1.000** held-out accuracy vs **1/9** pair-prior. Phase 9 adds answer-ID-neutral stable termination through the same reward path.

**Critical interpretation:** this is a **structured controller coupled to spiking assemblies**, not evidence that the unconstrained reservoir invented arithmetic. The plan’s own caution about memorization is respected — and so should any future writeup’s language.

### 3.2 Answers to the plan’s open questions (partial)

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

### 3.3 Against the plan’s success ladder (§23)

Levels 1–9 of the ladder are essentially achieved:

1. Stable sparse activity  
2. Immediate associations  
3. Delayed association  
4. (Stochasticity present and load-bearing for exploration; not fully ablated as an effect size)  
5. Structural growth preserving utility  
6. Consolidation reducing forgetting of useful pathways  
7. Workspace-assisted delayed retention  
8–9. One-op arithmetic above memorization + held-out structure + stable termination  

Level 10 — *interpretable emergent assemblies or pathways* — is only partially met. Working-memory assemblies and consolidated readout pathways are interpretable by construction and measurement; truly *discovered* subcircuits in the reservoir are not yet shown.

---

## 4. What is interesting / where the promise is

### 4.1 Strong promise (keep, deepen)

1. **Mechanism interaction science.** The best results are not single knobs — they are *couplings*:
   - self-excitation × homeostasis → selective persistence  
   - rewiring × homeostasis → stable structural exploration  
   - baseline for weights × raw reward for permanence → two timescales of memory  
   - eligibility × terminal reward × timeout scalar → one learning path for answer and stop  

2. **A reproducible “local learning lab.”** Few hobby/research toys can claim: deterministic multi-seed ablations, default-off composition, and byte-identical replays across structural rewiring. That is a platform advantage.

3. **Honest intermediate claims.** The project repeatedly refused fake mechanism tests (e.g. deleted the fresh-network WM specificity probe when it failed for a real reason). That culture is why the findings are trustworthy.

4. **Continual learning as structure, not just weight regularization.** Permanence bands + pruning give a discrete, measurable survival story that weight L2 cannot.

5. **Composition as a curriculum principle.** Training unit transitions and evaluating held-out pairs is the right kind of generalization claim for a small system.

### 4.2 Moderate promise (conditional)

1. **Workspace as algorithmic primitive.** Capacity-one bottleneck helps long delay; multi-item interference, learned admission, and context tasks remain open. Interesting *if* the next experiments stay one-flag causal.

2. **Structural search for reusable subcircuits.** Topology changes while the task holds — but there is not yet a demonstration that grown edges form *task-specific reusable motifs* rather than mild enrichment.

3. **Neuromorphic / online / damage-robust niches.** The plan correctly notes these may matter more than raw accuracy. Untested, but the sparse local-update design is well-aligned with those evaluation axes.

### 4.3 Weak / not yet promised

1. **Competing with backprop on arithmetic or language.** Not the goal; not close; wrong comparison.
2. **Emergent algorithm discovery in the reservoir.** Learning still lives mostly at the readout / controller interface.
3. **Biological realism as validation.** Metaphors are labeled as such; that discipline should continue.

---

## 5. Honest limitations (do not paper over these)

1. **Plasticity is mostly at the interface.** Reservoir weights are fixed; grown reservoir edges are non-plastic. “Local learning system” is accurate for the *rule class*, but the recurrent substrate is largely a fixed dynamical feature map plus structural turnover.

2. **Arithmetic is scaffolded.** Position-bound encodings, teaching currents during training, and an explicit transition controller do real work. That is good engineering and good curriculum design — it is not pure emergence.

3. **Tasks are small.** Two-choice, small numeral ranges, one operator, fixed or lightly variable episode structure. Ceiling effects (many 100% accuracies) mean future work needs harder tasks or harder splits to keep learning something.

4. **External baselines missing.** Without MLP/RNN/equal-parameter comparisons, claims about sample efficiency, robustness, or “interesting niches” remain qualitative.

5. **Workspace variance.** Mean causal benefit coexists with seeds near chance. That is scientifically fine; it is also a reliability gap if workspace is to become a default substrate.

6. **Scale.** 100 neurons, fixed max out-degree, hand-tuned curricula. Scaling laws, larger reservoirs, and automatic curricula are unknown.

7. **Open threads already noted in findings** remain valid: better credit assignment; true attractor WM; reservoir eligibility; activity-biased growth for visible turnover; multi-item workspace; stronger arithmetic ablations.

---

## 6. Is this worth continuing?

### Yes — under a clear charter

**Continue if the goal is:**

- a research sandbox for local, delayed, structural learning;
- mechanism science (what is necessary, what couples, what fails);
- eventually niche strengths (continual learning, sparsity, damage, online adaptation);
- or a path toward more ambitious recurrent local learning with the same experimental standards.

**Stop or freeze if the goal is:**

- state-of-the-art accuracy on symbolic or perceptual benchmarks;
- a product ML stack;
- or “prove brains work this way.”

The project already produced a **complete vertical slice** of the original plan. That is a success even as a closed chapter. Continuing is justified because the *next* questions are now well-posed and the infrastructure can answer them cheaply.

### Recommended stance

Treat Phases 1–9 as **v1 complete**. Do not invent Phase 10 as “more mechanisms.” Open **research tracks** with pass/fail questions, each with ablations from day one.

---

## 7. Highest-value improvements (near term)

These improve scientific return on the existing system without a redesign.

### 7.1 Science quality

1. **External baselines for the two-choice and arithmetic curricula**  
   Small RNN / echo-state + linear readout / tiny MLP on the same episode budget and seed set. Report equal-parameter and equal-example variants. Even if the local system loses on accuracy, document sparsity, forgetting, and online update cost.

2. **Stochasticity ablation matrix**  
   Deterministic fire / stochastic fire × deterministic release / stochastic release, on train and delay. Answers plan §22 directly.

3. **Credit-assignment upgrades for speed, not accuracy**  
   Lateral inhibition / WTA between action groups; credit only the chosen action. Measure episodes-to-90% across seeds. Findings already predict this is the right lever.

4. **Harder evaluation before harder models**  
   - Context-dependent mapping (Stage D)  
   - Held-out *operators* and *results*, not only combinations  
   - Multi-op sequential arithmetic with frozen unit transitions  
   - Noise / drop spikes / synapse deletion robustness  

5. **Reservoir interpretability probes**  
   After training, freeze and measure: which assemblies carry the stimulus during delay; whether grown edges concentrate near input/action groups; whether consolidated plastic edges form sparse pathways that explain readout winners.

### 7.2 Mechanism fixes / extensions that findings already justify

6. **Bistable / attractor working memory** if delays need to be unbounded — current WM is a *decaying* persistent state (~0.66 → ~0.15 over 40 steps).

7. **Activity- or error-biased growth (+ quiet-baseline co-activity)** if studying forgetting/turnover; pure local random + homeostatic rates yields quiet pruning.

8. **Reservoir edges eligible for reward-gated permanence** (optional flag, default off) — the natural next credit-assignment experiment after DEC-012, with severe ablations so reservoir plasticity cannot silently ruin determinism or stability.

9. **Workspace v2 only after one-flag discipline:** multi-item capacity, interference tasks, learned admission — each as a separate harness.

### 7.3 Engineering polish

10. **Plot scripts for workspace + arithmetic**; unify harness summary JSON (mean, gap, pass/fail, seed list).  
11. **Config presets** already in `configs/` — add “paper figure” configs that reproduce each findings number with one command.  
12. **Parametric scale tests** (N=100/200/400) on delay and continual only — check whether results are fragile to size.  
13. **Remove / quarantine temp probes** (`probe_tmp.zig` etc.) so the public surface matches AGENTS.md.

---

## 8. Extension roadmap (if continuing)

Prioritized by insight-per-effort and fidelity to the original hypothesis.

### Track A — Credit assignment (highest scientific leverage)

**Question:** How far can imperfect local credit go once the reservoir is allowed to change functionally, not just topologically?

- Optional plastic recurrent edges with very slow rates and strong homeostasis.  
- e-prop-like or random-feedback variants as *comparisons*, not rewrites.  
- Tasks: delayed association, context-dependent mapping, continual A/B.

**Success:** multi-seed above chance with recurrent plasticity *and* an ablation showing the plastic recurrent edges are necessary for a harder task the readout-only system fails.

### Track B — Compositional curricula (highest narrative leverage)

**Question:** Can transition composition scale past one operation without becoming a hand-built CPU?

- Multi-op sequences; intermediate state must survive workspace or WM.  
- Ablate teaching current and transition controller separately (findings already warn this is required for stronger claims).  
- Magnitude encodings if number-range generalization is attempted.

**Success:** held-out length or operator generalization above memorization and above a frozen pair table; controller ablation drops performance.

### Track C — Structure that means something (highest originality leverage)

**Question:** Does local random search discover reusable subcircuits, or only denser noise?

- Quantify motif statistics, input-assembly affinity, pathway reuse across tasks.  
- Compare local spatial growth vs global random growth (plan §14).  
- Continual learning where *structure* must transfer, not only readout permanence.

**Success:** grown connectivity predicts transfer; global-random growth underperforms local growth on a transfer task with matched degree budgets.

### Track D — Limited global broadcast under load

**Question:** When does capacity-one help vs hurt?

- Dual delayed cues, order-sensitive tasks, interrupt / rewrite trials.  
- Measure interference curves vs capacity.  
- Keep causal on/off ablation sacred.

**Success:** non-monotonic capacity curve or a task family where capacity-one is necessary for sequential control.

### Track E — Niche evaluation (application story)

**Question:** Where, if anywhere, do local sparse systems win?

- Online learning under nonstationarity  
- Synapse/neuron lesion robustness  
- Energy proxy: spikes × active synapses per correct answer  
- Catastrophic forgetting vs small replay-free backprop nets  

**Success:** a clear Pareto win on robustness/forgetting/energy even if accuracy is worse.

---

## 9. What this project is, finally

It is **not**:

- a brain simulation,
- a general learning algorithm ready for large-scale ML,
- or proof that local rules alone invent arithmetic.

It **is**:

- a complete, reproducible implementation of a carefully chosen stack of brain-*inspired* mechanisms;
- a sequence of causal experiments that largely validate the plan’s mechanism ladder;
- a body of non-obvious empirical lessons (exploration-limited learning; WM as system property; raw vs baseline reward; growth-dominated structural regimes; scaffolded composition);
- and a foundation that can support either a published experimental narrative or a second research phase with harder questions.

The central hypothesis receives a **qualified yes**:

> Organized computation *can* emerge from local reinforcement, stochastic exploration, and selective stabilization — **when** the system is given stable dynamics (homeostasis), a usable memory substrate (assemblies / workspace), short credit paths (readout plasticity), and, for symbolic composition, structured curricula. Without those supports, the same rules do not magically invent the solution.

That qualified yes is more valuable than a vague absolute yes. It tells you *which* ingredients earn their keep.

---

## 10. Bottom line

| Dimension | Assessment |
|-----------|------------|
| Plan completion | **Excellent** — Phases 0–9 exit criteria met |
| Scientific honesty | **Excellent** — ablations, deleted false probes, scaffolded claims labeled |
| Engineering quality | **Excellent** — determinism, modularity, harnesses |
| Novelty | **Moderate–high** for a small personal research system; mechanism *couplings* are the story |
| Immediate ML competitiveness | **Low** (and not the point) |
| Promise if continued carefully | **High** for mechanism science; **moderate** for niche continual/robust/online strengths |
| Worth continuing? | **Yes** — as focused research tracks, not feature stacking |

**Recommendation:** freeze v1 as a success; write the scientific story around the delayed-memory dissociation, consolidation’s raw-reward requirement, structural rewiring under homeostasis, workspace causality, and compositional arithmetic under controlled splits; then open only one or two tracks (credit assignment *or* compositional curriculum *or* structural meaning) with the same phase discipline that made this work.

> **Build one mechanism, prove that it behaves correctly, measure its effect, and only then add the next mechanism.**  
> That mantra already paid off. Keep it.
