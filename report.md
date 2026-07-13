### 1. Arithmetic and termination are currently controller-dominated

This is the largest issue—not a hidden software bug, because `findings.md` and `final.md` acknowledge the scaffolding, but the harness verdict remains stronger than the evidence.

The transition model calculates the exact composed answer. Evaluation then:

- injects answer-specific current of `100.0` into that answer assembly;
- inhibits all competing assemblies;
- adds a direct answer vote to `counts` every readout step.

See the [controller injection](/home/phagmaier/Code/brain/src/arithmetic_curriculum.zig:193) and [direct termination vote](/home/phagmaier/Code/brain/src/arithmetic_curriculum.zig:232). Evaluation always supplies `transitions.solve(e)` in [trainOne](/home/phagmaier/Code/brain/src/arithmetic_curriculum.zig:302).

Consequently:

- The 1.000 held-out score proves that the explicit successor/predecessor controller composes transitions correctly.
- It does not establish that the learned spiking readout is necessary.
- Stable termination is almost guaranteed once the controller supplies ten group-sized votes per step.
- The pair-lookup baseline is too weak; a controller-only finite-state baseline would likely achieve the same score.

Recommended correction: report four separate ablations:

1. Controller-only, no SNN.
2. Controller current into SNN, but spike counts only—no direct vote.
3. Learned SNN readout with the controller removed.
4. Untrained/frozen readout with the controller present.

Phase 9 should only count actual answer-assembly spikes. Until then, call it “evidence-triggered termination over controller-assisted answers,” not learned termination.

### 2. The supposedly unbiased integer sampler is mathematically biased

The rejection bound in [rng.zig](/home/phagmaier/Code/brain/src/rng.zig:91) accepts a number of `u64` values that is generally not divisible by `n`. The bias is astronomically small for the ranges used here, so it almost certainly did not affect the reported results, but the implementation and test claim are incorrect.

Fix the rejection calculation and add a small-word reference helper that exhaustively tests every possible input. Because this changes RNG semantics, bump `prng_impl_version` in [rng.zig](/home/phagmaier/Code/brain/src/rng.zig:25), even if current seeds happen to produce identical output.

### 3. Configuration validation does not fully protect model invariants

[Config.validate](/home/phagmaier/Code/brain/src/config.zig:365) covers the headline constraints but leaves many dangerous values accepted. Examples include:

- Negative `task_ia_weight_init` or `task_recurrent_weight`, violating the non-negative-weight rule in optimized builds.
- Negative or non-finite learning rates and trace increments.
- Invalid pre/post trace and reward-baseline decays.
- Invalid adaptation, structural decay, coactivity, or weight-decay coefficients.
- `NaN`/infinity, which can bypass ordinary comparisons.
- Consolidation enabled without plasticity or any plastic task.
- Task and arithmetic enabled together even though both layouts occupy low excitatory IDs.
- Non-uniform graph construction with a zero/non-finite spatial sigma.

Centralize finite/range validation and add table-driven invalid-config tests. Either reject simultaneous task/arithmetic layouts or allocate them explicitly disjoint ranges.

### 4. Allocation failure cleanup is incomplete

The multi-allocation initializers construct entire structures before installing cleanup:

- [Neurons.init](/home/phagmaier/Code/brain/src/net.zig:53)
- Synapse arrays in [Network.build](/home/phagmaier/Code/brain/src/net.zig:343)
- `Trace.init` in the perturbation harness

If a later allocation fails, earlier allocations can leak because the `errdefer` is only established after the full initializer succeeds. Allocate fields incrementally with paired `errdefer`s, and test using `std.testing.checkAllAllocationFailures`.

This does not affect normal simulations, but it conflicts with the project’s explicit allocator-cleanup convention.

### 5. The consolidation experiment needs a stronger causal protocol

The structural survival contrast is real, but comparing high-permanence against low-permanence synapses is partly built into the pruning rule: high permanence is definitionally what resists pruning.

More importantly, in the reproduced OFF runs, two seeds had not convincingly mastered block A (`0.473` and `0.740`) before the forgetting comparison. See [runCondition](/home/phagmaier/Code/brain/src/continual.zig:155). That weakens the functional retest comparison.

Improve it by:

- training each condition to a prespecified mastery criterion before block B;
- excluding/reporting non-mastered seeds separately;
- using 20–50 paired seeds;
- measuring pathway-specific lesion effects, not only survival;
- comparing raw-reward consolidation against centered-reward consolidation directly.

### 6. Reproducibility needs a cross-version golden guard

[check-determinism.sh](/home/phagmaier/Code/brain/scripts/check-determinism.sh:25) proves two runs of the current code agree. It does not prove that the Phase 1 baseline remained byte-identical across feature additions.

Add a small committed golden manifest containing:

- default artifact hashes;
- config and PRNG version;
- Zig version;
- source commit;
- expected summary statistics.

Intentional dynamics changes would update it explicitly. This would turn the “baseline unchanged” claim into an automatically enforced regression property.

### 7. Scaling and maintainability

The current complexity is appropriate for 100 neurons, but structural growth scans every candidate target and rescans the outgoing slice for duplicates, roughly \(O(N^2D)\) per growth event. Preallocated target-mark scratch space could reduce that while preserving stable order and RNG behavior.

The largest routines—`Network.build`, `Sim.step`, `applyStructuralPlasticity`, and arithmetic `runEpisode`—would benefit from private stage functions. Keep `Sim.step` itself as an explicit ordered sequence so DEC ordering remains auditable.

## Recommended next stages

### Stage 0: Harden and freeze v1

Fix the RNG bound, allocation cleanup, and configuration validation. Add golden baseline hashes and experiment provenance manifests. Re-run all tests and harnesses; document any intentional PRNG-version change.

### Stage 1: Validate the existing scientific claims more aggressively

Before adding mechanisms:

- Run the arithmetic/controller ablation matrix above.
- Strengthen the consolidation protocol.
- Expand workspace and continual experiments to 20–50 paired seeds with confidence intervals.
- Add conventional baselines: linear reservoir readout, small BPTT RNN, and finite-state/tabular controls.
- Measure online-update cost, sparsity, forgetting, lesion resistance, and distribution-shift adaptation—not only final accuracy.

### Stage 2: Recurrent plasticity on a context-dependent delayed task

I agree with `final.md`: this is the best flagship next experiment.

Use the delayed XOR-style mapping:

- context X + cue A → action 0
- context X + cue B → action 1
- context Y + cue A → action 1
- context Y + cue B → action 0

Present context and cue at separate times. Prevent direct context/cue-to-action shortcuts. Compare:

1. Fixed reservoir + plastic readout.
2. Structural changes only.
3. Locally plastic recurrent edges.
4. Recurrent plasticity plus consolidation.
5. Frozen/lesioned learned recurrent edges.

Require not merely above-chance performance, but a reliable advantage over readout-only, loss of that advantage after recurrent lesions, and representational evidence that context changes the cue state.

That would be the project’s first strong test of locally learned computation inside the recurrent substrate.

### Stage 3: Mechanism-science tracks

The highest-value secondary experiments are:

- Firing stochasticity × release stochasticity factorial, plus forced exploration and winner-take-all credit.
- Working-memory phase diagram across excitation, homeostasis, inhibition, delay, refractory period, and size.
- Structural growth controls: local random versus global, coactivity-biased, and reward/error-biased growth; evaluate transfer and targeted lesions.
- Workspace stress: distractors, replacement, ordered recall, conflicting cues, and capacity 1/2/4.
- Robustness: neuron lesions, synapse deletion, input noise, parameter drift, and post-training distribution shifts.

## Bottom line

Phases 1–7 form a credible, reproducible mechanism platform. The strongest results remain the interaction findings: homeostasis enabling selective persistent memory, homeostasis stabilizing rewiring, and raw reward serving consolidation while centered reward stabilizes fast learning.

The immediate priority should be correctness hardening and dismantling the arithmetic/controller confound. After that, recurrent plasticity on a context-dependent delayed task is the clearest route to a genuinely more interesting—and potentially more novel—result.
