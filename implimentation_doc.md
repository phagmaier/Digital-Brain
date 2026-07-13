---
title: Brain-Inspired Local Learning System
aliases:
  - Experimental Local-Learning Spiking Network
  - Brain Metaphor Learning Project
tags:
  - project
  - machine-learning
  - neuroscience
  - spiking-neural-networks
  - artificial-life
  - zig
status: idea
created: 2026-07-12
---

# Brain-Inspired Local Learning System

> [!abstract]
> Build a small computational system inspired by several mechanisms observed in biological brains—not a literal brain simulation. The system will consist of stochastic spiking neurons, excitatory and inhibitory signaling, recurrent activity, local reward-modulated learning, homeostasis, and eventually structural plasticity and a limited global workspace.
>
> The main research question is:
>
> **Can organized computation emerge from local reinforcement, stochastic exploration, and selective stabilization of connections?**

## 1. Project philosophy

The aim is **not** to reproduce every biological detail or claim that neuroscience has discovered one complete “brain learning algorithm.” Instead, the project translates a selected set of biological ideas into simple computational metaphors and tests what those mechanisms can do.

The project should prioritize:

1. **Computationally meaningful biological inspiration**
2. **Local rules rather than global backpropagation**
3. **Explicit time and recurrent state**
4. **Controlled experiments and comparison baselines**
5. **Incremental implementation**
6. **Interpretability and visualization**
7. **Learning from failure**

A system that only learns a delayed two-choice task may still be a successful project if it reveals why particular mechanisms are necessary.

> [!warning]
> Do not begin by implementing every idea simultaneously. A failed full system would be almost impossible to debug because it would be unclear whether the problem came from neural dynamics, encoding, plasticity, reward timing, structural growth, the workspace, or the task itself.

---

## 2. Central hypothesis

A useful learning system may emerge from the interaction of:

- sparse stochastic firing;
- excitatory and inhibitory populations;
- recurrent temporal activity;
- local synaptic traces;
- delayed global reward;
- homeostatic regulation;
- occasional local connection search;
- pruning and consolidation;
- competition for limited global broadcast.

A conventional neural network learns by directly optimizing a global objective through gradients. This project asks whether a system can instead discover useful computation through **local trial, delayed reinforcement, and stabilization**.

---

## 3. What is biologically motivated versus invented

### Biologically motivated ideas

These have recognizable counterparts in neuroscience:

- Neurons communicate through discrete spikes.
- Neural systems operate over time.
- Neurons and synapses can behave stochastically.
- Chemical synaptic transmission can fail probabilistically.
- Neurons are commonly excitatory or inhibitory in their outgoing effect.
- Synaptic strength depends partly on relative spike timing.
- Neuromodulators can alter plasticity based on behavioral significance.
- Neurons regulate their own excitability and firing rates.
- Synapses and dendritic spines can form, stabilize, weaken, and disappear.
- Brain networks are recurrent and sparse.
- Global workspace theories involve competition, ignition, and broad broadcast.

### Deliberate computational inventions

These are engineering metaphors, not claims about exact brain implementation:

- A scalar synapse `permanence` value
- Explicit workspace slots
- A discrete `done` neuron
- Fixed Euclidean neuron coordinates
- A sigmoid firing probability
- A reward value of exactly `+1` or `-1`
- A strict maximum number of synapses per neuron
- Reading an answer by counting output spikes
- Running structural growth every fixed number of steps

These simplifications are legitimate as long as they are labeled as hypotheses.

---

## 4. Initial system architecture

### 4.1 Neurons

Each neuron should initially contain:

```c
id
type                    // excitatory or inhibitory
position                // optional 2D coordinate
membrane_potential
threshold
resting_potential
refractory_remaining
adaptation
recent_firing_rate
pre_trace
post_trace
incoming_synapse_ids
outgoing_synapse_ids
```

A neuron emits a binary event:

$$
s_i(t)\in\{0,1\}
$$

It does **not** itself emit `-1`. Instead, its fixed type determines whether its outgoing spikes increase or decrease postsynaptic membrane potential.

Define:

$$
d_i =
\begin{cases}
+1 & \text{excitatory neuron}\\
-1 & \text{inhibitory neuron}
\end{cases}
$$

Then an arriving spike contributes:

$$
I_j(t+1) \mathrel{+}= d_i w_{ij}.
$$

Keep:

$$
w_{ij}\ge 0
$$

so sign and magnitude remain separate.

This is a simplified computational version of maintaining separate excitatory and inhibitory neurons.

### 4.2 Synapses

Each synapse can initially contain:

```c
source_neuron
target_neuron
weight
release_probability
delay
eligibility
permanence
age
last_active_step
is_plastic
```

Later additions might include:

```text
short_term_facilitation
short_term_depression
consolidation_level
utility_estimate
```

Avoid these until the basic system works.

### 4.3 Populations

A useful initial split:

- 80% excitatory neurons
- 20% inhibitory neurons

This is only a starting engineering choice, not a universal biological law.

Suggested functional groups:

- **sensory/input neurons**
- **recurrent association neurons**
- **inhibitory neurons**
- **action/readout neurons**
- eventually **workspace gateway/broadcast neurons**

The action neurons should still be ordinary members of the recurrent system. They are an interface to the environment, not necessarily a privileged dense output layer.

---

## 5. Discrete-time dynamics

Use explicit discrete timesteps:

$$
t=0,1,2,\ldots,T-1.
$$

A simulation step can be:

1. Deliver delayed spikes scheduled for this timestep.
2. Add external sensory input.
3. Add any current workspace broadcast.
4. Leak neuron membrane potentials toward rest.
5. Apply refractory and adaptation effects.
6. Compute firing probability for each eligible neuron.
7. Sample spikes.
8. Schedule outgoing synaptic transmissions.
9. Update pre- and postsynaptic traces.
10. Update synaptic eligibility traces.
11. Update workspace competition, if enabled.
12. Apply homeostatic adjustments on the chosen schedule.
13. Check episode termination.

Do **not** initially update weights or grow connections on every timestep unless the rule specifically requires it.

### 5.1 Basic membrane update

A simple discrete leaky integrate-and-fire metaphor:

$$
V_i(t+1)
=
\lambda_V V_i(t)
+
I_i(t)
-
s_i(t)V_{\text{reset}}
-
a_i(t)
$$

where:

- $V_i$ is membrane potential;
- $\lambda_V\in[0,1]$ controls leak;
- $I_i(t)$ is incoming current;
- $a_i(t)$ is adaptation;
- firing resets or subtracts from potential.

A more explicit form:

$$
V_i(t+1)
=
V_{\mathrm{rest}}
+
\lambda_V\left(V_i(t)-V_{\mathrm{rest}}\right)
+
I_i(t)
-
s_i(t)\Delta V_{\mathrm{reset}}.
$$

### 5.2 Stochastic firing

Instead of a strict threshold:

$$
P(s_i(t)=1)
=
\sigma\left(\beta(V_i(t)-\theta_i)\right)
$$

where:

- $\sigma$ is a sigmoid;
- $\theta_i$ is the firing threshold;
- $\beta$ controls determinism.

Large $\beta$ produces nearly deterministic threshold firing. Smaller $\beta$ creates more exploratory firing.

Possible implementation:

```text
p_fire = sigmoid(beta * (potential - threshold))
fired = random_float_0_1() < p_fire
```

A refractory neuron cannot fire regardless of probability.

> [!note]
> Stochastic neural behavior should not automatically be described as “the brain’s dropout.” It may provide regularization or exploration in this model, but biological variability has many causes and possible functions.

### 5.3 Stochastic synaptic release

A spike may fail to cross a synapse:

$$
z_{ij}(t)
=
s_i(t)\cdot
\operatorname{Bernoulli}(p^{\mathrm{release}}_{ij}).
$$

Then:

$$
I_j(t+d_{ij})
\mathrel{+}=
d_i w_{ij}z_{ij}(t).
$$

This separates:

- whether the neuron fires;
- whether a particular synapse transmits the spike.

These two sources of randomness can be independently ablated later.

### 5.4 Refractory period

After firing:

```python
refractory_remaining = refractory_steps
```

Each timestep:

```python
if refractory_remaining > 0:
    refractory_remaining -= 1
    cannot_fire = true
```

### 5.5 Spike-frequency adaptation

Repeated firing should make firing temporarily more difficult:

$$
a_i(t+1)
=
\lambda_a a_i(t)+\alpha_a s_i(t).
$$

Adaptation can either subtract from membrane potential or increase the threshold.

---

## 6. Learning rules

### 6.1 Fast activity traces

Maintain a decaying trace of recent spikes:

$$
x_i(t+1)=\lambda_x x_i(t)+s_i(t)
$$

for presynaptic activity, and:

$$
y_j(t+1)=\lambda_y y_j(t)+s_j(t)
$$

for postsynaptic activity.

These traces allow a synapse to remember recent local events without storing the full simulation history.

### 6.2 Timing-sensitive eligibility

A simple eligibility rule:

$$
e_{ij}(t+1)
=
\lambda_e e_{ij}(t)
+
x_i(t)s_j(t)
-
\alpha\,s_i(t)y_j(t).
$$

Interpretation:

- presynaptic firing shortly before postsynaptic firing creates positive eligibility;
- the reverse order can create negative eligibility;
- eligibility fades over time.

The precise formula should be treated as an experimental choice.

### 6.3 Reward-modulated update

When an episode receives reward $r$:

$$
\Delta w_{ij} = \eta_w r e_{ij}.
$$

Then:

$$
w_{ij}
\leftarrow
\operatorname{clip}
\left(
w_{ij}+\Delta w_{ij},
w_{\min},
w_{\max}
\right).
$$

Potential rewards:

```text
correct answer:       +1.0
incorrect answer:     -1.0
timeout:              -0.2
excessive activity:   optional small penalty
```

Start with simple terminal reward. Do not add many handcrafted rewards until you know they are required.

### 6.4 The credit-assignment limitation

This rule cannot precisely determine which synapses caused a delayed answer. It only reinforces recently eligible connections.

That is not merely a bug; it is a central experiment:

> How far can imperfect local credit assignment go?

Keep this distinction clear:

- **STDP-like timing** discovers local temporal correlation.
- **Reward modulation** indicates whether recent behavior was useful.
- Their product approximates useful credit assignment.

### 6.5 Weight decay

A basic slow decay:

$$
w_{ij}\leftarrow (1-\lambda_w)w_{ij}.
$$

Do not apply strong decay indiscriminately. Later, make decay depend on permanence or consolidation.

---

## 7. Homeostasis

Without homeostasis, local Hebbian rules can cause:

- runaway excitation;
- permanent silence;
- a few neurons dominating every task;
- exploding weights;
- loss of useful variability.

### 7.1 Adaptive firing threshold

Track recent firing rate:

$$
\rho_i(t+1)
=
\lambda_\rho \rho_i(t)
+
(1-\lambda_\rho)s_i(t).
$$

Adjust threshold:

$$
\theta_i
\leftarrow
\theta_i
+
\eta_h(\rho_i-\rho_{\mathrm{target}}).
$$

If the neuron fires too frequently, its threshold rises. If it fires too little, its threshold falls.

### 7.2 Weight normalization

Possible rule:

$$
w_{ij}
\leftarrow
w_{ij}
\frac{C}
{\sum_k w_{ik}+\epsilon}
$$

for a fixed outgoing synaptic budget $C$.

Use cautiously: normalization can interfere with the meaning of reward updates.

### 7.3 Excitation/inhibition balance

Monitor:

```text
mean firing rate
fraction of silent neurons
fraction firing per step
excitatory input magnitude
inhibitory input magnitude
population synchrony
```

The first dynamics milestone is not task accuracy. It is maintaining a healthy activity regime that neither explodes nor dies.

---

## 8. Structural plasticity: local random connection search

This is one of the project’s defining ideas.

### 8.1 Spatial organization

Assign each neuron a coordinate:

$$
\mathbf{x}_i\in\mathbb{R}^2.
$$

Initial connections should be biased toward nearby neurons.

A candidate target probability:

$$
P(i\rightarrow j)
\propto
\exp\left(
-\frac{\|\mathbf{x}_i-\mathbf{x}_j\|^2}
{2\sigma_{\mathrm{distance}}^2}
\right).
$$

This can be implemented without calculating all pairwise probabilities:

1. Select a source neuron.
2. Sample a point near it.
3. Find one or several neurons near that point.
4. Reject illegal or duplicate connections.
5. Create a weak tentative synapse.

For a small system, brute-force nearby sampling is initially acceptable.

### 8.2 Synaptic permanence

Give every synapse:

$$
q_{ij}\in[0,1]
$$

representing structural permanence.

This is a computational abstraction distinct from functional weight $w_{ij}$.

- Weight describes current influence.
- Permanence describes resistance to pruning.

### 8.3 Tentative, established, consolidated

Suggested conceptual states:

#### Tentative

- newly created;
- weak;
- rapid decay;
- easily removed.

#### Established

- repeatedly active or useful;
- moderate decay;
- survives temporary inactivity.

#### Consolidated

- repeatedly associated with rewarded behavior;
- very slow decay;
- difficult but not impossible to remove.

The state need not be an enum. It can emerge from thresholds over permanence.

### 8.4 Permanence update

One experimental rule:

$$
q_{ij}
\leftarrow
\operatorname{clip}
\left(
q_{ij}
+
\eta_q\max(0,r e_{ij})
+
\eta_a A_{ij}
-
\lambda_q U_{ij},
0,1
\right)
$$

where:

- $A_{ij}$ measures meaningful co-activity;
- $U_{ij}$ measures disuse;
- positive rewarded eligibility stabilizes the synapse.

Avoid strengthening permanence merely because a synapse changes frequently; frequent harmful changes should not consolidate it.

### 8.5 Permanence-dependent decay

$$
w_{ij}
\leftarrow
w_{ij}
\left[
1-\lambda_w(1-q_{ij})
\right].
$$

Low-permanence connections decay quickly. High-permanence connections decay slowly.

### 8.6 Pruning

A connection can be removed when:

$$
q_{ij}<q_{\min}
\quad\land\quad
w_{ij}<w_{\min}
\quad\land\quad
\text{age}_{ij}>\text{minimum trial age}.
$$

The minimum age prevents new connections from being deleted before they have a chance to participate.

### 8.7 Growth schedule

Structural growth should operate much more slowly than spikes:

- fast neural dynamics: every timestep;
- synaptic learning: reward events or every few steps;
- homeostasis: every episode or window;
- growth/pruning: every tens or hundreds of episodes.

A neuron may receive a limited connection budget:

```text
maximum incoming synapses
maximum outgoing synapses
target incoming degree
target outgoing degree
```

When connections are pruned, the neuron gains capacity for new local exploration.

### 8.8 Candidate growth heuristics

Test these separately:

1. **Pure local random search**
2. **Activity-biased search**
3. **Correlated-neighbor search**
4. **Error/reward-biased search**
5. **Novelty-biased search**

Begin with pure local random search. More directed search can later be compared against it.

---

## 9. Global workspace metaphor

### 9.1 Purpose

The workspace is not merely an output layer. It is a small, limited-capacity shared state that allows one or a few representations to become broadly available to the rest of the system.

Core properties:

- many populations can submit candidates;
- candidates compete;
- admission requires sufficient activation or salience;
- admitted content persists temporarily;
- workspace content is broadcast broadly;
- content decays unless refreshed;
- capacity is limited.

### 9.2 Minimal implementation

Represent the workspace as one winning assembly or vector:

$$
W(t).
$$

Candidate $c$ has activation:

$$
a_c(t)
=
\sum_{i\in c}s_i(t)
+
\lambda_c a_c(t-1).
$$

Winner:

$$
c^*=\arg\max_c a_c(t).
$$

Admission condition:

$$
a_{c^*}(t)>\theta_W.
$$

Workspace update:

$$
W(t+1)
=
\gamma W(t)
+
\operatorname{encode}(c^*).
$$

Broadcast:

$$
I_i(t+1)
\mathrel{+}=
B_i W(t).
$$

### 9.3 Avoid unrestricted shared memory

Do not let every neuron freely write arbitrary values into a large shared array. That would remove the competition and bottleneck that make the workspace idea interesting.

A better first version:

- define small candidate assemblies;
- allow one winner at a time;
- broadcast the identity or activity pattern of the winner;
- decay it over several steps.

### 9.4 Workspace and consciousness

The project can borrow the computational motif of global broadcast without claiming to create consciousness.

Use language such as:

> “global-workspace-inspired broadcast mechanism”

rather than:

> “consciousness module.”

Modern global neuronal workspace research remains contested, and recent adversarial testing has not produced a simple final verdict between major consciousness theories.

---

## 10. Inputs, actions, and termination

### 10.1 Input encoding

For simple symbolic tasks, begin with one-hot or population-coded input neurons.

Example symbols:

```text
0 1 2 3 4
+ -
START
END
```

Each symbol is presented for a fixed number of timesteps, followed by a short gap.

Example:

```text
START   5 steps
2       5 steps
+       5 steps
1       5 steps
END     5 steps
settle  30 steps
```

Avoid visual character recognition. Direct symbolic encoding isolates learning and memory from perception.

### 10.2 Action neurons

For answers in a bounded range, create one neuron or small assembly per possible response:

```text
ANSWER_-5
ANSWER_-4
...
ANSWER_10
WAIT
DONE
```

The answer interface may read the most active action assembly over a window.

### 10.3 Initial termination

Use fixed episode duration first:

1. Present the sequence.
2. Allow the system to settle.
3. Count answer-neuron spikes during the final window.
4. Choose the most active answer.
5. Deliver reward.

This removes the need to learn both the answer and when to answer.

### 10.4 Later learned termination

Possible rules:

- terminate when `DONE` fires;
- terminate when an answer remains dominant for $N$ steps;
- terminate when an answer enters the workspace and is followed by `DONE`;
- terminate at a maximum timestep regardless.

Add learned termination only after fixed-duration answering works.

---

## 11. Task curriculum

Arithmetic should be a later task because it combines symbol processing, order, working memory, state updating, and generalization.

### Stage A: Stable autonomous dynamics

No learning task.

Goal:

- activity remains sparse but nonzero;
- no explosive synchronization;
- inhibitory neurons regulate activity;
- different random seeds produce sane behavior.

Measurements:

- mean firing rate;
- distribution of neuron firing rates;
- silent-neuron percentage;
- spikes per timestep;
- membrane-potential distribution;
- excitatory/inhibitory input balance.

### Stage B: Immediate association

Input A should activate action A. Input B should activate action B.

Goal:

- validate encoding;
- validate reward-modulated eligibility;
- validate action reading.

No delay and no structural growth yet.

### Stage C: Delayed association

Present A or B, wait, then require the corresponding answer.

Goal:

- test recurrent memory;
- test eligibility over time;
- test adaptation and workspace usefulness.

### Stage D: Context-dependent mapping

The same symbol requires different responses under different contexts.

Example:

```text
context X + input A -> action 1
context Y + input A -> action 2
```

Goal:

- require integration of multiple signals;
- prevent direct one-input/one-output memorization.

### Stage E: Increment/decrement

Examples:

```text
0 + 1
1 + 1
2 - 1
3 - 1
```

Goal:

- learn simple structured transitions.

### Stage F: One-operation arithmetic

Examples:

```text
2 + 3
4 - 1
```

Keep number range small.

### Stage G: Sequential arithmetic

Examples:

```text
3 + 2 - 1
4 - 1 + 2 - 3
```

This should be attempted only after memory and context tasks work.

---

## 12. Preventing arithmetic memorization

A finite arithmetic dataset is easy to memorize. Evaluate genuine generalization.

Possible splits:

### Held-out combinations

Train on some operand pairs and test unseen pairs.

### Held-out results

Train without examples producing particular results, then test them carefully.

### Length generalization

Train on one or two operations and test on three.

### Number-range generalization

Train on numbers 0–4 and test on 5–6. This is very difficult with one-hot symbols and may require a structured magnitude encoding.

### Equivalent-form testing

Where mathematically valid, test alternate expression forms.

### Noise robustness

- omit occasional input spikes;
- add random background spikes;
- vary symbol duration;
- perturb synaptic transmission.

### Continual learning

Train task A, then task B, then retest A.

This may reveal whether consolidation and structural plasticity reduce forgetting.

---

## 13. Conventional baselines

Compare the brain-inspired system against:

1. A small multilayer perceptron for fixed-length inputs
2. A small recurrent neural network for sequential inputs
3. Optionally a fixed reservoir with trained linear readout
4. Ablated versions of the proposed system

“Same size” has multiple meanings. Report more than one:

- equal trainable parameter count;
- equal neuron/unit count;
- equal approximate memory use;
- equal number of training examples;
- approximate computation or wall-clock budget.

Do not expect the local system to beat backpropagation on raw arithmetic accuracy. More interesting comparisons may include:

- online learning;
- sample efficiency;
- robustness to noise;
- robustness to neuron/synapse deletion;
- catastrophic forgetting;
- sparsity of activity;
- adaptation after environmental change.

---

## 14. Essential ablation experiments

Ablations are necessary to learn which mechanisms matter.

| Model | Mechanisms |
|---|---|
| A | deterministic firing, fixed graph |
| B | stochastic firing |
| C | stochastic synaptic release |
| D | reward-modulated eligibility |
| E | D + adaptive thresholds |
| F | E + excitatory/inhibitory constraint |
| G | F + structural growth/pruning |
| H | G + permanence/consolidation |
| I | H + workspace broadcast |

Additional comparisons:

- fixed graph versus rewiring;
- global random rewiring versus local spatial rewiring;
- no weight decay versus decay;
- no negative reward versus explicit punishment;
- fixed firing threshold versus homeostasis;
- fixed episode duration versus learned termination;
- workspace capacity 1 versus larger capacity.

---

## 15. Metrics and logging

Log enough information to understand failure.

### Per timestep

```text
spike_count
excitatory_spike_count
inhibitory_spike_count
mean_membrane_potential
workspace_state
scheduled_event_count
```

### Per episode

```text
input
expected_answer
chosen_answer
reward
episode_length
total_spikes
mean_firing_rate
number_of_active_neurons
number_of_synapses
mean_weight
mean_permanence
connections_created
connections_pruned
```

### Per training run

```text
accuracy_curve
generalization_accuracy
time_to_threshold_accuracy
activity_cost
seed
configuration
final_graph_statistics
catastrophic_forgetting_score
noise_robustness
damage_robustness
```

Always save the random seed and complete configuration.

---

## 16. Visualization ideas

Visualization should begin early enough to debug dynamics, even if the polished interactive visualization comes later.

### Useful early plots

- raster plot: neuron versus timestep;
- population firing rate over time;
- membrane potential of selected neurons;
- synaptic-weight histogram;
- firing-rate histogram;
- eligibility trace over one episode;
- number of synapses over training;
- accuracy curve;
- workspace winner over time.

### Network visualization

Place neurons at their 2D coordinates.

Possible visual encodings:

- node shape: excitatory/inhibitory/input/action;
- node size: recent firing rate;
- edge thickness: synaptic weight;
- edge opacity: permanence;
- temporary highlight: recent spike transmission;
- workspace border: currently broadcast assembly.

For performance, log snapshots rather than rendering every simulation step.

---

## 17. Recommended implementation stages

### Phase 0 — Experimental scaffold

Build:

- deterministic random-number generator;
- configuration struct;
- run seed recording;
- CSV or JSON logging;
- basic test harness;
- reproducible episode generation.

Exit criterion:

> The same seed and configuration produce identical results.

### Phase 1 — Fixed recurrent spiking simulator

Build:

- neuron state;
- excitatory/inhibitory type;
- fixed sparse graph;
- leaky membrane;
- probabilistic firing;
- refractory period;
- delayed event queue;
- stochastic synaptic release;
- raster logging.

No learning.

Exit criteria:

- activity neither dies nor explodes across many steps;
- inhibitory manipulation visibly changes dynamics;
- deterministic seeds reproduce runs.

### Phase 2 — Homeostasis

Build:

- recent firing-rate estimate;
- adaptive thresholds;
- optional simple weight normalization.

Exit criterion:

> The network returns toward a target activity range after moderate perturbation.

### Phase 3 — Local reward learning

Build:

- pre/post traces;
- eligibility traces;
- reward-modulated weight update;
- immediate association task.

Exit criterion:

> The network learns a two-choice immediate association above chance across multiple random seeds.

### Phase 4 — Delayed learning

Build:

- delayed association task;
- longer eligibility decay;
- recurrent state analysis;
- optional adaptation tuning.

Exit criterion:

> The network retains information across a nonzero delay better than chance.

### Phase 5 — Structural plasticity

Build:

- neuron coordinates;
- synaptic permanence;
- tentative connection creation;
- local target sampling;
- pruning;
- connection budgets.

Exit criterion:

> Connections change over training while activity remains stable, and useful performance is not destroyed.

### Phase 6 — Consolidation

Build:

- permanence-dependent decay;
- separate fast weight and slow structure timescales;
- continual-learning experiment.

Exit criterion:

> Previously useful pathways survive better than unused tentative pathways.

### Phase 7 — Workspace-inspired broadcast

Build:

- candidate assemblies;
- competition;
- ignition threshold;
- limited capacity;
- decay;
- broadcast feedback.

Exit criterion:

> Workspace state can be causally shown to improve at least one delayed or context task.

### Phase 8 — Arithmetic curriculum

Build:

- symbol sequence encoder;
- action assemblies;
- fixed-duration answer reading;
- increment/decrement tasks;
- single-operation arithmetic;
- held-out evaluation.

Exit criterion:

> Performance exceeds memorization baselines on at least one controlled generalization split.

### Phase 9 — Learned termination

Build:

- `DONE` assembly or stable-answer rule;
- timeout behavior;
- termination reward design.

Only add after answer production is reliable.

---

## 18. Suggested first concrete experiment

### Task

Learn a delayed binary association:

```text
input A -> after delay -> choose action A
input B -> after delay -> choose action B
```

### Initial system

```text
neurons:                 100
excitatory:              80
inhibitory:              20
initial connection rate: 5–10%
input neurons:           2 small assemblies
action neurons:          2 small assemblies
episode duration:        fixed
structural plasticity:   disabled
workspace:               disabled
```

### Training episode

1. Reset short-term state, but not weights.
2. Present A or B for several timesteps.
3. Run a delay with no task input.
4. Allow action activity.
5. Read the more active action assembly.
6. Deliver reward.
7. Update eligible synapses.
8. Update homeostasis.
9. Log results.

### Why this is the correct first task

It tests:

- temporal state;
- local learning;
- delayed reward;
- action selection;
- recurrent pathways.

It does not yet require:

- arithmetic;
- structural growth;
- a workspace;
- learned termination;
- complex symbolic representations.

---

## 19. Pseudocode for the full eventual loop

```python
initialize_network(seed)

for episode in training_episodes:
    reset_fast_episode_state()
    task = sample_task()

    for t in 0..<max_steps:
        deliver_scheduled_synaptic_events(t)
        inject_task_input(task, t)
        inject_workspace_broadcast()

        for neuron in neurons:
            update_membrane(neuron)
            update_adaptation(neuron)
            update_refractory(neuron)

        for neuron in neurons:
            if can_fire(neuron):
                neuron.fired = sample_firing(neuron)
            else:
                neuron.fired = false

        for fired_neuron in fired_neurons:
            for synapse in fired_neuron.outgoing:
                if sample_release(synapse):
                    schedule_synaptic_event(synapse)

        update_neural_traces()
        update_synaptic_eligibility()

        if workspace_enabled:
            update_workspace_candidates()
            run_workspace_competition()
            decay_workspace()

        if termination_condition_met():
            break

    answer = read_action()
    reward = task_reward(task, answer)

    apply_reward_modulated_weight_updates(reward)
    update_homeostasis()

    if structural_update_due():
        update_permanence(reward)
        prune_synapses()
        grow_tentative_synapses()

    log_episode()
```

---

## 20. Engineering considerations for Zig

### Data layout

Prefer compact arrays over deeply nested heap objects.

Possible structure-of-arrays layout:

```c
NeuronState:
    potentials: []f32
    thresholds: []f32
    adaptation: []f32
    firing_rates: []f32
    refractory: []u16
    fired: []bool
    neuron_types: []NeuronType
    positions_x: []f32
    positions_y: []f32
```

Synapses:

```c
SynapseStore:
    sources: []NeuronId
    targets: []NeuronId
    weights: []f32
    release_probabilities: []f32
    eligibility: []f32
    permanence: []f32
    delays: []u16
    alive: []bool
```

Adjacency can use:

- compressed index ranges per source;
- per-neuron dynamic lists;
- stable synapse IDs plus free-list reuse.

For early versions, clarity matters more than maximum performance.

### Event queue

For bounded delays, use a ring buffer:

```c
event_buckets[maximum_delay + 1]
```

At timestep $t$:

```c
bucket_index = t % bucket_count
```

Deliver and clear that bucket, then schedule future events into the appropriate future bucket.

### Randomness

Use separate random streams or derived seeds for:

- initialization;
- firing;
- synaptic release;
- task sampling;
- structural growth.

This makes experiments easier to reproduce and isolate.

### Floating-point precision

`f32` should be more than sufficient for this simulator. Numerical precision is unlikely to be the limiting factor.

### Testing

Unit-test:

- membrane leak;
- refractory behavior;
- firing probability bounds;
- delayed event delivery;
- excitatory versus inhibitory sign;
- trace decay;
- eligibility sign;
- clipping;
- pruning criteria;
- deterministic reproduction from a seed.

---

## 21. Likely failure modes

### Everything goes silent

Possible causes:

- thresholds too high;
- membrane leak too strong;
- weak input encoding;
- excessive inhibition;
- low release probabilities;
- homeostatic adaptation too slow.

### Everything fires continuously

Possible causes:

- recurrent excitation too strong;
- thresholds too low;
- inadequate inhibition;
- weak refractory period;
- Hebbian runaway;
- reward update too large.

### Learning remains at chance

Possible causes:

- eligibility decays before reward;
- action neurons are not reachable;
- negative reward erases useful exploration;
- firing is too sparse;
- firing is too noisy;
- task state resets incorrectly;
- input coding is ambiguous.

### It memorizes but does not generalize

Possible causes:

- symbolic representation lacks structure;
- training space is too small;
- too many parameters relative to examples;
- direct pathways memorize complete sequences;
- evaluation split is weak.

### Structural growth destroys learning

Possible causes:

- growth occurs too frequently;
- new weights are too strong;
- pruning is too aggressive;
- permanence updates reinforce noise;
- no grace period for tentative synapses;
- rewiring changes too many connections at once.

### One action always wins

Possible causes:

- biased initialization;
- action competition is absent;
- homeostatic targets are uneven;
- one output has more incoming paths;
- incorrect tie-breaking;
- punishment produces asymmetric collapse.

### Workspace becomes ordinary memory

Possible causes:

- unlimited capacity;
- unrestricted writes;
- no competition;
- no ignition threshold;
- no decay;
- direct access bypasses local circuitry.

---

## 22. Open research questions for this project

- Does stochastic firing improve generalization or merely slow learning?
- Is stochastic release more useful than stochastic firing?
- Can adaptive thresholds preserve useful sparse activity?
- Does local spatial rewiring outperform global random rewiring?
- Do separate weight and permanence variables reduce catastrophic forgetting?
- Does a workspace bottleneck improve compositional tasks?
- Can learned assemblies represent intermediate arithmetic state?
- Does structural plasticity discover reusable subcircuits?
- Can a system trained on short expressions generalize to longer ones?
- How much global reward information is required?
- Can useful learning occur without explicitly labeled output neurons?
- What failure modes appear consistently across random seeds?

---

## 23. What would count as success?

The project does not need to solve arithmetic competitively.

Increasing levels of success:

1. Stable sparse recurrent activity
2. Learnable immediate associations
3. Delayed association above chance
4. Meaningful effect from stochasticity
5. Structural growth that preserves useful pathways
6. Reduced forgetting through permanence/consolidation
7. Workspace-assisted context integration
8. Single-operation arithmetic above memorization controls
9. Generalization to held-out structures
10. Interpretable emergent assemblies or pathways

A negative result can also be useful:

> “Pure reward-modulated timing rules could not reliably learn delayed composition without an additional targeted learning signal.”

That would be a substantive conclusion.

---

## 24. Reading and reference map

### Eligibility traces and local recurrent learning

- Bellec et al. (2020), **A solution to the learning dilemma for recurrent networks of spiking neurons**. Introduces e-prop, combining local eligibility traces with learning signals.
- [https://doi.org/10.1038/s41467-020-17236-y](https://doi.org/10.1038/s41467-020-17236-y)
- Wang et al. (2026), **Model-agnostic linear-memory online learning in spiking neural networks**. A recent local online-learning direction using eligibility-like temporal mechanisms.
- [https://doi.org/10.1038/s41467-026-68453-w](https://doi.org/10.1038/s41467-026-68453-w)
### Structural plasticity

- Fauth & Tetzlaff (2016), **Opposing Effects of Neuronal Activity on Structural Plasticity**. Reviews Hebbian and homeostatic structural plasticity.
- [https://doi.org/10.3389/fnana.2016.00075](https://doi.org/10.3389/fnana.2016.00075)
- Yuan et al. (2023), **Incorporating structural plasticity into self-organization recurrent network with homeostasis**. Directly relevant to combining rewiring, STDP, and homeostatic processes.
- [https://doi.org/10.3389/fnins.2023.1224752](https://doi.org/10.3389/fnins.2023.1224752)
- Bogdan et al. (2018), **Structural Plasticity on the SpiNNaker Many-Core Neuromorphic System**. Includes synaptic rewiring and STDP in a spatially organized system.
- [https://doi.org/10.3389/fnins.2018.00434](https://doi.org/10.3389/fnins.2018.00434)
- Pan et al. (2023), **Adaptive structure evolution and biologically plausible learning in a liquid state machine**.
- [https://doi.org/10.1038/s41598-023-43488-x](https://doi.org/10.1038/s41598-023-43488-x)
### Stochastic synapses and probabilistic computation

- Dutta et al. (2022), **Neural sampling machine with stochastic synapse allows brain-like learning and inference**.
- [https://doi.org/10.1038/s41467-022-30305-8](https://doi.org/10.1038/s41467-022-30305-8)
- Dürst et al. (2022), **Vesicular release probability sets the strength of individual Schaffer collateral synapses**.
- [https://doi.org/10.1038/s41467-022-33565-6](https://doi.org/10.1038/s41467-022-33565-6)
- Buesing et al. (2011), **Neural Dynamics as Sampling: A Model for Stochastic Computation in Recurrent Networks of Spiking Neurons**.
- [https://doi.org/10.1371/journal.pcbi.1002211](https://doi.org/10.1371/journal.pcbi.1002211)
### Homeostasis and intrinsic plasticity

- Zhang et al. (2019), **Information-Theoretic Intrinsic Plasticity for Online Unsupervised Learning in Spiking Neural Networks**.
- [https://doi.org/10.3389/fnins.2019.00031](https://doi.org/10.3389/fnins.2019.00031)
- Zhang et al. (2024), **Composing recurrent spiking neural networks using locally distributed intrinsic plasticity**.
- [https://doi.org/10.3389/fnins.2024.1412559](https://doi.org/10.3389/fnins.2024.1412559)
### Global workspace

- Cogitate Consortium (2025), **Adversarial testing of global neuronal workspace and integrated information theories of consciousness**.
- [https://doi.org/10.1038/s41586-025-08888-1](https://doi.org/10.1038/s41586-025-08888-1)
- Martín-Signes et al. (2024), **Streams of conscious visual experience**. Contains a concise description of global broadcast and competition motifs.
- [https://doi.org/10.1038/s42003-024-06593-9](https://doi.org/10.1038/s42003-024-06593-9)
> [!caution]
> Global neuronal workspace theory is a theory of conscious access, not an established recipe for an AI output layer. Use its competition-and-broadcast motif as an engineering hypothesis.

### Learned plasticity rules

- Shervani-Tabar et al. (2023), **Meta-learning biologically plausible plasticity rules with random feedback pathways**.
- [https://doi.org/10.1038/s41467-023-37562-1](https://doi.org/10.1038/s41467-023-37562-1)
This may become interesting much later: instead of manually choosing a plasticity equation forever, an outer optimization process could search for local rules. It should not be part of the first implementation.

---

## 25. Vocabulary

| Term                           | Meaning                                                                                                          |
| ------------------------------ | ---------------------------------------------------------------------------------------------------------------- |
| **Spike**                      | A discrete neuron-firing event.                                                                                  |
| **Membrane potential**         | The neuron’s current internal activation state.                                                                  |
| **Refractory period**          | A short period after a spike during which the neuron cannot fire again.                                          |
| **STDP**                       | Spike-timing-dependent plasticity; synaptic change based on relative spike timing.                               |
| **Eligibility trace**          | A decaying local record that a synapse may recently have contributed to activity.                                |
| **Three-factor learning rule** | A plasticity rule involving presynaptic activity, postsynaptic activity, and a modulatory signal such as reward. |
| **Homeostasis**                | Processes that keep neural activity within a functional range.                                                   |
| **Structural plasticity**      | Creation, stabilization, and removal of physical connections.                                                    |
| **Consolidation**              | Slow stabilization of learning so that it becomes resistant to decay or interference.                            |
| **Neural assembly**            | A group of neurons whose coordinated activity represents or performs something.                                  |
| **Global workspace**           | A limited-capacity mechanism in which selected information becomes broadly broadcast.                            |
| **Ablation**                   | Removing or disabling one mechanism to determine its causal contribution.                                        |

---

## 26. Immediate next action

Implement **Phase 0 and Phase 1 only**.

The first coding target is:

> A reproducible simulator of 100 fixed-graph stochastic spiking neurons with excitatory/inhibitory types, membrane leak, refractory periods, probabilistic synaptic transmission, delayed events, and raster logging.

Do not implement learning, growth, arithmetic, or the workspace until this network can maintain a stable, interpretable activity regime.

### Phase 1 completion checklist

- [ ] Same seed produces identical spike history
- [ ] Excitatory spikes increase target potential
- [ ] Inhibitory spikes decrease target potential
- [ ] Membrane potential leaks toward rest
- [ ] Refractory period prevents immediate refiring
- [ ] Synaptic release probability behaves statistically correctly
- [ ] Delays deliver events at the expected timestep
- [ ] Activity remains nonzero for a useful interval
- [ ] Activity does not explode into constant firing
- [ ] Raster data can be plotted or inspected
- [ ] Configuration and seed are saved with each run

---

## 27. Project mantra

> **Build one mechanism, prove that it behaves correctly, measure its effect, and only then add the next mechanism.**


## NOTES

1. Only the readout learns, not the recurrent reservoir. This is the DEC-008 design choice for reliability. The full recurrent net learning is a bigger scope (and not what "immediate association" requires).
2. Learning is exploration-limited (~600-episode plateau then takeoff). Raising the learning rate barely helped. Faster learning would need better credit assignment — winner-take-all/lateral inhibition between action groups, or crediting only the chosen action. Not needed for the criterion, but the natural lever if you later want snappier learning or a working-memory delay version (the doc's delay step, which I kept at 0 for "immediate").