---
id: cff-2gev
status: open
deps: [cff-hp3y, cff-4oq8, cff-b47g]
links: [cff-weff]
created: 2026-07-10T03:34:54Z
type: feature
priority: 3
assignee: coopwilliams
---
# Continual transition curiosity under nonstationarity

## Background in Plain English

The current GFE-TM objective values a transition when observing its outcome is
expected to teach the agent about uncertain dynamics, so this value naturally
shrinks as a stationary transition posterior accumulates evidence. A changing
environment needs an explicit model that lets old evidence lose force or lets a
new regime become plausible; multiplying curiosity by a one-way schedule does
not provide that model. Closed, unmerged PRs #42 and #43 preserve the earlier
annealing proposal, but their run stopped at a mixed-GFE limit cycle before any
comparative cell and therefore supplied no evidence about annealing. In the
corrected flat Hallway study, curiosity achieved 0.16 task success and
visitation-weighted transition KL 2.650, while the learn-only arm achieved 0.76
and 2.534, so those results are negative evidence about that flat stationary
model rather than a prescription for nonstationary learning. This ticket
selects and derives one generative nonstationarity mechanism, integrates it
with declared transition learnables, and demonstrates that transition
uncertainty and directed information seeking return after a controlled change
without altering the default stationary semantics.

## Outcomes / Acceptance Criteria

### 1. The v1 model is explicit and derived

- [ ] 1.1 A reviewable design record selects exactly one v1 mechanism for
  nonstationary transition learning.
- [ ] 1.2 The design states the environmental assumption that makes historical
  transition evidence less informative.
- [ ] 1.3 The design defines every posterior state variable maintained by the
  selected mechanism.
- [ ] 1.4 The design derives the transition-posterior update from the stated
  model rather than from a curiosity multiplier.
- [ ] 1.5 The design states whether each update occurs per observation, per
  tick, per episode, or at another declared boundary.
- [ ] 1.6 The design distinguishes observing more evidence for the current
  dynamics from revising the belief that the current dynamics still apply.
- [ ] 1.7 The design defines a stationary limit in which the probability of a
  dynamics change is zero or the selected mechanism's equivalent condition
  holds.
- [ ] 1.8 The stationary-limit derivation reduces to the repository's existing
  Dirichlet transition update.
- [ ] 1.9 Any approximation introduced by the mechanism has a named exact
  reference quantity or an independently checkable bound.

### 2. Stationary GFE-TM behavior remains the default

- [ ] 2.1 A transition learnable without an explicit nonstationarity
  declaration retains the current accumulating Dirichlet posterior.
- [ ] 2.2 The default planner retains the complete GFE-TM parameter-novelty
  channel from `docs/DERIVATIONS.md` D1.
- [ ] 2.3 No elapsed-time, episode-count, or accumulated-count coefficient
  scales the default GFE-TM objective.
- [ ] 2.4 On a frozen validation corpus, stationary-limit posterior
  concentrations match the pre-change update within preregistered dtype-aware
  tolerances.
- [ ] 2.5 On the same frozen corpus, stationary-limit GFE-TM messages match the
  pre-change implementation up to log-message gauge within preregistered
  dtype-aware tolerances.
- [ ] 2.6 On the same frozen corpus, stationary-limit node-local GFE values
  match the pre-change implementation within preregistered dtype-aware
  tolerances.
- [ ] 2.7 On the same frozen corpus, stationary-limit policy rankings match the
  pre-change implementation, with ties defined before execution.

### 3. The selected model restores useful transition uncertainty after change

- [ ] 3.1 The evaluation declares which transition parameter blocks are
  changed by the environment.
- [ ] 3.2 The evaluation declares which transition parameter blocks remain
  unchanged.
- [ ] 3.3 Before the controlled change, a preregistered saturation diagnostic
  is satisfied for the changed blocks.
- [ ] 3.4 After the controlled change, the selected uncertainty observable
  increases for the changed blocks by its preregistered threshold.
- [ ] 3.5 Unchanged blocks do not undergo a global reset unless the selected
  generative model entails that reset.
- [ ] 3.6 The preregistration names every reset entailed by the selected model.
- [ ] 3.7 After the controlled change, the GFE-TM expected-information-gain
  contribution increases for at least one action that probes a changed block.
- [ ] 3.8 A matched no-transition-epistemics arm visits the informative changed
  blocks less often than the selected model by the preregistered behavioral
  threshold.
- [ ] 3.9 A matched stationary-posterior arm adapts its transition prediction
  more slowly than the selected model by the preregistered recovery threshold.
- [ ] 3.10 The post-change transition-prediction metric is computed from held-out
  transitions or another independent source that is fixed before execution.
- [ ] 3.11 Task success is reported separately from transition-prediction
  quality.
- [ ] 3.12 A failed behavioral threshold is reported as a failed claim rather
  than reinterpreted as successful curiosity.

### 4. The declaration, runtime state, and factored models compose

- [ ] 4.1 The selected mechanism is declared in the authoritative serialized
  template structure
  for each participating transition learnable.
- [ ] 4.2 The declaration round-trips through JSON without semantic loss.
- [ ] 4.3 The declaration participates in the structure hash.
- [ ] 4.4 Plate scoping preserves the declaration on copied learnables.
- [ ] 4.5 Composition cannot silently merge incompatible nonstationarity
  declarations.
- [ ] 4.6 The factored rollout builder accepts the mechanism for an arbitrary
  named transition block without relying on corridor-specific names.
- [ ] 4.7 Factors tied through one transition blob contribute to one shared
  posterior state.
- [ ] 4.8 Independent transition learnables maintain independent posterior
  states.
- [ ] 4.9 Planning consumes the updated posterior parameters without requiring
  a structure-changing recompilation.
- [ ] 4.10 The current checkpoint format restores every mechanism-owned state
  variable with the equality semantics declared by its schema.
- [ ] 4.11 Given the same next observation and PRNG stream, a restored agent
  produces the same next posterior state as an uninterrupted agent.
- [ ] 4.12 Checkpoint restoration rejects mechanism state whose declaration is
  incompatible with the compiled model.
- [ ] 4.13 Checkpoint restoration rejects a mechanism-state array whose shape is
  incompatible with the compiled model.
- [ ] 4.14 The legacy checkpoint decoder either supplies the documented
  stationary default or rejects the missing mechanism state with a migration
  error.
- [ ] 4.15 Bundle loading reconstructs the mechanism declaration from the
  serialized template and external parameter data.
- [ ] 4.16 Any stochastic update accepts an explicit JAX PRNG key.
- [ ] 4.17 A checkpoint includes the PRNG stream when deterministic resumption
  depends on that stream.

### 5. Numerical and support behavior is defined on hostile inputs

- [ ] 5.1 A template-declared structural zero remains exactly impossible after
  every nonstationary update.
- [ ] 5.2 A structurally allowed transition with zero observations retains the
  positive support implied by its proper prior.
- [ ] 5.3 A mechanism that permits the structural support itself to change
  represents that possibility in its generative state rather than with a
  numeric floor.
- [ ] 5.4 Every required prior is checked for propriety before compilation or
  state restoration.
- [ ] 5.5 An improper transition prior fails with an error that names the
  learnable and violated condition.
- [ ] 5.6 A declared vacuous-prior representation either maps to a proper
  distribution or fails with a documented validation error.
- [ ] 5.7 A vacuous incoming categorical message follows the existing GFE-TM
  uniform-message semantics.
- [ ] 5.8 An infeasible exact-support intersection remains a loud divergent or
  invalid runtime result.
- [ ] 5.9 Every floor, clamp, ridge, or stopping tolerance derives from the
  active dtype and names the invariant it protects.
- [ ] 5.10 The float32 hostile case produces finite posterior state.
- [ ] 5.11 The float32 hostile case produces finite planner messages.
- [ ] 5.12 The float32 hostile case produces finite free energy.
- [ ] 5.13 Non-finite posterior state, residual, message, or free energy is
  classified as divergence.
- [ ] 5.14 Any new iterative inner solve distinguishes convergence, budget
  exhaustion, and divergence.
- [ ] 5.15 Any new Newton-type inner solve uses globalization unless the design
  proves its domain is globally safe.
- [ ] 5.16 The hostile-scale case contains at least ten times as many
  transition parameters as the smallest behavioral fixture.
- [ ] 5.17 The hostile-scale case changes the environment only after the
  preregistered saturation diagnostic has fired.

### 6. Every affected execution surface is verified

- [ ] 6.1 The implementation handoff walks the `New IR field` row in
  `docs/ARCHITECTURE.md` and records each surface as changed or structurally
  irrelevant, with a reason.
- [ ] 6.2 The implementation handoff walks the `Learnable or checkpoint
  feature` row in `docs/ARCHITECTURE.md` and records each surface as changed or
  structurally irrelevant, with a reason.
- [ ] 6.3 If the selected mechanism adds a factor or rule family, the handoff
  walks the `New factor or rule family` row and records each surface as changed
  or structurally irrelevant, with a reason.
- [ ] 6.4 If the selected mechanism adds a numerical or status invariant, the
  handoff walks the `Runtime numerical or status invariant` row and records
  each surface as changed or structurally irrelevant, with a reason.
- [ ] 6.5 Host and JIT update paths have equivalent posterior-state results on
  the focused validation corpus.
- [ ] 6.6 Single-instance and `vmap` execution have equivalent posterior-state
  results on the focused validation corpus.
- [ ] 6.7 If a GFE-TM rule changes, flooding and scheduled execution agree on
  messages within preregistered tolerances.
- [ ] 6.8 If a GFE-TM rule changes, flooding and scheduled execution agree on
  free energy within preregistered tolerances.
- [ ] 6.9 If a GFE-TM rule changes, flooding and scheduled execution report the
  same runtime status.
- [ ] 6.10 If a GFE-TM rule changes, its update and node-local energy are changed
  and reviewed together.
- [ ] 6.11 If a GFE-TM rule changes, exact-support behavior is tested through
  the host support check.
- [ ] 6.12 If a GFE-TM rule changes, exact-support behavior is tested through
  the JIT support check.
- [ ] 6.13 Current and legacy checkpoint decoding follow the migration behavior
  declared by criterion 4.14.
- [ ] 6.14 The focused suite includes `tests/test_learnables.py`.
- [ ] 6.15 The focused suite includes `tests/test_agent_spec.py`.
- [ ] 6.16 The focused suite includes `tests/test_agent.py`.
- [ ] 6.17 The focused suite includes `tests/test_bundle.py`.
- [ ] 6.18 The focused suite includes the affected GFE-TM tests in
  `tests/test_b_epistemics.py`.
- [ ] 6.19 The focused suite includes the affected factored-rollout tests in
  `tests/test_factored_rollout.py`.
- [ ] 6.20 The final code change is followed by one passing default short-suite
  run through the repository dashboard workflow.

### 7. The replacement study is frozen before its evidence is inspected

- [ ] 7.1 The preregistration identifies the serialized model artifact used by
  each arm.
- [ ] 7.2 The preregistration records the structure hash of each arm.
- [ ] 7.3 The preregistration fixes the nonstationarity mechanism.
- [ ] 7.4 The preregistration fixes every nonstationarity hyperparameter.
- [ ] 7.5 The preregistration fixes the stationary-posterior arm.
- [ ] 7.6 The preregistration fixes the no-transition-epistemics arm.
- [ ] 7.7 The preregistration fixes the selected-model arm.
- [ ] 7.8 The preregistration fixes the environmental transition change.
- [ ] 7.9 The preregistration fixes the time of the environmental change.
- [ ] 7.10 The preregistration fixes the saturation diagnostic.
- [ ] 7.11 The preregistration fixes the saturation observation window.
- [ ] 7.12 The preregistration fixes the seed set.
- [ ] 7.13 The preregistration fixes the inference budget.
- [ ] 7.14 The preregistration fixes the interaction budget.
- [ ] 7.15 The preregistration fixes the valid-run conditions in terms of
  `RunStatus` and outer-solver convergence.
- [ ] 7.16 The preregistration fixes the changed-block uncertainty observable
  and its acceptance threshold.
- [ ] 7.17 The preregistration fixes the directed-information-seeking metric and
  its acceptance threshold.
- [ ] 7.18 The preregistration fixes the independent transition-prediction
  metric and its acceptance threshold.
- [ ] 7.19 The preregistration fixes the task-success metric and its acceptance
  threshold.
- [ ] 7.20 The preregistration fixes the posterior-update wall-time metric.
- [ ] 7.21 The preregistration fixes the planning wall-time metric.
- [ ] 7.22 The preregistration fixes the posterior-update peak-memory metric.
- [ ] 7.23 The preregistration fixes the planning peak-memory metric.
- [ ] 7.24 The preregistration fixes the objective tolerance for an optimized
  path.
- [ ] 7.25 The preregistration fixes the message tolerance for an optimized
  path.
- [ ] 7.26 The preregistration fixes the policy-ordering tolerance for an
  optimized path.
- [ ] 7.27 The preregistration is committed before any comparative result is
  inspected.
- [ ] 7.28 Any post-freeze protocol change creates a new protocol version.
- [ ] 7.29 A post-freeze protocol change invalidates every earlier cell affected
  by that change.
- [ ] 7.30 Every valid preregistered cell is reported with raw traces sufficient
  to recompute its summary.
- [ ] 7.31 Every valid preregistered cell records its code revision.
- [ ] 7.32 Every valid preregistered cell records its model structure hash.
- [ ] 7.33 Every valid preregistered cell records its seed.
- [ ] 7.34 Every valid preregistered cell records its complete experiment
  configuration.
- [ ] 7.35 Invalid, divergent, and budget-exhausted cells are reported by their
  actual status.

## Methodological Suggestions

### Model selection and derivation

Start from `docs/DERIVATIONS.md` D1: for one Dirichlet transition column,
GFE-TM's parameter novelty is the exact expected KL from the current posterior
to the one-observation posterior. Keep that node-local objective intact and
make the transition posterior supplied to it represent the selected
nonstationary model. Compare at least these model classes before choosing v1:
a generatively justified power-prior or evidence-discount model, explicit
change-point inference, and latent transition regimes. Compare their
assumptions, posterior state, identifiability, stationary limit, support
semantics, and computational scaling; implementation convenience is not a
scientific reason to choose one.

Prefer locating the mechanism in the declared learnable update when the model
still presents a Dirichlet or mixture-of-Dirichlets posterior to GFE-TM. Change
`gfe_tm_update` or `gfe_tm_energy` only when the derivation shows that the
selected posterior cannot be represented by their existing raw-concentration
contract. Document any new statistics or approximation in
`docs/DERIVATIONS.md`, update the learnable/checkpoint contract in
`docs/ARCHITECTURE.md`, document public construction and cadence in `README.md`,
and state any accepted approximation in `docs/LIMITATIONS.md`.

### Change-path and execution-surface handoff

The applicable `docs/ARCHITECTURE.md` rows are `New IR field` and `Learnable or
checkpoint feature`: the nonstationarity assumption must be serialized, and
its posterior state must move through learning, rollover, checkpointing, and
bundling. Walk `LearnableSpec` or the selected replacement IR authority,
validation, JSON decoding/encoding, structure hashing, plate scoping,
composition/edit behavior, compiler `_resolve_learnables`,
`Arena.learn_plans`, `member_of`, engine readouts and parameter writes,
`RollingAgent` cadence, current and legacy checkpoints, and bundles. Also walk
the `New factor or rule family` row if the selected model introduces one, and
the `Runtime numerical or status invariant` row if it introduces an iterative
solver or new failure state.

Review the duplicated pairs explicitly: host/JIT update logic,
flooding/scheduled inference when rules change, single-instance/`vmap`
execution, and current/legacy checkpoint decoding. Mark a paired surface
structurally irrelevant with a reason when the selected mechanism cannot reach
it. Keep posterior
arrays as runtime/checkpoint data rather than inline template blobs, and make
the serialized template plus referenced data sufficient to reconstruct the
compiled arena.

### Behavioral and scientific design

Use the factored position-by-mode corridor delivered by cff-b47g rather than
the flat Hallway as the principal nonstationary fixture. Arrange the change so
that some transition blocks change and some remain stationary; this makes a
selective posterior response distinguishable from an indiscriminate reset.
Hold pragmatic goals, observations, action costs, seeds, and inference budgets
constant across the selected-model, stationary-posterior, and
no-transition-epistemics arms.

Measure four claims separately: posterior uncertainty in changed blocks,
action visitation that identifies those blocks, held-out transition prediction,
and task success. Record planner and learning wall time plus peak memory as
engineering measurements rather than treating lower compute as evidence of
better curiosity. Use cff-weff's truthful solver statuses and reject every
comparative cell that fails its preregistered convergence condition.

### Evidence anchors and dependency state

- `docs/DERIVATIONS.md` D1 defines the current exact Dirichlet
  expected-information-gain channel and names `test_novelty_identity` as its
  executable anchor.
- `docs/LIMITATIONS.md` H7 records that indefinitely accumulated Dirichlet
  evidence adapts slowly after environmental change and that ad hoc count
  decay loses ordinary stationary Bayesian semantics.
- `docs/LIMITATIONS.md` S1c records GFE-TM's latent-next-state optimism, its
  approximate fourfold planning cost, and the corrected flat Hallway result.
- cff-hp3y supplies the corrected GFE-TM cavity and node-local ledger on which
  the Hallway result depends.
- cff-4oq8 supplies the exact behavioral-epistemics controls and separates
  relevance, satiation, action cost, and observation quality.
- cff-b47g supplies the serialized factored corridor, tied transition
  learnables, and scale fixture.
- cff-weff supplies the linked mixed-GFE globalization and truthful run-status
  behavior; it is closed and is not a dependency of this terminal ticket.
- [PR #42](https://github.com/engimatic-systems/cffg/pull/42) is the closed,
  unmerged preregistration of the superseded monotone-annealing study.
- [PR #43](https://github.com/engimatic-systems/cffg/pull/43) is the closed,
  unmerged implementation whose run ended before comparative cells.

Legend: `-->` means prerequisite to dependent; `~~~` means related evidence
without a dependency edge.

```text
[x] cff-hp3y --+
[x] cff-4oq8 --+--> [ ] cff-2gev (ready; terminal)
[x] cff-b47g --+

[x] cff-weff ~~~~~> [ ] cff-2gev (link only)

[ ] cff-2gev --> no dependent roadmap feature or epic
```

## Future work

No future work is delegated by this ticket. The three declared prerequisites
are closed, cff-weff is closed related evidence rather than a blocker, and the
current dependency graph names no downstream feature or epic. Alternative
nonstationarity models not selected for v1 remain rejected alternatives unless
new evidence justifies a separately scoped ticket.

## Definitions

- **GFE-TM**: the transition-mixture generalized-free-energy composite derived
  in `docs/DERIVATIONS.md` D1.
- **Transition curiosity**: action preference contributed by expected
  information about future state transitions or their parameters.
- **Parameter novelty**: the GFE-TM conditional mutual-information term that,
  for a Dirichlet column, equals the expected KL from the current posterior to
  the posterior after one transition observation.
- **Generative nonstationarity mechanism**: a probabilistic model in which the
  environment's transition law may change over time.
- **Stationary limit**: the parameter setting under which the transition law
  does not change and ordinary accumulating Bayesian evidence is recovered.
- **Evidence discounting**: an update that reduces the influence of older
  sufficient statistics according to a declared probabilistic assumption.
- **Power prior**: a prior construction that raises a previous posterior
  density to a declared power before incorporating new evidence.
- **Change point**: a latent event indicating that the transition law has
  changed.
- **Latent regime**: an unobserved state that selects one of several transition
  laws.
- **Novelty annealing**: external scaling of an epistemic term according to
  elapsed time or accumulated evidence; it is not the selected mechanism merely
  because posterior uncertainty declines naturally.
- **Transition parameter block**: one named transition learnable and the tied
  factors whose sufficient statistics update that posterior.
- **Structural zero**: a transition outcome declared impossible by model
  structure rather than merely unobserved in data.
- **Proper Dirichlet prior**: a concentration tensor with finite, strictly
  positive entries on every structurally allowed outcome.
- **Vacuous categorical message**: a uniform incoming message that contributes
  no relative log preference.
- **Saturation diagnostic**: a preregistered observable and threshold showing
  that the relevant stationary posterior has accumulated enough evidence
  before the environment changes.
- **Changed-block uncertainty observable**: the preregistered scalar used to
  measure uncertainty for transition blocks changed by the environment.
- **Directed information seeking**: action visitation that preferentially
  gathers observations capable of distinguishing the changed transition law.
- **Independent transition-prediction metric**: a prediction score computed
  from data not folded into the posterior value being evaluated.
- **Policy ordering**: the ranking of candidate policies by their declared
  free-energy score, with preregistered handling of numerical ties.
- **RunStatus**: the runtime classification that distinguishes convergence,
  budget exhaustion, and divergence.
- **Log-message gauge**: an additive constant that leaves a normalized
  categorical message unchanged.
- **Hostile-scale case**: a fixture with at least ten times the transition
  parameter count of the smallest behavioral fixture.

## Notes

### 2026-08-07 — Historical evidence and rewrite audit

The bounded historical ambiguity slice returned 21 candidates across 33
occurrences and classified zero occurrences as problems. The post-rewrite scan
returned 45 candidates across 290 occurrences. Manual context review found no
ambiguity requiring another edit; the mapping below records each retained
meaning.

Retained candidate meanings:

- `fixed`: a preregistered artifact is pinned.
- `integrate`: internal components are wired together.
- `preserve`: a historical artifact is retained or a semantic invariant is
  maintained.
- `public`: the documented consumer contract is meant.
- `reusable`: the identified corpus or input artifact is reused rather than
  merely resembling another artifact.
- `same`: the named corpus or input is reused, or values are equal under the
  stated comparison.
- `implement`: the existing baseline, the historical component, or planned
  component wiring is meant according to the surrounding sentence.
- `provide`: an internal model mechanism supplies the behavior.
- `support`: the feasible probability set is meant.
- `complete`: the named field set is exhaustive, or the review marker records
  ticket-scope completion.
- `deliver`: a closed prerequisite supplied the named output.
- `ready`: no declared dependency blocks the ticket.
- `remaining`: a state or classification continues to hold.
- `approximate`: a named mathematical approximation or the explicitly rough
  fourfold cost estimate is meant.
- `close`: a pull request or ticket reached work-item completion.
- `sufficient`: the mathematical statistic is meant, or a named artifact meets
  the stated reconstruction or recomputation threshold.
- `check`: a named validator or runtime support check is meant.
- `converge`: the runtime convergence classification is meant.
- `correct`: a previously identified defect was repaired.
- `equivalent`: mathematical or value equivalence under the stated corpus is
  meant.
- `estimate`: the explicitly rough fourfold computational-cost quantity is
  meant.
- `evidence`: transition observations, statistical evidence, empirical support,
  or a source relationship is meant according to the surrounding sentence.
- `exact`: a cardinality, mathematical identity, literal zero, or literal value
  is meant.
- `match`: numeric value equality or a controlled experimental configuration is
  meant.
- `pass`: test success is meant.
- `test`: a named executable check or focused validation file is meant.
- `validate`: a validation corpus or validation operation is meant.
- `verify`: acceptance-coverage checking is meant.
- `accumulate`: a posterior evidence or count update is meant.
- `infer`: an inference procedure or its budget is meant.
- `learn`: posterior evidence accumulation or runtime learning is meant.
- `learnable`: the concrete serialized IR object is meant.
- `recover`: stationary-update equality or post-change adaptation is meant.
- `rollover`: the hyperparameter-to-runtime-parameter operation is meant.
- `scale`: objective multiplication, computational growth, or fixture size is
  meant according to the surrounding sentence.
- `small`: the relative domain size of a named fixture is meant.
- `expected`: mathematical expectation is meant.
- `may`: a possibility permitted by the generative model is meant.
- `any` and `every`: the stated scope is universal.
- `at least`: the bound is inclusive.
- `only`: a hard exclusion, temporal gate, experimental-arm label, or link-only
  relationship is meant according to the surrounding sentence.
- `up to`: equivalence modulo the named log-message gauge is meant.
- `current`: the present repository contract, the posterior at the stated time,
  or the current-checkpoint side of an explicit current/legacy pair is meant.
- `existing`: an already-present implementation or semantic contract is meant
  without a version claim.

Intelligible-ticket review: complete.
