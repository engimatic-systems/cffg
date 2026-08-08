---
id: cff-pyy9
status: open
deps: [cff-1d6t, cff-gjjk, cff-d56d, cff-mjku]
links: []
created: 2026-07-05T22:12:32Z
type: epic
priority: 3
assignee: coopwilliams
---
# M7: Hierarchical couplings and shallow contingent planning

## Background in Plain English

CFFG can already build a parent-dependent transition inside one model, but it
cannot yet run two coupled models on different clocks or return a plan whose
later action depends on a future observation. This epic adds a two-level model
in which a slow categorical parent selects the transition regime of a faster
child and receives child evidence without counting its own prior twice. It also
adds a depth-two policy template that chooses a later action for each possible
observation rather than collapsing those choices into one action marginal. The
closed T-maze, learned-goal, and schedule tickets provide the mathematical and
runtime foundations; the open cff-wbhu → cff-9xmq → cff-mjku chain provides the
compiled loop, balanced fan-out, and observation-branch construction needed by
this work. Completion means both capabilities have exact small-model oracles,
honest free-energy and runtime status, hostile-input coverage, and end-to-end
behavioral evidence.

## Outcomes / Acceptance Criteria

### 1. The two-level model has a declared mathematical contract

- [ ] 1.1 The v1 hierarchy contains exactly one categorical parent level and
  one child level.
- [ ] 1.2 The parent state selects a named child `transition_mixture` regime.
- [ ] 1.3 The design writes the joint probability model for one parent period.
- [ ] 1.4 The design defines the message sent from the parent to the child.
- [ ] 1.5 The design derives the cavity message sent from the child to the
  parent.
- [ ] 1.6 The child-to-parent cavity excludes the incoming parent message.
- [ ] 1.7 The design defines temporal thickness as the number of child
  environment ticks in one parent period.
- [ ] 1.8 The design distinguishes an environment tick from an inference
  iteration.
- [ ] 1.9 The design distinguishes an inference iteration from a posterior
  rollover.
- [ ] 1.10 Temporal thickness equal to one reduces to the declared same-clock
  reference model.
- [ ] 1.11 The formal derivation states the variational factorization used at
  the inter-level boundary.
- [ ] 1.12 Every approximation introduced by the factorization has a named
  exact reference or bound.

### 2. The coupled free-energy ledger is complete

- [ ] 2.1 The derivation defines one joint BFE or GFE for the coupled hierarchy.
- [ ] 2.2 The joint ledger includes each parent node-local term once.
- [ ] 2.3 The joint ledger includes each child node-local term once.
- [ ] 2.4 The joint ledger includes the inter-level edge-entropy correction
  once.
- [ ] 2.5 The joint ledger excludes every empirical-prior coupling already
  represented by an inter-level message.
- [ ] 2.6 On a tiny categorical hierarchy, the joint ledger matches direct
  enumeration within a preregistered dtype-aware tolerance.
- [ ] 2.7 On the tiny hierarchy, the parent marginal matches direct enumeration
  within a preregistered dtype-aware tolerance.
- [ ] 2.8 On the tiny hierarchy, the child marginal matches direct enumeration
  within a preregistered dtype-aware tolerance.
- [ ] 2.9 A sum of per-level free energies is not labeled a joint objective
  unless criterion 2.6 establishes that identity.
- [ ] 2.10 The outer coupled iteration uses the derived joint free energy as its
  convergence signal.

### 3. Hierarchy structure and boundary bindings are explicit

- [ ] 3.1 A public hierarchy declaration names the parent template.
- [ ] 3.2 The declaration names the child template.
- [ ] 3.3 The declaration names the parent-to-child port binding.
- [ ] 3.4 The declaration names the child-to-parent port binding.
- [ ] 3.5 The declaration records temporal thickness.
- [ ] 3.6 The declaration accepts arbitrary valid factor and port names.
- [ ] 3.7 A parent-to-child cardinality mismatch fails before inference begins.
- [ ] 3.8 A child-to-parent cardinality mismatch fails before inference begins.
- [ ] 3.9 A zero temporal thickness fails validation.
- [ ] 3.10 A negative temporal thickness fails validation.
- [ ] 3.11 A nonintegral temporal thickness fails validation.
- [ ] 3.12 The hierarchy declaration round-trips without semantic loss.
- [ ] 3.13 The hierarchy structure participates in its artifact hash.
- [ ] 3.14 Changing temporal thickness changes the hierarchy artifact hash.
- [ ] 3.15 Large numeric parameter arrays remain external data references.
- [ ] 3.16 The two templates and hierarchy declaration reconstruct the runtime
  without hidden process state.
- [ ] 3.17 Hierarchy configuration errors name the invalid level or boundary.

### 4. Nested execution respects both clocks and every runtime status

- [ ] 4.1 The child reaches its declared convergence condition before its
  outbound boundary message is consumed.
- [ ] 4.2 A child `BUDGET_EXHAUSTED` result prevents the parent update for that
  boundary exchange.
- [ ] 4.3 A child `DIVERGED` result prevents the parent update for that boundary
  exchange.
- [ ] 4.4 A child `INFEASIBLE` result prevents the parent update for that
  boundary exchange.
- [ ] 4.5 The hierarchy result retains the child's actual runtime status.
- [ ] 4.6 The hierarchy result retains the parent's actual runtime status.
- [ ] 4.7 The hierarchy result distinguishes coupled convergence from budget
  exhaustion.
- [ ] 4.8 The hierarchy result distinguishes coupled convergence from
  divergence.
- [ ] 4.9 The hierarchy result distinguishes coupled convergence from
  infeasibility.
- [ ] 4.10 A non-finite joint free energy is classified as divergence.
- [ ] 4.11 A non-finite inter-level residual is classified as divergence.
- [ ] 4.12 The modeled parent regime cannot transition during child ticks within
  one parent period.
- [ ] 4.13 The parent advances once after the declared number of completed
  child ticks.
- [ ] 4.14 Child transition statistics accumulate once per executed child
  transition.
- [ ] 4.15 Inference sweeps do not create child transition statistics.
- [ ] 4.16 Parent statistics accumulate once per completed parent period.
- [ ] 4.17 Re-evaluating an inter-level message does not create parent
  statistics.
- [ ] 4.18 A goal-hyperprior child rolls its staged goal posterior once at its
  declared episode boundary.
- [ ] 4.19 Nested inference sweeps do not roll a staged goal posterior.
- [ ] 4.20 Under identical completed trials, nested goal-hyperprior state
  matches the standalone cff-gjjk update.

### 5. Nested execution is reproducible and resumable

- [ ] 5.1 A checkpoint records the parent inference state.
- [ ] 5.2 A checkpoint records the child inference state.
- [ ] 5.3 A checkpoint records the parent-period counter.
- [ ] 5.4 A checkpoint records the child-tick counter.
- [ ] 5.5 A checkpoint records pending inter-level messages.
- [ ] 5.6 A checkpoint records pending learned-goal rollover state.
- [ ] 5.7 A stochastic hierarchy accepts an explicit JAX PRNG key.
- [ ] 5.8 A stochastic hierarchy checkpoint records its PRNG stream.
- [ ] 5.9 Given the same next observation and PRNG stream, a restored hierarchy
  produces the same next posterior state as uninterrupted execution.
- [ ] 5.10 Current checkpoint decoding rejects state from an incompatible
  hierarchy declaration.
- [ ] 5.11 Legacy checkpoint decoding follows a documented migration or refusal
  rule.
- [ ] 5.12 Bundle loading reconstructs both model levels.
- [ ] 5.13 Bundle loading reconstructs their boundary declaration.

### 6. Hierarchical behavior is demonstrated on slow context drift

- [ ] 6.1 A two-level T-maze fixture defines a categorical context that changes
  on the parent clock.
- [ ] 6.2 The fixture defines child trials whose transition law depends on that
  context.
- [ ] 6.3 The comparison includes a flat agent without the slow parent state.
- [ ] 6.4 The comparison holds observations fixed across agents.
- [ ] 6.5 The comparison holds pragmatic preferences fixed across agents.
- [ ] 6.6 The comparison holds interaction budgets fixed across agents.
- [ ] 6.7 The comparison holds the seed set fixed across agents.
- [ ] 6.8 A committed preregistration fixes the context-drift protocol before
  comparative results are inspected.
- [ ] 6.9 The preregistration fixes the parent-posterior metric.
- [ ] 6.10 The preregistration fixes the parent-posterior acceptance threshold.
- [ ] 6.11 The preregistration fixes the child task-success metric.
- [ ] 6.12 The preregistration fixes the child task-success acceptance
  threshold.
- [ ] 6.13 The hierarchical parent-posterior result satisfies its frozen
  threshold.
- [ ] 6.14 The hierarchical child task-success result satisfies its frozen
  threshold against the flat agent.
- [ ] 6.15 Invalid experimental runs are reported with their actual runtime
  status.
- [ ] 6.16 Every valid experimental cell records its code revision.
- [ ] 6.17 Every valid experimental cell records its hierarchy artifact hash.
- [ ] 6.18 Every valid experimental cell records its seed.
- [ ] 6.19 Every valid experimental cell records its complete configuration.
- [ ] 6.20 Every valid experimental cell retains raw traces sufficient to
  recompute the reported metrics.

### 7. A depth-two contingent policy is a real conditional plan

- [ ] 7.1 A public builder expands one future categorical observation into a
  first layer of policy branches.
- [ ] 7.2 The builder expands a second future categorical observation into a
  second layer of policy branches.
- [ ] 7.3 An action before a branch point is shared by every descendant branch.
- [ ] 7.4 An action after a branch point is indexed by the observation history
  available at that point.
- [ ] 7.5 The returned policy object exposes the action distribution for each
  reachable observation history.
- [ ] 7.6 The returned policy object excludes unreachable observation
  histories.
- [ ] 7.7 Executing a returned branch action requires no new inference run.
- [ ] 7.8 The expanded template uses the cff-mjku observation-branch operation.
- [ ] 7.9 The expanded template uses the cff-mjku fan-out operation.
- [ ] 7.10 The expanded template inherits cff-9xmq's balanced equality topology.
- [ ] 7.11 Branch names are deterministic for a given input template.
- [ ] 7.12 The expanded template passes model validation.
- [ ] 7.13 The expanded template round-trips through JSON.
- [ ] 7.14 The expanded template preserves its structure hash across a
  round-trip.
- [ ] 7.15 The expanded template round-trips through a `.cffgz` bundle.
- [ ] 7.16 The builder rejects a requested branch depth greater than two.
- [ ] 7.17 The rejection for depth greater than two explains the combinatorial
  boundary.
- [ ] 7.18 Existing noncontingent rollout builders retain their serialized
  output.

### 8. Contingent-policy scoring follows the declared objective

- [ ] 8.1 The branch-template derivation defines the probability of each
  observation history.
- [ ] 8.2 The branch-template derivation defines the free-energy contribution of
  each observation history.
- [ ] 8.3 The complete policy score weights each branch contribution by its
  declared branch probability.
- [ ] 8.4 An impossible observation branch contributes zero probability mass.
- [ ] 8.5 An impossible observation branch contributes no finite surrogate
  reward.
- [ ] 8.6 The branch score is obtained from the expanded CFFG ledger rather than
  a separate heuristic bonus.
- [ ] 8.7 On a tiny depth-one tree, the policy score matches direct enumeration
  within a preregistered dtype-aware tolerance.
- [ ] 8.8 On a tiny depth-two tree, the policy score matches direct enumeration
  within a preregistered dtype-aware tolerance.
- [ ] 8.9 On the tiny depth-two tree, each conditional action marginal matches
  direct enumeration within a preregistered dtype-aware tolerance.
- [ ] 8.10 A commitment fixture has an analytically known gap between the best
  open-loop action sequence and the best contingent policy.
- [ ] 8.11 The depth-two policy attains the contingent value in the commitment
  fixture within its frozen tolerance.
- [ ] 8.12 The noncontingent baseline attains the open-loop value in the
  commitment fixture within its frozen tolerance.
- [ ] 8.13 The reported policy-tree result is not described as deep
  sophisticated inference.

### 9. Hostile inputs and uncomfortable scale fail safely or remain finite

- [ ] 9.1 An exact-zero parent regime remains absent from the child selector
  support.
- [ ] 9.2 An exact-zero child likelihood remains absent from the
  child-to-parent cavity support.
- [ ] 9.3 Positive child evidence does not resurrect a parent regime assigned
  exact-zero prior support.
- [ ] 9.4 An empty intersection between parent support and child-cavity support
  produces the declared infeasible status.
- [ ] 9.5 A vacuous parent message follows the declared uniform-message
  semantics.
- [ ] 9.6 An improper categorical prior fails validation before execution.
- [ ] 9.7 A depth-two tree with an exact-zero observation branch remains finite
  on its reachable branches.
- [ ] 9.8 A depth-two tree with a vacuous observation message remains finite.
- [ ] 9.9 Every numeric floor, clamp, ridge, or stopping tolerance derives from
  the active dtype.
- [ ] 9.10 Every numeric guard names the invariant it protects.
- [ ] 9.11 The float32 two-level hierarchy produces finite supported messages.
- [ ] 9.12 The float32 two-level hierarchy produces finite joint free energy.
- [ ] 9.13 The float32 depth-two policy tree produces finite supported messages.
- [ ] 9.14 The float32 depth-two policy tree produces finite free energy.
- [ ] 9.15 The hierarchy scale case has at least ten times the child-tick count
  of the smallest hierarchy oracle.
- [ ] 9.16 The policy-tree scale case has at least ten times the categorical
  branch entries of the smallest contingent oracle.
- [ ] 9.17 The resource report records compile time by branch depth.
- [ ] 9.18 The resource report records warm execution time by branch depth.
- [ ] 9.19 The resource report records peak arena bytes by branch depth.
- [ ] 9.20 The resource report records generated factor count by branch depth.
- [ ] 9.21 Any new Newton-type inner solve uses globalization unless its domain
  is proven globally safe.

### 10. All affected execution surfaces and documentation agree

- [ ] 10.1 The implementation handoff walks the `Runtime numerical or status
  invariant` row in `docs/ARCHITECTURE.md` and marks each surface changed or
  structurally irrelevant with a reason.
- [ ] 10.2 The implementation handoff walks the `Learnable or checkpoint
  feature` row in `docs/ARCHITECTURE.md` and marks each surface changed or
  structurally irrelevant with a reason.
- [ ] 10.3 If implementation adds an IR field, the handoff walks the `New IR
  field` row and marks each surface changed or structurally irrelevant with a
  reason.
- [ ] 10.4 If implementation adds a factor or rule family, the handoff walks the
  `New factor or rule family` row and marks each surface changed or
  structurally irrelevant with a reason.
- [ ] 10.5 Host and JIT inter-level message paths agree within preregistered
  tolerances.
- [ ] 10.6 Flooding and scheduled child inference agree within preregistered
  tolerances.
- [ ] 10.7 Single-instance and `vmap` hierarchy execution are marked changed or
  structurally irrelevant with a reason.
- [ ] 10.8 Current and legacy hierarchy checkpoint decoding follow the rule in
  criterion 5.11.
- [ ] 10.9 `tests/test_schedule.py` covers the nested runtime changes.
- [ ] 10.10 `tests/test_runtime_status.py` covers nested status propagation.
- [ ] 10.11 `tests/test_goal_hyperprior.py` covers the nested cadence regression.
- [ ] 10.12 `tests/test_compose.py` covers the hierarchy boundary construction.
- [ ] 10.13 `tests/test_factored_rollout.py` covers the parent-selected child
  transition.
- [ ] 10.14 Focused contingent-template tests cover depth-one enumeration.
- [ ] 10.15 Focused contingent-template tests cover depth-two enumeration.
- [ ] 10.16 Focused contingent-template tests cover branch-depth refusal.
- [ ] 10.17 `docs/DERIVATIONS.md` contains the inter-level message derivation.
- [ ] 10.18 `docs/DERIVATIONS.md` contains the coupled free-energy derivation.
- [ ] 10.19 `docs/DERIVATIONS.md` names an executable anchor for the inter-level
  message derivation.
- [ ] 10.20 `docs/DERIVATIONS.md` names an executable anchor for the coupled
  free-energy derivation.
- [ ] 10.21 `docs/ARCHITECTURE.md` documents the delivered hierarchy boundary.
- [ ] 10.22 `docs/ARCHITECTURE.md` documents the delivered shallow-policy
  boundary.
- [ ] 10.23 `README.md` documents the public hierarchy workflow.
- [ ] 10.24 `README.md` documents the public contingent-policy workflow.
- [ ] 10.25 `docs/LIMITATIONS.md` states the depth-two policy-tree boundary.
- [ ] 10.26 `docs/LIMITATIONS.md` states the two-level hierarchy boundary.
- [ ] 10.27 `docs/todo-notes/m7-hierarchies.md` is deleted in the same commit
  that lands the completed capability.
- [ ] 10.28 The final code change is followed by one passing default short-suite
  run through the repository dashboard workflow.

## Methodological Suggestions

### Derive the boundary before choosing the driver shape

The todo note suggests two `Engine` instances plus a small host driver. That is
a strong starting point, but another implementation is acceptable if it
preserves the declared model, cavity, joint ledger, and clock semantics. Start
with a two-state parent and a tiny child `transition_mixture`; enumerate the
joint distribution independently, then derive the parent-to-child message,
child-to-parent cavity, inter-level entropy correction, and joint free energy.
Use that oracle before adding T-maze behavior.

Keep the model boundary explicit. A parent belief may supply an empirical prior
to the child, but the child's return message must divide out that incoming
information in log space. Exchange only messages from converged level states.
Treat inference iterations as optimization work, child ticks as observations,
and parent periods as the slow clock; none of those counters may stand in for
another.

### Reuse the delivered factor and learning machinery

The categorical parent-selector node already exists as
`transition_mixture`, and `StateBlockSpec.selector` plus `CouplingSpec` already
lower parent-dependent transitions inside a factored template. Reuse those
rules unless the derivation proves they cannot express the selected boundary.
Use cff-gjjk's staged goal posterior as the depth-two cadence regression: the
nested driver should move it at the declared episode boundary, not on every
inference pass.

The primary change-path rows are `Runtime numerical or status invariant` and
`Learnable or checkpoint feature`. Walk flooding, colored/mixed, and exact
schedules; host and JIT support checks; `_make_step`, `_make_exact_unit`,
`_make_propagate`, and `_make_bfe`; `RunStatus` consumers; agent cadence;
checkpoint encode/decode; legacy migration; and bundles. Architecture currently
anticipates no new IR or factor family. If implementation changes that
assumption, walk the corresponding row before coding and update
`docs/ARCHITECTURE.md` in the same commit.

### Build contingency as a bounded template expansion

Use cff-mjku's observation-branch and fan-out operations to synthesize an
ordinary CFFG template. Let cff-9xmq supply balanced equality topology. Keep the
expanded graph as the policy score's source of truth; do not implement a
separate tree-search scorer. A branch-conditioned readout may index action
marginals by observation history, but each marginal must come from a named
variable in the expanded template.

Freeze v1 at depth two. Use a commitment task whose observation becomes
available after the common action but whose later action mapping must be chosen
before another inference call; direct enumeration then gives both the best
open-loop value and the best contingent value. Report template size, compile
time, warm execution time, and arena bytes by branch depth so the exponential
boundary remains visible.

### Preserve falsifiable scientific claims

Commit the slow-context T-maze protocol before inspecting comparative results.
Specify the context process, temporal thickness, priors, observations,
preferences, seed set, budgets, convergence conditions, parent-posterior
metric, child-success metric, and acceptance thresholds. Keep regime tracking
separate from task success. Preserve invalid cells with their actual statuses
instead of dropping them from the report.

### Evidence anchors and dependency state

- `docs/ARCHITECTURE.md` “Hierarchical and sophisticated inference” defines the
  two-level message/schedule target and the bounded policy-template approach.
- `docs/LIMITATIONS.md` H5 explains why a factorized control posterior is not a
  contingent policy.
- `docs/LIMITATIONS.md` S3 records the missing inter-level derivation and
  temporal-thickness implementation.
- `docs/LIMITATIONS.md` S4 records shallow branching as a compute-for-policy-
  expressiveness trade.
- `notes/pt2.txt` §7.3 supplies the two-timescale buyer/seller precedent: two
  child actions occur per parent offer, and the parent parameter posterior
  rolls between trials.
- cff-1d6t supplies the closed T-maze parity baseline.
- cff-gjjk supplies the closed learned-goal message, ledger, rollover, and
  checkpoint base case.
- cff-d56d supplies the closed deterministic mixed/exact schedule foundation
  and truthful fixed-point certification.
- cff-wbhu owns one-dispatch iterative execution for `Engine.run` and is ready.
- cff-9xmq owns balanced Broadcast expansion and waits on cff-wbhu.
- cff-mjku owns public fan-out and interior observation-branch edits and waits
  on cff-9xmq.
- cff-pyy9 waits on cff-mjku and has no dependent roadmap ticket.

Legend: `-->` means prerequisite to dependent.

```text
[x] cff-j4mu --+
[x] cff-d56d --+--> [ ] cff-wbhu (ready)
[x] cff-44we --+             |
                             +--> [ ] cff-9xmq
                                         |
                                         +--> [ ] cff-mjku
                                                     |
[x] cff-1d6t ----------------------------------------+
[x] cff-gjjk ----------------------------------------+--> [ ] cff-pyy9
[x] cff-d56d ----------------------------------------+        (blocked; terminal)

[ ] cff-pyy9 --> no dependent roadmap feature or epic
```

## Future work

No future work is delegated by this epic. Hierarchies deeper than two levels
and policy trees deeper than two observation branches are explicit unsupported
boundaries, not implied follow-up commitments. A later request for either
capability requires a separate ticket with its own derivation, scaling limits,
and evidence.

## Definitions

- **BFE**: Bethe free energy, assembled from node-local terms and edge-entropy
  corrections.
- **GFE**: generalized free energy, the P-substituted objective used for active
  inference.
- **Hierarchy**: two coupled probabilistic models whose states update on
  different clocks.
- **Parent level**: the slow categorical model that supplies the child's
  transition-regime belief.
- **Child level**: the faster model that executes several environment ticks
  during one parent period.
- **Flat agent**: the comparison agent that models no slow parent regime.
- **Inter-level boundary**: the named pair of ports through which parent and
  child exchange log messages.
- **Empirical prior**: a prior supplied by another inferred belief rather than
  a fixed parameter table.
- **Cavity message**: an outbound message with the recipient's incoming message
  removed so the recipient does not count its own information twice.
- **Temporal thickness**: the positive integer number of child environment
  ticks in one parent period.
- **Environment tick**: one executed action-and-observation transition in the
  modeled world.
- **Inference iteration**: one optimization update that does not itself create
  an observation.
- **Posterior rollover**: an explicit transfer of a completed period's
  posterior into the next period's prior.
- **Joint ledger**: the coupled model's node-local free-energy terms plus its
  edge-entropy corrections.
- **Goal hyperprior**: a Dirichlet belief over goal statistics that is updated
  between completed trials.
- **Contingent policy**: a mapping from possible observation histories to later
  actions.
- **Open-loop action sequence**: a sequence fixed before future observations
  are known.
- **Observation branch**: one policy-tree branch associated with a possible
  future categorical observation.
- **Branch depth**: the number of future observation points represented in the
  policy tree.
- **Depth-two policy tree**: a policy template with two future observation
  branch points.
- **Sophisticated inference**: planning over policies that condition actions on
  future observations; this ticket implements only its bounded template form.
- **Commitment fixture**: a task in which a later action mapping must be chosen
  before another inference call.
- **Direct enumeration**: an independent calculation that sums every state,
  observation, and action case in a tiny model.
- **RunStatus**: the runtime classification for convergence, budget exhaustion,
  divergence, or infeasible support.
- **Structure hash**: the deterministic identifier of serialized model
  structure, excluding relocatable parameter paths.
- **Hostile-scale case**: a fixture at least ten times larger than the smallest
  matching oracle along the dimension named by its acceptance criterion.

## Notes

### 2026-08-07 — Historical evidence and rewrite audit

The bounded historical slice returned one `validate` candidate in one
occurrence, classified as component validation, with zero problem occurrences.
The final scan returned 55 candidates across 318 occurrences. Manual context
review found no ambiguity requiring another edit.

Retained candidate meanings:

- `end_to_end`: the consumer workflow from model declaration through behavioral
  evidence is meant.
- `fixed`: an experimental control is pinned, a parameter table is supplied, or
  an open-loop sequence is chosen before observations.
- `preserve`: a semantic invariant, artifact identity, or actual runtime status
  is maintained.
- `public`: a documented consumer-facing contract is meant.
- `reusable`: an already-delivered component is used again.
- `same`: the declared clock, input, output, or atomic commit is identical under
  the stated comparison.
- `available`: an observation is known at the named decision point.
- `expose`: the policy object returns the named consumer-visible readout.
- `implement`: planned component work, an already-present component, or the
  bounded capability delivered by this epic is meant according to context.
- `provide`: prerequisite tickets supply named foundations.
- `support`: probability support is meant except where the text names host and
  JIT support-check functions.
- `complete`: an exhaustive ledger or configuration, a finished modeled period,
  or ticket-scope completion is meant according to context.
- `deliver`: a prerequisite supplied a named capability or documentation states
  a capability boundary.
- `land`: the completed capability is merged into the repository.
- `ready`: declared prerequisites permit work to start.
- `remaining`: a named state or property continues to hold.
- `approximate`: a mathematically named approximation with a reference or bound
  is meant.
- `close`: a ticket reached work-item completion.
- `sufficient`: the raw traces meet the stated metric-recomputation threshold.
- `certify`: the schedule's fixed-point certificate proves its declared
  condition.
- `check`: a named host or JIT support-check function is meant.
- `converge`: the runtime convergence state or its declared condition is meant.
- `evidence`: observed child information, empirical behavioral support, or a
  source anchor is meant according to context.
- `exact`: a cardinality, mathematical identity, literal zero, or exact schedule
  class is meant.
- `fixed_point`: the mathematical message state unchanged by another update is
  meant.
- `match`: numeric equality within a stated tolerance or correspondence to a
  named oracle is meant.
- `oracle`: an independently calculated tiny-model reference is meant.
- `parity`: equality with the published T-maze reference is meant.
- `pass`: test success or one inference traversal is meant according to context.
- `reproduce`: deterministic resumption of the named execution is meant.
- `test`: a named executable file or focused executable claim is meant.
- `validate`: input validation is meant in criteria 3.9–3.11 and 9.5;
  component validation is meant for the expanded template and historical use.
- `accumulate`: sufficient statistics are added at an executed model boundary.
- `infer`: an inference operation, inference state, or inferred belief is meant.
- `learn`: goal-learning state or delivered learning machinery is meant.
- `learnable`: the concrete serialized `LearnableSpec` contract is meant.
- `rollover`: an explicit completed-period posterior-to-prior transfer is meant.
- `trial`: one completed experimental episode is meant.
- `scale`: fixture size or computational growth is meant.
- `small`: the relative domain of an oracle or the narrow host-driver code scope
  is meant.
- `speed`: the parent's and child's relative model clocks are meant.
- `can`: an already-present model-construction capability is meant.
- `may`: a method is permitted but not required.
- `should`: an optional methodological recommendation is meant.
- `explicit_owner`: ticket ownership is stated for cff-wbhu, cff-9xmq, and
  cff-mjku; “its own prior” refers to information returning to its source.
- `follow_up`: the text explicitly denies an implied future commitment.
- `named_ticket_handoff`: cff-wbhu, cff-9xmq, and cff-mjku own the named
  prerequisite outputs.
- `all`: every listed schedule family is in scope.
- `any`: the stated new-solver scope is universal.
- `at least`: the lower bound is inclusive.
- `every`: the stated collection is universal.
- `only`: a hard temporal or capability boundary is meant.
- `current`: the repository's present architecture or the current side of a
  current/legacy checkpoint pair is meant.
- `existing`: a component or serialized behavior already present in the
  repository is meant.
- `later`: an action or request occurs after the earlier event named in the
  sentence.

Intelligible-ticket review: complete.
