---
id: cff-9xmq
status: open
deps: [cff-wjqg]
links: [cff-6s31, cff-ahps]
created: 2026-07-10T03:30:52Z
type: feature
priority: 2
assignee: coopwilliams
---
# Plate Broadcast: balanced equality tree — O(log N) BP diameter on hub fan-outs

## Background in Plain English

A plate `Broadcast` represents one hub value shared by every replicated branch.
Forney-style graph edges may touch at most two factors, so the plate expander
currently realizes that sharing with a chain of ternary equality factors. The
chain is mathematically correct, but evidence may need to cross a number of
factors proportional to the branch count: the recorded 49-patch perception
model needed about 70 flooding sweeps and processed only 4.1 sequential or 5.1
batched images per second against a 20-images-per-second bar. Its 784-leaf
version already compiled in about 1.2 seconds and used about 1.3 GB, so the
measured bottleneck was propagation diameter rather than compilation or one
sweep's arithmetic.

This ticket replaces the derived chain with one deterministic balanced equality
tree while leaving the serialized `Broadcast` and `PlateSpec` contract
unchanged. A shared topology planner will describe the tree without requiring a
materialized `Template`; the current plate expander will materialize that plan,
cff-6s31 can later lower it into compact index tables, and cff-mjku can later use
it for non-plate fan-out. The result retains equality semantics and linear
structural size while reducing the equality subtree's diameter to logarithmic
growth.

## Outcomes / Acceptance Criteria

### 1. One authority defines the balanced topology

- [ ] 1.1 The implementation has one balanced equality-tree planning function.
- [ ] 1.2 The planning function accepts an ordered sequence of branch endpoint
  specifications.
- [ ] 1.3 The planning function accepts one hub endpoint specification.
- [ ] 1.4 The planning function does not require a materialized `Template`.
- [ ] 1.5 The planning result describes every generated equality factor.
- [ ] 1.6 The planning result describes every generated intermediate variable.
- [ ] 1.7 The planning result preserves the input branch order.
- [ ] 1.8 Plate branches enter the planner in the expander's current row-major
  order.
- [ ] 1.9 A branch interval is split into two contiguous subintervals at each
  internal node.
- [ ] 1.10 The two child subintervals at each split differ in branch count by at
  most one.
- [ ] 1.11 Generated factor names are deterministic for a given plate name,
  hub name, and ordered branch sequence.
- [ ] 1.12 Generated intermediate-variable names are deterministic for the same
  inputs.
- [ ] 1.13 A generated name collision fails before the source template is
  mutated.
- [ ] 1.14 Every generated factor uses the repository's current ternary
  `equality` factor type.
- [ ] 1.15 Every generated variable retains the branch domain.
- [ ] 1.16 The planner introduces no parameter blob.
- [ ] 1.17 The planner introduces no numeric approximation.

### 2. Tree size and diameter have reviewable bounds

- [ ] 2.1 In this ticket, `N` denotes the number of Broadcast copy branches and
  excludes the hub endpoint.
- [ ] 2.2 `N=1` retains the current zero-factor rename of the branch to the hub.
- [ ] 2.3 `N=2` produces one equality factor joining the hub and both branches.
- [ ] 2.4 Every `N>=2` expansion contains exactly `N-1` equality factors.
- [ ] 2.5 Every `N>=2` expansion contains exactly `N-2` generated intermediate
  variables.
- [ ] 2.6 No generated variable has factor degree greater than two.
- [ ] 2.7 The longest hub-to-branch path contains at most `ceil(log2(N))`
  equality factors.
- [ ] 2.8 The longest branch-to-branch path contains at most
  `2*ceil(log2(N))` equality factors.
- [ ] 2.9 A 49-branch expansion therefore has an equality-factor diameter no
  greater than 12.
- [ ] 2.10 A structural test checks factor count for `N` values 1, 2, 3, 4, 49,
  and 784.
- [ ] 2.11 A structural test checks intermediate-variable count for the same
  `N` values.
- [ ] 2.12 A structural test checks both path bounds for the same `N` values.
- [ ] 2.13 A structural test checks deterministic names across two independent
  expansions.

### 3. The serialized Broadcast boundary remains compatible

- [ ] 3.1 The fields of `Broadcast` remain `port` and `hub`.
- [ ] 3.2 The fields of `PlateSpec` remain unchanged.
- [ ] 3.3 The schema version remains unchanged.
- [ ] 3.4 A frozen pre-change Broadcast JSON fixture decodes without migration.
- [ ] 3.5 Re-encoding that fixture reproduces its canonical JSON value.
- [ ] 3.6 Its compact-template structure hash remains unchanged.
- [ ] 3.7 A template JSON round-trip reconstructs the balanced derived topology.
- [ ] 3.8 A `.cffgz` round-trip reconstructs the balanced derived topology.
- [ ] 3.9 The derived Arena contains the balanced equality connections.
- [ ] 3.10 The derived schedule fingerprint is deterministic across independent
  compiles.
- [ ] 3.11 A change from the historical chain's schedule fingerprint is treated
  as an expected derived-artifact change rather than a schema change.
- [ ] 3.12 Existing model-construction code using `Broadcast(port, hub)` needs no
  source change.

### 4. Balanced and chain expansions have the same inference semantics

- [ ] 4.1 An independently authored linear reference remains in the test
  evidence.
- [ ] 4.2 A categorical balanced hub marginal matches that linear reference at
  float64 absolute tolerance `1e-10`.
- [ ] 4.3 Every categorical balanced branch marginal matches that linear
  reference at float64 absolute tolerance `1e-10`.
- [ ] 4.4 The categorical balanced BFE matches that linear reference at float64
  absolute tolerance `1e-10`.
- [ ] 4.5 The categorical hub marginal matches the independently computed
  normalized product of branch messages at float64 absolute tolerance `1e-10`.
- [ ] 4.6 A Gaussian balanced hub mean matches the independently summed
  information-form reference at a preregistered dtype-aware tolerance.
- [ ] 4.7 A Gaussian balanced hub covariance matches the independently summed
  information-form reference at the same tolerance.
- [ ] 4.8 Every Gaussian balanced branch marginal matches the linear reference
  at the same tolerance.
- [ ] 4.9 The Gaussian balanced BFE matches the linear reference at the same
  tolerance.
- [ ] 4.10 Sum-product scheduled execution matches sum-product flooding after
  both report their successful runtime state.
- [ ] 4.11 Max-product scheduled decoding matches max-product flooding decoding.
- [ ] 4.12 Max-product scheduled objective matches max-product flooding
  objective at a preregistered dtype-aware tolerance.
- [ ] 4.13 The existing `test_broadcast_matches_hand_expanded_reference`
  continues to pass after its generated side becomes balanced.
- [ ] 4.14 The existing `test_sensor_array_information_addition` continues to
  pass after its generated side becomes balanced.
- [ ] 4.15 The existing `test_broadcast_n1_and_n2_edge_cases` continues to pass.

### 5. Scheduling exposes the logarithmic propagation result

- [ ] 5.1 A pure balanced Broadcast tree receives a `two_pass` automatic
  schedule.
- [ ] 5.2 The `two_pass` schedule fires each equality factor once inward.
- [ ] 5.3 The `two_pass` schedule fires each equality factor once outward.
- [ ] 5.4 The `two_pass` result is certified as a fixed point by the current
  runtime contract.
- [ ] 5.5 Before topology replacement, a committed consumer-equivalent 49-branch
  flooding fixture records the historical linear-chain sweep count.
- [ ] 5.6 That fixture fixes its categorical cardinality, branch potentials,
  initialization, tolerance, sweep budget, dtype, and seed before measuring the
  balanced implementation.
- [ ] 5.7 The frozen fixture's historical chain result falls within 60 through
  80 performed flooding sweeps.
- [ ] 5.8 The balanced version of the frozen fixture reports `CONVERGED` in at
  most 12 performed flooding sweeps.
- [ ] 5.9 The balanced fixture does not satisfy criterion 5.8 by exhausting its
  budget.
- [ ] 5.10 The balanced fixture does not satisfy criterion 5.8 with a non-finite
  free energy.
- [ ] 5.11 A mixed-schedule fixture re-equilibrates a balanced Broadcast forest
  island through the current exact forest pass.
- [ ] 5.12 The mixed-schedule fixture matches flooding marginals at a
  preregistered dtype-aware tolerance.
- [ ] 5.13 The mixed-schedule fixture matches flooding BFE at the same tolerance.

### 6. Hostile support, dtype, and scale cases remain truthful

- [ ] 6.1 A categorical branch with exact log-zero entries keeps those entries
  outside support after propagation.
- [ ] 6.2 Compatible sparse categorical branches produce finite supported
  messages.
- [ ] 6.3 Disjoint categorical branch supports report the runtime's infeasible
  status.
- [ ] 6.4 A vacuous categorical branch contributes no relative log preference.
- [ ] 6.5 A vacuous Gaussian branch contributes zero information.
- [ ] 6.6 An injected non-finite categorical message reports the runtime's
  divergence status.
- [ ] 6.7 A float32 categorical tree produces finite supported messages.
- [ ] 6.8 A float32 categorical tree produces finite BFE.
- [ ] 6.9 A float32 Gaussian tree produces finite information-form messages.
- [ ] 6.10 A float32 Gaussian tree produces finite BFE.
- [ ] 6.11 Float32 semantic comparisons use tolerances derived from the active
  dtype.
- [ ] 6.12 Equality-tree synthesis adds no numeric floor, ridge, or clamp.
- [ ] 6.13 Every numeric guard added by implementation derives from the active
  dtype.
- [ ] 6.14 Every numeric guard added by implementation names the invariant it
  protects.
- [ ] 6.15 The uncomfortable categorical fixture uses cardinality 32.
- [ ] 6.16 The uncomfortable Gaussian fixture uses dimension 32.
- [ ] 6.17 A 784-branch fixture supplies the required greater-than-tenfold scale
  increase over the 49-branch fixture.
- [ ] 6.18 The 784-branch planner completes without recursion-depth failure.
- [ ] 6.19 The 784-branch expansion passes template validation.
- [ ] 6.20 The 784-branch expansion compiles successfully.
- [ ] 6.21 The 784-branch equality-factor count is 783.
- [ ] 6.22 The 784-branch intermediate-variable count is 782.
- [ ] 6.23 The 784-branch equality-factor diameter is no greater than 20.

### 7. The measured performance claim separates topology from dispatch

- [ ] 7.1 Performance measurement begins after cff-wbhu's compiled sweep loop is
  present.
- [ ] 7.2 One committed benchmark command reproduces the frozen 49-branch
  perception-equivalent workload.
- [ ] 7.3 The benchmark reports compile time separately from warm inference
  time.
- [ ] 7.4 The benchmark reports performed flooding sweeps separately from time
  per sweep.
- [ ] 7.5 The benchmark reports sequential images per second.
- [ ] 7.6 The benchmark reports batched images per second.
- [ ] 7.7 The benchmark records the hardware identifier.
- [ ] 7.8 The benchmark records the JAX and jaxlib versions.
- [ ] 7.9 The benchmark records the active dtype.
- [ ] 7.10 The benchmark records the batch size.
- [ ] 7.11 The benchmark records the warmup count.
- [ ] 7.12 The benchmark records the timed count.
- [ ] 7.13 The benchmark records the convergence tolerance.
- [ ] 7.14 The benchmark records the flooding sweep budget.
- [ ] 7.15 The balanced benchmark reaches at least 20 sequential images per
  second on the recorded Apple M2 Max, 64 GB, CPU-only configuration.
- [ ] 7.16 The balanced benchmark reaches at least 20 batched images per second
  on the recorded Apple M2 Max, 64 GB, CPU-only configuration.
- [ ] 7.17 If that baseline hardware is unavailable, the handoff reports the
  blocker instead of claiming the 20-images-per-second outcome from different
  hardware.
- [ ] 7.18 A same-process A/B benchmark shows the balanced topology performs
  fewer flooding sweeps than the linear reference.
- [ ] 7.19 The 784-branch compiled persistent-state byte count does not exceed
  the linear reference's corresponding count.
- [ ] 7.20 The 784-branch generated factor count does not exceed the linear
  reference's corresponding count.
- [ ] 7.21 The 784-branch generated variable count does not exceed the linear
  reference's corresponding count.
- [ ] 7.22 The handoff does not attribute a per-sweep speedup to this topology
  change unless the benchmark measures one.

### 8. Shared-plan boundaries and change-path evidence are complete

- [ ] 8.1 Plate `_apply_broadcast` materializes the shared planner's output.
- [ ] 8.2 The planner's output can be lowered from logical endpoint ordinals
  without copied fragment objects.
- [ ] 8.3 The planner's output exposes analytic factor and intermediate-variable
  counts to cff-6s31.
- [ ] 8.4 The planner accepts non-plate endpoint specifications for cff-mjku.
- [ ] 8.5 This ticket adds no public non-plate fan-out operation.
- [ ] 8.6 This ticket adds no compact plate compiler.
- [ ] 8.7 The implementation handoff walks the `New IR field` change-path row as
  the closest authority-to-derived-surface checklist.
- [ ] 8.8 The handoff records `Broadcast` and `PlateSpec` validation as unchanged
  or changed with focused evidence.
- [ ] 8.9 The handoff records JSON decoding, encoding, and structure hashing as
  unchanged or changed with focused evidence.
- [ ] 8.10 The handoff records plate scoping and expansion as changed with
  focused evidence.
- [ ] 8.11 The handoff records composition and edit actions as structurally
  irrelevant because this ticket adds no public edit.
- [ ] 8.12 The handoff records compiler and Arena derivation as changed or
  unchanged with focused evidence.
- [ ] 8.13 The handoff records bundle reconstruction as unchanged or changed
  with focused evidence.
- [ ] 8.14 The handoff records agent-state persistence as structurally
  irrelevant because the topology is not runtime state.
- [ ] 8.15 The handoff records `New factor or rule family` as structurally
  irrelevant because the existing equality factor supplies the semantics.
- [ ] 8.16 The handoff records `Runtime numerical or status invariant` as
  unchanged unless implementation modifies the current runtime contract.
- [ ] 8.17 The handoff records `Learnable or checkpoint feature` as structurally
  irrelevant unless implementation stores new persistent state.
- [ ] 8.18 Host and JIT equality execution are marked unchanged or changed with
  focused evidence.
- [ ] 8.19 Flooding and scheduled equality execution are marked unchanged or
  changed with focused evidence.
- [ ] 8.20 Single-instance and `vmap` equality execution are marked unchanged or
  changed with focused evidence.
- [ ] 8.21 Current and legacy checkpoint decoding are marked structurally
  irrelevant unless implementation stores new persistent state.
- [ ] 8.22 `tests/test_plates.py` covers topology, discrete semantics, Gaussian
  semantics, edge cases, exact-zero support, float32, and the 784-branch case.
- [ ] 8.23 `tests/test_schedule.py` covers two-pass, flooding, mixed scheduling,
  max-product, and the frozen 49-branch sweep bound.
- [ ] 8.24 `tests/test_serialize.py` covers the frozen Broadcast JSON and
  structure-hash contract.
- [ ] 8.25 `tests/test_bundle.py` covers balanced reconstruction from a compact
  plate bundle.
- [ ] 8.26 Focused `vmap` or `Colony` evidence covers the balanced compiled
  topology after cff-wbhu.
- [ ] 8.27 `docs/ARCHITECTURE.md` names the shared balanced-tree planning
  authority and its materialized and compact consumers.
- [ ] 8.28 `README.md` replaces the linear-diameter Broadcast warning with the
  delivered logarithmic bound and benchmark result.
- [ ] 8.29 `docs/LIMITATIONS.md` removes the linear Broadcast bottleneck only
  after the sweep and throughput criteria pass.
- [ ] 8.30 The final code change is followed by one passing default short-suite
  run through the repository dashboard workflow.

## Methodological Suggestions

### Plan a rooted binary tree of ternary equalities

Treat the hub as the external root edge and the ordered copy branches as the
leaves. Recursively divide each contiguous branch interval into halves whose
sizes differ by at most one. Materialize one ternary equality for an interval:
its `a` role faces the parent edge, while `b` and `c` face a leaf or the
intermediate edge for the two child intervals. An iterative work list is
preferable at uncomfortable scale, but another implementation is acceptable if
it proves the declared counts, order, name determinism, and diameter bounds.

Keep topology description separate from template mutation. A small frozen plan
value can carry endpoint ordinals, child references, and deterministic
structural paths; `_apply_broadcast` can then allocate ordinary `VariableSpec`
and `FactorSpec` objects from it. Validate all generated names before applying
the plan so a collision cannot leave a partially expanded template. This
boundary lets cff-6s31 consume indices and counts without materializing copied
fragments, and lets cff-mjku reuse the balancing decision without inheriting
plate naming.

### Prove semantics with independent references

Retain a linear graph only as a test oracle and A/B benchmark, not as a second
production synthesis authority. Compare hub and branch marginals plus BFE; do
not compare raw log messages because their additive gauges may differ. Use the
normalized product of categorical branch messages and the sum of Gaussian
information vectors and precision matrices as independent analytic anchors.
Exercise sum-product, max-product, sparse support, vacuous messages, float32,
and runtime failure statuses through the existing equality kernels.

### Freeze the performance experiment before replacing the chain

Reconstruct the 49-patch consumer workload from the recorded adversarial-review
measurement and commit its complete configuration before changing topology.
Record both the historical chain and balanced plan in one process after JIT
warmup. Separate compilation, sweep count, per-sweep time, and end-to-end
throughput so cff-wbhu's dispatch work is not credited to balancing and this
ticket is not credited with a faster equality kernel.

### Evidence anchors and dependency state

- `src/cffg/core/ir.py::Broadcast` and `PlateSpec` define the serialized
  authority that remains unchanged.
- `src/cffg/core/plates.py::_apply_broadcast` contains the current left-deep
  chain and row-major branch collection.
- `tests/test_plates.py::test_broadcast_matches_hand_expanded_reference` is the
  current discrete linear-reference comparison.
- `tests/test_plates.py::test_sensor_array_information_addition` is the current
  Gaussian information-addition comparison.
- `src/cffg/schedule.py::build_schedule` classifies acyclic BP structure for
  exact two-pass execution and constructs deterministic fingerprints.
- `notes/adversarial-review-2026-07-10/02-performance.txt` records the 49-patch
  sweep count and the sequential and batched throughput measurements.
- `docs/ARCHITECTURE.md` records the 784-leaf compilation measurement and names
  linear Broadcast diameter as the perception bottleneck.
- cff-wjqg owns the ordinary compiled sweep loop and is the immediate ready prerequisite.
- cff-9xmq owns balanced topology, semantic parity, propagation diameter, and
  the corresponding performance evidence.
- cff-mjku owns the public non-plate fan-out operations that consume this plan.
- cff-6s31 owns compact plate lowering and is linked rather than prerequisite.
- cff-ahps no longer requires balanced broadcast topology for closure. This
  ticket remains an optional scaling extension selected from measured demand.

Legend: `-->` means prerequisite to dependent; `~~~>` means linked interface
coordination without a dependency edge.

```text
[x] cff-j4mu --+
[x] cff-d56d --+--> [ ] cff-wjqg (ready)
[x] cff-44we --+             |
                             +--> [ ] cff-9xmq (blocked)
                                         |
                                         +--> [ ] cff-mjku (blocked)
                                         |           |
                                         |           +--> [ ] cff-pyy9
                                         |
                                         +--> [ ] optional scaling consumers

[ ] cff-9xmq ~~~> [ ] cff-6s31 (linked compact lowering contract)
```

## Future work

cff-mjku is the direct dependent that turns the shared planner into public
non-plate fan-out and observation-branch operations; this ticket supplies no
public edit API. cff-6s31 owns compact index-plan lowering, recursive plate
execution, and compact resource accounting; this ticket supplies only a
materialization-independent topology description and analytic tree counts.
cff-ahps records the finite extensible-kernel baseline and links the
delivered Broadcast performance. No ticket currently owns a higher-arity
equality kernel because the minimal ternary tree meets this ticket's contract.

## Definitions

- **Forney-style factor graph (FFG)**: a factor graph in which a variable is an
  edge connecting at most two factors.
- **Plate**: a serialized instruction to replicate one template fragment over
  declared dimensions.
- **Broadcast**: a plate wiring rule that represents every selected copy port
  as the same value as one parent-level hub.
- **Branch**: one replicated copy edge attached to the equality tree.
- **Hub**: the parent-level edge shared with every Broadcast branch.
- **Endpoint specification**: a logical name or ordinal describing an edge that
  the topology planner must connect.
- **Equality factor**: the existing factor that constrains its three variable
  edges to one value.
- **Intermediate variable**: a generated degree-two edge joining two equality
  factors.
- **Balanced equality tree**: the rooted tree produced by recursively splitting
  contiguous branch ranges into halves differing in size by at most one.
- **Equality-factor diameter**: the greatest number of equality factors on a
  path between two external branch or hub edges.
- **Flooding sweep**: one synchronous update of every eligible factor in the
  flooding schedule.
- **Two-pass schedule**: the exact inward-then-outward message schedule used for
  an acyclic BP graph.
- **Fixed point**: a message state left unchanged by another applicable update,
  as certified by the current runtime contract.
- **BFE**: Bethe free energy computed from node-local terms and edge-entropy
  corrections.
- **Vacuous message**: a message that contributes no relative categorical
  preference or Gaussian information.
- **Exact support**: the set of categorical states whose log probability is not
  literal negative infinity.
- **Derived topology**: factors and variables reconstructed from serialized
  plate instructions rather than stored in those instructions.
- **Structure hash**: the deterministic identifier of serialized template
  structure, excluding relocatable blob paths.
- **Schedule fingerprint**: the deterministic identifier of a compiled message
  schedule and its derived graph structure.
- **Materialized template**: a `Template` containing one concrete object for
  every expanded plate variable and factor.
- **Compact index plan**: cff-6s31's future representation of replicated graph
  structure through indices instead of copied template objects.
- **Consumer-equivalent fixture**: the committed 49-branch workload whose
  configuration reproduces the recorded perception model's propagation
  behavior.
- **Hostile-scale fixture**: the required 784-branch case with categorical
  cardinality 32 or Gaussian dimension 32, more than ten times the 49-branch
  structural baseline and the small semantic fixtures' domain size.

## Notes

### 2026-08-07 — Historical evidence and rewrite audit

The bounded historical slice returned eight candidates across ten occurrences
and classified zero occurrences as problems. Historical meanings were `same`
as value equality and object identity, `converge` as a runtime status criterion,
`exact` as no mathematical approximation, `fixed_point` as a mathematical
stationary solution, `test` as automated execution, `infer` as state or regime
posterior computation, `any` as a universal quantifier, and `existing` as a
compatibility baseline that predates the change.

The final scan returned 57 candidates across 343 occurrences. Manual contextual
review found zero ambiguous occurrences. Current-scan meanings are recorded
below; each retained candidate has one intended meaning or an explicitly
enumerated set of intended meanings in this ticket.

- `canonical`: the repository's normalized JSON value is meant.
- `end_to_end`: the timed benchmark path from a prepared compiled engine through
  inference completion and consumer readout is meant.
- `fixed`: model parameters are supplied and not learned during the inference
  comparison.
- `preserve`: ordered branches or historical measurements retain the property
  named in the surrounding sentence.
- `public`: the documented non-plate edit API owned by cff-mjku is meant.
- `reusable`: one internal topology description is consumed by more than one
  lowering path.
- `same`: value equality, identical inputs, or a shared source is meant
  according to the surrounding sentence.
- `available`: analytic counts are reachable by the named internal compact
  lowering consumer.
- `expose`: a planner result makes counts available to an internal caller or a
  schedule result makes the diameter effect observable.
- `implement`: planned code work or a conditional implementation change is
  meant according to context.
- `support`: the feasible categorical probability set or evidence favoring a
  named claim is meant according to context.
- `complete`: all required configuration fields, planner output, or ticket
  evidence are meant according to context.
- `deliver`: cff-9xmq supplies the named topology and performance result.
- `land`: a named implementation is merged into the repository.
- `ready`: cff-wbhu has no unfinished declared prerequisite.
- `remaining`: a serialized field, behavior, or baseline retains the stated
  property after this change.
- `approximate`: a numeric approximation is explicitly absent from topology
  planning or historical equality semantics.
- `certify`: the runtime's existing fixed-point check supplies the named
  execution certificate.
- `check`: an executable assertion or the architecture checklist is meant.
- `converge`: `RunStatus.CONVERGED` under the frozen runtime stopping criterion
  is meant.
- `correct`: the historical chain implements equality semantics without a
  mathematical error.
- `equivalent`: the committed fixture matches the named consumer controls that
  determine propagation and throughput.
- `estimate`: parameter estimation is excluded from the fixed-parameter
  inference comparison.
- `evidence`: a named test, benchmark result, code location, or dependency fact
  supports the surrounding claim.
- `exact`: a mathematical identity, literal count, literal log zero, or the
  repository's certified two-pass schedule is meant according to context.
- `fixed_point`: the solver state certified by the current runtime contract is
  meant.
- `match`: numeric equality within the tolerance named by the criterion is
  meant.
- `oracle`: the independently authored linear graph used only as test and A/B
  benchmark reference is meant.
- `parity`: equal inference outputs across the compared topology or execution
  surfaces are meant.
- `pass`: validator or test-suite success is meant.
- `replicate`: plate fragment copying or the corresponding copied branch is
  meant.
- `reproduce`: the frozen fixture attains the stated historical measurement or
  serialization value.
- `test`: a named test artifact or automated executable assertion is meant.
- `validate`: template validation or contextual audit review is meant according
  to context.
- `infer`: posterior message propagation at fixed model parameters is meant.
- `learn`: parameter estimation is explicitly excluded from the fixed-parameter
  inference comparison.
- `learnable`: the named architecture change-path category is meant.
- `performance`: the measured compile time, warm inference time, sweep count,
  or image throughput named by the surrounding sentence is meant.
- `scale`: branch-count growth or the declared 784-branch fixture is meant.
- `small`: the planner result is a bounded topology descriptor rather than a
  materialized copied graph.
- `speed`: measured per-sweep or image-throughput rate is meant.
- `can`: the shared plan permits the named later consumer to lower or reuse it.
- `expected`: the schedule-fingerprint difference follows from the declared
  topology replacement.
- `intended`: the contextual interpretation assigned to each scan candidate is
  meant.
- `may`: graph structure permits no more than the stated endpoint count.
- `should`: a methodological recommendation is meant.
- `will`: the ticket commits the planner and plate expander to the named future
  behavior.
- `explicit_owner`: `owns` assigns the named work to cff-wbhu, cff-9xmq,
  cff-mjku, cff-6s31, cff-ahps, or no current ticket.
- `named_ticket_handoff`: the named ticket supplies the output stated beside it.
- `all`: every member of the locally named generated-name or configuration set
  is meant.
- `every`: the statement applies universally to the named branch, factor,
  variable, message, or generated-object collection.
- `only`: a hard scope exclusion, one measured cause, or one shared value is
  meant according to context.
- `any`: every member of the named input class is meant.
- `at_least`: 20 images per second is an inclusive measured lower bound.
- `current`: the repository implementation or runtime contract inspected on
  2026-08-07 is meant.
- `existing`: the factor, test, runtime path, API shape, or behavior already
  present in the repository is meant.
- `later`: work explicitly assigned to cff-mjku or cff-6s31 follows this
  ticket's shared-plan delivery.

Intelligible-ticket review: complete.
