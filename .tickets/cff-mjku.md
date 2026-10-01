---
id: cff-mjku
status: open
deps: [cff-9xmq]
links: [cff-b47g, cff-xa6z]
created: 2026-07-10T03:31:28Z
type: feature
priority: 3
assignee: coopwilliams
---
# Modeling ergonomics: fan-out and observation branches for interior latents

## Background in Plain English

Forney-style graph edges may touch at most two factors, so sharing one value
with several factors requires equality nodes. The plate expander and factored
rollout builder already create those nodes internally, but model authors still
have to hand-build the same structure in ordinary templates. A model author
also cannot expose a fully connected latent as observed data without splitting
its edge, inserting an equality node, and adding a DATA port as one coherent
edit. This ticket adds one public fan-out operation and one public
observation-branch operation that perform those transformations atomically with
deterministic names. cff-9xmq supplies the balanced equality-tree algorithm;
cff-mjku makes that algorithm reusable by general model code and supplies the
two structural operations required by cff-pyy9's shallow policy templates.

## Outcomes / Acceptance Criteria

### 1. Fan-out has one precise public contract

- [ ] 1.1 The public fan-out operation accepts a validated input `Template`.
- [ ] 1.2 The operation identifies its target by the target variable's complete
  model-local name.
- [ ] 1.3 `branch_count` means the number of new branch ports requested by the
  caller.
- [ ] 1.4 The operation requires `branch_count` to be a positive integer.
- [ ] 1.5 The operation preserves the target edge's first existing endpoint.
- [ ] 1.6 The operation preserves the target edge's second existing endpoint.
- [ ] 1.7 The operation returns one new GLUE port for each requested branch.
- [ ] 1.8 The operation returns the new GLUE port names in deterministic order.
- [ ] 1.9 Each returned GLUE port has the target variable's domain.
- [ ] 1.10 Each returned GLUE port represents the same random value as the
  target edge.
- [ ] 1.11 A degree-one target with its required existing port is accepted.
- [ ] 1.12 A degree-two interior target is accepted.
- [ ] 1.13 A target used as a parameter edge is rejected in v1.
- [ ] 1.14 The parameter-edge rejection names the unsupported target.
- [ ] 1.15 The operation accepts scoped names containing `/`.
- [ ] 1.16 The operation does not depend on a caller's factor naming convention.
- [ ] 1.17 The operation does not mutate the input `Template`.

### 2. Fan-out lowers to one balanced equality-tree authority

- [ ] 2.1 The public fan-out operation uses the balanced equality-tree
  synthesis delivered by cff-9xmq.
- [ ] 2.2 Plate `Broadcast` uses the same equality-tree synthesis authority.
- [ ] 2.3 The factored rollout builder uses the same equality-tree synthesis
  authority for logical-net fan-out.
- [ ] 2.4 The generated tree contains only the existing `equality` factor type.
- [ ] 2.5 The generated tree introduces no new parameter blob.
- [ ] 2.6 The generated tree introduces no variable with factor degree greater
  than two.
- [ ] 2.7 Every generated degree-one variable has exactly one port.
- [ ] 2.8 Every generated degree-two variable has no port.
- [ ] 2.9 The generated equality-factor count is minimal for a ternary tree with
  the declared endpoint count.
- [ ] 2.10 Generated factor names are deterministic for a given input template
  and call.
- [ ] 2.11 Generated variable names are deterministic for a given input template
  and call.
- [ ] 2.12 The generated schedule diameter grows logarithmically with branch
  count.
- [ ] 2.13 A 49-leaf fan-out inherits cff-9xmq's frozen convergence-sweep bound.
- [ ] 2.14 A one-branch call produces the smallest valid equality expansion.
- [ ] 2.15 A two-branch call produces the smallest valid equality expansion.
- [ ] 2.16 A 16-branch call requires no hand-written equality specification.

### 3. Fan-out preserves inference and free-energy semantics

- [ ] 3.1 A discrete fan-out's hub marginal matches an independently hand-built
  balanced reference within a preregistered dtype-aware tolerance.
- [ ] 3.2 Each discrete branch marginal matches the same hand-built reference
  within a preregistered dtype-aware tolerance.
- [ ] 3.3 The discrete fan-out BFE matches the same hand-built reference within a
  preregistered dtype-aware tolerance.
- [ ] 3.4 A Gaussian fan-out's hub marginal matches an independently hand-built
  balanced reference within a preregistered dtype-aware tolerance.
- [ ] 3.5 Each Gaussian branch marginal matches the same hand-built reference
  within a preregistered dtype-aware tolerance.
- [ ] 3.6 The Gaussian fan-out BFE matches the same hand-built reference within a
  preregistered dtype-aware tolerance.
- [ ] 3.7 Exact-zero discrete support is not resurrected by the equality tree.
- [ ] 3.8 A vacuous discrete branch contributes no relative log preference.
- [ ] 3.9 A vacuous Gaussian branch contributes zero information.
- [ ] 3.10 An empty support intersection retains the engine's actual infeasible
  status.
- [ ] 3.11 A non-finite message retains the engine's actual divergence status.

### 4. Opening an observation branch is one atomic graph transformation

- [ ] 4.1 The public observation-branch operation accepts a validated input
  `Template`.
- [ ] 4.2 The operation identifies its target by the target variable's complete
  model-local name.
- [ ] 4.3 The target must be a degree-two interior variable.
- [ ] 4.4 The target must have no existing port.
- [ ] 4.5 The target must belong to a domain accepted by the existing DATA
  binding path.
- [ ] 4.6 The operation preserves the first factor endpoint of the target edge.
- [ ] 4.7 The operation preserves the second factor endpoint of the target edge.
- [ ] 4.8 The operation inserts exactly one existing `equality` factor.
- [ ] 4.9 The operation creates exactly one observation variable.
- [ ] 4.10 The operation creates exactly one DATA port on the observation
  variable.
- [ ] 4.11 The operation returns the DATA port's deterministic name.
- [ ] 4.12 The observation variable has the target latent's domain.
- [ ] 4.13 The operation creates no likelihood factor.
- [ ] 4.14 The operation creates no parameter blob.
- [ ] 4.15 The operation preserves every constraint attached to the original
  target edge.
- [ ] 4.16 The operation accepts scoped names containing `/`.
- [ ] 4.17 The operation does not depend on a caller's factor naming convention.
- [ ] 4.18 The operation does not mutate the input `Template`.

### 5. Observation branches reproduce manual clamp semantics

- [ ] 5.1 An unbound categorical observation branch leaves the original latent
  marginal unchanged within a preregistered dtype-aware tolerance.
- [ ] 5.2 An unbound categorical observation branch leaves the original BFE
  unchanged within a preregistered dtype-aware tolerance.
- [ ] 5.3 A bound categorical observation branch matches a manually expanded
  equality-and-DATA reference marginal within a preregistered dtype-aware
  tolerance.
- [ ] 5.4 A bound categorical observation branch matches the manually expanded
  reference BFE within a preregistered dtype-aware tolerance.
- [ ] 5.5 An exact-zero categorical clamp remains exact zero off the observed
  state.
- [ ] 5.6 An unbound Gaussian observation branch leaves the original latent
  marginal unchanged within a preregistered dtype-aware tolerance.
- [ ] 5.7 An unbound Gaussian observation branch leaves the original BFE
  unchanged within a preregistered dtype-aware tolerance.
- [ ] 5.8 A bound Gaussian observation branch matches a manually expanded
  equality-and-DATA reference marginal within a preregistered dtype-aware
  tolerance.
- [ ] 5.9 A bound Gaussian observation branch matches the manually expanded
  reference BFE within a preregistered dtype-aware tolerance.
- [ ] 5.10 An observation value outside a discrete target's cardinality fails
  through the existing DATA-binding validation.
- [ ] 5.11 A Gaussian observation with the wrong shape fails through the
  existing DATA-binding validation.

### 6. Invalid edits fail atomically and explain the violated boundary

- [ ] 6.1 Fan-out of an unknown variable fails before returning a template.
- [ ] 6.2 Fan-out with `branch_count=0` fails before returning a template.
- [ ] 6.3 Fan-out with a negative branch count fails before returning a
  template.
- [ ] 6.4 Fan-out with a nonintegral branch count fails before returning a
  template.
- [ ] 6.5 Fan-out with a generated factor-name collision fails before returning
  a template.
- [ ] 6.6 Fan-out with a generated variable-name collision fails before
  returning a template.
- [ ] 6.7 Fan-out with a generated port-name collision fails before returning a
  template.
- [ ] 6.8 Observation branching of an unknown variable fails before returning a
  template.
- [ ] 6.9 Observation branching of a degree-one variable fails before returning
  a template.
- [ ] 6.10 Observation branching of a variable that already has a port fails
  before returning a template.
- [ ] 6.11 Observation branching of an unsupported DATA domain fails before
  returning a template.
- [ ] 6.12 Observation branching with a generated factor-name collision fails
  before returning a template.
- [ ] 6.13 Observation branching with a generated variable-name collision fails
  before returning a template.
- [ ] 6.14 Observation branching with a generated port-name collision fails
  before returning a template.
- [ ] 6.15 Reapplying the same edit fails on its deterministic name collision.
- [ ] 6.16 A failed edit leaves the source template byte-for-byte unchanged in
  canonical serialized form.
- [ ] 6.17 Each failure identifies the requested operation.
- [ ] 6.18 Each failure identifies the invalid target or generated name.
- [ ] 6.19 `check_edits` reports each invalid request without mutating the input
  template.

### 7. Both operations are first-class edit and serialization clients

- [ ] 7.1 Each operation has a frozen edit-action value object.
- [ ] 7.2 Each direct Python helper delegates to the same lowering used by its
  edit action.
- [ ] 7.3 Each edit action applies through `apply_edits`.
- [ ] 7.4 Each edit action round-trips through the edit-bundle JSON codec.
- [ ] 7.5 An unknown serialized field in either new edit action fails decoding.
- [ ] 7.6 A missing required serialized field in either new edit action fails
  decoding.
- [ ] 7.7 The transformed template passes `validate`.
- [ ] 7.8 The transformed template round-trips through template JSON.
- [ ] 7.9 The transformed template preserves its structure hash across a JSON
  round-trip.
- [ ] 7.10 Each nontrivial operation changes the source template's structure
  hash.
- [ ] 7.11 A transformed template round-trips through a `.cffgz` bundle.
- [ ] 7.12 The edit codec preserves scoped target names exactly.
- [ ] 7.13 The public operations are exported from the documented core modeling
  namespace.

### 8. The helpers compose into their actual downstream uses

- [ ] 8.1 The factored rollout builder no longer owns a private linear fan-out
  algorithm.
- [ ] 8.2 The factored rollout builder's external construction API remains
  unchanged.
- [ ] 8.3 The factored rollout builder's model semantics remain unchanged.
- [ ] 8.4 The factored rollout parity fixture retains its joint marginals within
  its existing tolerance.
- [ ] 8.5 The factored rollout parity fixture retains its BFE within its existing
  tolerance.
- [ ] 8.6 A generic non-plate template creates a 16-leaf fan-out with one public
  call.
- [ ] 8.7 A generic non-plate template opens a supervised categorical latent
  with one public call.
- [ ] 8.8 A generic non-plate template opens a Gaussian latent with one public
  call.
- [ ] 8.9 The two operations can build a depth-two observation-branch skeleton
  without hand-written equality factors.
- [ ] 8.10 The depth-two skeleton passes model validation.
- [ ] 8.11 This ticket adds no policy-scoring behavior to the depth-two skeleton.
- [ ] 8.12 This ticket adds no hierarchy runtime behavior to the depth-two
  skeleton.

### 9. Hostile dtype and scale cases remain finite

- [ ] 9.1 A float32 discrete fan-out produces finite supported messages.
- [ ] 9.2 A float32 discrete fan-out produces finite BFE.
- [ ] 9.3 A float32 Gaussian fan-out produces finite information-form messages.
- [ ] 9.4 A float32 Gaussian fan-out produces finite BFE.
- [ ] 9.5 A float32 categorical observation branch produces finite supported
  messages.
- [ ] 9.6 A float32 categorical observation branch produces finite BFE.
- [ ] 9.7 A float32 Gaussian observation branch produces finite
  information-form messages.
- [ ] 9.8 A float32 Gaussian observation branch produces finite BFE.
- [ ] 9.9 Every numeric guard introduced by this work derives from the active
  dtype.
- [ ] 9.10 Every numeric guard introduced by this work names the invariant it
  protects.
- [ ] 9.11 The uncomfortable fan-out fixture creates 64 new branches.
- [ ] 9.12 The uncomfortable fan-out fixture uses categorical cardinality 32.
- [ ] 9.13 The uncomfortable fan-out fixture compiles successfully.
- [ ] 9.14 The uncomfortable fan-out fixture reports `CONVERGED` within its
  frozen inference budget.
- [ ] 9.15 The uncomfortable fan-out fixture allocates equality-tree structure
  linearly in branch count.

### 10. Change-path review, tests, and documentation cover the public boundary

- [ ] 10.1 The implementation handoff walks the `New IR field` row in
  `docs/ARCHITECTURE.md` for the serialized edit-action additions.
- [ ] 10.2 The handoff marks each `New IR field` surface changed or structurally
  irrelevant with a reason.
- [ ] 10.3 The handoff records `New factor or rule family` as structurally
  irrelevant because both operations use existing equality semantics.
- [ ] 10.4 The handoff records `Runtime numerical or status invariant` as
  structurally irrelevant unless implementation changes existing runtime
  behavior.
- [ ] 10.5 The handoff records `Learnable or checkpoint feature` as structurally
  irrelevant unless implementation changes existing learnable state.
- [ ] 10.6 Host and JIT equality execution are marked unchanged or changed with
  focused evidence.
- [ ] 10.7 Flooding and scheduled equality execution are marked unchanged or
  changed with focused evidence.
- [ ] 10.8 Single-instance and `vmap` equality execution are marked unchanged or
  changed with focused evidence.
- [ ] 10.9 Current and legacy checkpoint decoding are marked structurally
  irrelevant unless implementation stores runtime state.
- [ ] 10.10 `tests/test_edits.py` covers successful atomic transformations.
- [ ] 10.11 `tests/test_edits.py` covers rollback after transformation failure.
- [ ] 10.12 `tests/test_edit_codec.py` covers both new action codecs.
- [ ] 10.13 `tests/test_compose.py` covers generic non-plate fan-out composition.
- [ ] 10.14 `tests/test_plates.py` covers the shared balanced synthesis authority.
- [ ] 10.15 `tests/test_factored_rollout.py` covers migration from private fan-out.
- [ ] 10.16 `tests/test_serialize.py` covers transformed-template JSON and hash
  behavior.
- [ ] 10.17 `tests/test_bundle.py` covers transformed-template bundle round-trip.
- [ ] 10.18 Focused hostile-input tests cover exact-zero support.
- [ ] 10.19 Focused hostile-input tests cover vacuous messages.
- [ ] 10.20 Focused hostile-input tests cover float32.
- [ ] 10.21 Focused hostile-input tests cover the 64-branch scale case.
- [ ] 10.22 `docs/ARCHITECTURE.md` names the shared equality-tree synthesis
  authority.
- [ ] 10.23 `docs/ARCHITECTURE.md` documents the observation-branch transform.
- [ ] 10.24 `README.md` documents the public fan-out workflow.
- [ ] 10.25 `README.md` documents the public observation-branch workflow.
- [ ] 10.26 `docs/LIMITATIONS.md` removes the one-call-helper gap only after both
  public workflows land.
- [ ] 10.27 The final code change is followed by one passing default short-suite
  run through the repository dashboard workflow.

## Methodological Suggestions

### Extract one structural authority

cff-9xmq should leave one domain-preserving balanced-tree synthesis function
that accepts ordered endpoint specifications and emits ordinary equality
variables and factors with deterministic names. Plate `Broadcast`, public
fan-out, and the factored builder's logical-net materialization should call that
one function. Another implementation is acceptable if it demonstrably keeps a
single topology authority and avoids three subtly different balancing and
naming algorithms.

Define fan-out around a valid edge, not around an invalid intermediate
template. Treat the edge's existing two endpoints as leaves, append the
caller's ordered new GLUE leaves, then synthesize the minimum balanced ternary
equality tree. For a degree-one edge, its required existing port is the second
endpoint. Return the new ports rather than making callers rediscover generated
names.

### Make observation opening a specialized edit

For a degree-two latent, deterministically retain one endpoint on the original
variable, move the other endpoint to a same-domain variable, and connect those
two edges to the new DATA edge through one ternary equality. Preserve an edge
constraint on the retained original variable; equality propagates its semantic
effect to the other leaves. Reject parameter edges in v1 rather than assuming
that copying a variational parameter has ordinary structural-edge semantics.

Implement both transformations as frozen edit-action objects in
`src/cffg/core/edits.py`, and let convenience functions call the action path.
Extend the existing action JSON codec rather than adding another wire format.
Use `apply_edits` for transactional validation and `check_edits` for dry-run
diagnostics.

### Validate semantics independently of generated names

Build small manual balanced references for categorical and Gaussian cases.
Compare marginals and BFE, not raw message gauges. For categorical fan-out,
the normalized product of branch messages supplies an analytic anchor. For
Gaussian fan-out, summed information vectors and precision matrices supply the
anchor. Exercise an unbound DATA port as a vacuous message before checking a
clamped observation.

Keep exact zeros exact. Equality introduces no need for a numeric floor. If
implementation adds a guard for indexing or normalization, derive it from the
active dtype and state the invariant beside the code.

### Evidence anchors and dependency state

- `src/cffg/core/validate.py::_check_degrees_and_ports` enforces the degree-two
  edge and port invariants these operations must preserve.
- `src/cffg/core/plates.py::_apply_broadcast` is the current linear synthesis
  that cff-9xmq replaces with a balanced authority.
- `tests/test_plates.py::test_broadcast_matches_hand_expanded_reference` pins
  discrete Broadcast semantics against a manual equality graph.
- `tests/test_plates.py::test_sensor_array_information_addition` pins Gaussian
  equality information addition.
- `src/cffg/models/factored.py:_Graph` is the private linear fan-out client to
  migrate onto the shared authority.
- cff-b47g supplies closed factored-rollout marginal, BFE, exact-support,
  float32, and scale evidence for that migration.
- cff-wbhu owns the compiled sweep loop and is ready.
- cff-9xmq owns balanced equality-tree topology and waits on cff-wbhu.
- cff-mjku waits on cff-9xmq.
- cff-pyy9 consumes both operations for shallow policy templates and waits on
  cff-mjku.

Legend: `-->` means prerequisite to dependent; `~~~` means linked evidence
without a dependency edge.

```text
[x] cff-j4mu --+
[x] cff-d56d --+--> [ ] cff-wbhu (ready)
[x] cff-44we --+             |
                             +--> [ ] cff-9xmq
                                         |
                                         +--> [ ] cff-mjku (blocked)
                                                     |
                                                     +--> [ ] cff-pyy9

[x] cff-b47g ~~~~~> [ ] cff-mjku (linked implementation evidence)
```

## Future work

cff-pyy9 is the dependent ticket. It owns policy-tree expansion, conditional
action readouts, policy scoring, hierarchy runtime, and behavioral evidence;
this ticket supplies only the two graph transformations it consumes. Fan-out
of parameter edges remains unsupported because copying variational parameters
needs a separate mathematical contract; no ticket currently owns that work.

## Definitions

- **Forney-style factor graph (FFG)**: a factor graph in which a variable is an
  edge connecting at most two factors.
- **Endpoint**: one factor role or port attached to a variable edge.
- **Factor degree**: the number of factors attached to one variable.
- **Fan-out**: a structural transform that represents one random value on
  several equal variable edges.
- **Branch count**: the number of new GLUE ports requested by one fan-out call.
- **Equality factor**: a factor that constrains each attached variable to the
  same value.
- **Ternary equality tree**: a tree of equality factors with three variable
  roles per factor.
- **Balanced tree**: a tree whose longest root-to-leaf path grows
  logarithmically with leaf count.
- **Hub**: the source side of a fan-out whose information combines with every
  branch.
- **GLUE port**: a composition boundary intended to bond two templates.
- **DATA port**: a runtime boundary through which a caller clamps an observed
  value.
- **Interior latent**: an unobserved variable with two factor endpoints and no
  port.
- **Clamp**: conditioning a DATA variable on a supplied observed value.
- **Vacuous message**: a message that contributes no relative categorical
  preference or Gaussian information.
- **Information form**: a Gaussian representation using precision and
  precision-weighted mean.
- **Edit action**: a serializable description of one template transformation.
- **Atomic edit**: a transformation that returns a valid new template or leaves
  the original unchanged after failure.
- **Scoped name**: a model-local name containing composition prefixes separated
  by `/`.
- **Structure hash**: the deterministic identifier of serialized template
  structure, excluding relocatable blob paths.
- **Hand-built reference**: an independently authored small graph used to check
  generated topology and inference.
- **Parameter edge**: a variable connected to a factor's uncertain parameter
  role rather than to one of its structural roles.
- **Hostile-scale fixture**: the 64-branch, cardinality-32 fan-out required by
  criteria 9.11–9.15.

## Notes

### 2026-08-07 — Historical evidence and rewrite audit

The bounded historical slice returned five candidates across five occurrences
and classified zero occurrences as problems. Historical meanings were
`reusable` as internal code reuse, `test` as automated execution, `validate` as
acceptance validation, `can` as possibility, and `every` as distribution over
the named class. The final scan returned 39 candidates across 265 occurrences.
Manual context review found no ambiguity requiring another edit.

Retained candidate meanings:

- `canonical`: the unique normalized template serialization is meant.
- `generic`: an ordinary non-plate template with no model-specific naming is
  meant.
- `preserve`: endpoint semantics, constraints, names, hashes, or source bytes
  retain the identity stated by the surrounding criterion.
- `public`: the documented consumer-facing modeling API is meant.
- `reusable`: shared internal code or a downstream-consumable operation is
  meant.
- `same`: object identity, value equality, one shared authority, or one domain
  is meant according to the surrounding sentence.
- `expose`: the operation creates a DATA interface to an interior latent.
- `implement`: planned code work, a conditional implementation change, or
  historical implementation evidence is meant according to context.
- `support`: the feasible categorical probability set is meant.
- `complete`: the full scoped name, the exhaustive meaning map, or ticket-scope
  completion is meant according to context.
- `deliver`: cff-9xmq supplies the named balancing algorithm.
- `land`: both workflows are merged into the repository.
- `ready`: cff-wbhu has no unfinished declared prerequisite.
- `remaining`: a named value, API, semantics, or unsupported boundary continues
  unchanged.
- `close`: cff-b47g reached work-item completion.
- `check`: a manual or executable comparison against a named semantic claim is
  meant.
- `converge`: the frozen sweep threshold or `CONVERGED` runtime state is meant.
- `evidence`: focused executable support, prerequisite results, or a dependency
  relationship is meant according to context.
- `exact`: a cardinality, literal zero, literal name identity, or exact-support
  property is meant.
- `match`: numeric equality within the tolerance named by the criterion is
  meant.
- `parity`: equality with the linked factored-rollout reference is meant.
- `pass`: model-validator success or short-suite success is meant.
- `reproduce`: generated observation semantics equal the manual reference.
- `test`: a named test artifact or automated executable check is meant.
- `validate`: input validation, component validation, or the historical
  acceptance-validation classification is meant according to context.
- `infer`: inference behavior or its frozen budget is meant.
- `learnable`: the named architecture change-path row or already-present
  learnable state is meant.
- `scale`: the declared 64-branch hostile fixture or linked scale evidence is
  meant.
- `small`: a minimum valid expansion or an independently authored narrow
  reference graph is meant.
- `can`: the two public operations demonstrate the stated composability.
- `intended`: the documented purpose of a GLUE port is meant.
- `may`: the FFG structural rule permits at most two factor endpoints.
- `should`: an optional methodological recommendation is meant.
- `explicit_owner`: “belongs to a domain” means domain membership; `owns` names
  the work assigned to cff-wbhu, cff-9xmq, cff-pyy9, or no ticket.
- `named_ticket_handoff`: cff-wbhu, cff-9xmq, and cff-pyy9 own the named outputs.
- `every`: the stated generated-variable, constraint, guard, or branch
  collection is universal.
- `only`: a hard factor-family, temporal, or ticket-scope exclusion is meant.
- `current`: the repository's present synthesis or the current side of a
  current/legacy checkpoint pair is meant.
- `existing`: an endpoint, port, factor, validator, tolerance, runtime surface,
  learnable state, or codec already present in the repository is meant.

Intelligible-ticket review: complete.

**2026-09-25T03:47:40Z**

PCGym evidence under cff-xa6z: the experiment repeatedly hand-built equality nodes to tap interior continuous states. This is additional evidence for the existing public fan-out/observation-branch contract, not a new duplicate ticket. Keep its cff-9xmq prerequisite and degree-two/serialization requirements; PCGym recipes can show explicit equality construction until it lands.
