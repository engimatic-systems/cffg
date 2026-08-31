---
id: cff-ocxl
status: open
deps: [cff-iur0]
links: []
created: 2026-08-31T03:47:35Z
type: feature
priority: 2
assignee: coopwilliams
parent: cff-zov5
tags: [rslds, inference, hybrid, compiler]
---
# Implement the derived regime-trajectory inference component

## Background in Plain English — WHY

cff-iur0 determines whether reliable source-specific learning needs a richer
posterior relationship between the hidden regime path and continuous path.
This ticket turns the accepted relationship into reusable engine behavior.

The rejected historical integration recognized one complete rSLDS template
and redirected inference, learning, and free-energy evaluation around the
ordinary factor path. An extra attachment or constraint made that recognition
fail and silently restored a materially different approximation. That is not
a composable factor-graph contract. The replacement must be selected from
serialized local semantics, expose explicit boundaries, and coexist with
ordinary factors outside the component.

## Outcomes / Acceptance Criteria — WHAT

1. Before implementation touches the compiler or engine, docs/ARCHITECTURE.md's
   change-path index gains a **Derived component coordinate** row naming the
   semantic authority, every derived compiler/runtime/persistence surface,
   focused tests, and durable documentation. The ticket handoff then marks
   each surface applicable or N/A.
2. A template selects the accepted coordinate through serialized factor and
   constraint semantics or an explicit serialized component declaration.
   Selection does not inspect an rSLDS model class, factor spelling, sorted
   names, or one complete benchmark topology.
3. Any new serialized semantic choice participates in validation, JSON
   round-trip, structure hashing, plate scoping, edits/composition, and bundle
   identity through the **New IR field** change path.
4. Compilation produces a bounded component plan containing its internal
   factors and variables, boundary ports, array indexes, schedule stage,
   approximation identity, and objective ownership. The serialized template
   plus data remains sufficient to reconstruct it.
5. Every boundary uses the same categorical log-potential or Gaussian natural
   message contract as an ordinary neighboring factor. Component-private
   beliefs are not exposed as a second graph state authority.
6. The component owns exactly the internal factor energies, region entropies,
   and separator corrections derived by cff-iur0. Ordinary boundary-edge
   corrections remain engine-owned, and the summed Bethe free energy matches
   the tiny independent objective oracle.
7. Synchronous and scheduled execution produce the same boundary messages,
   internal readouts, Bethe free energy, support, and terminal status within
   dtype-scaled tolerances. Host/JIT and single-instance/`vmap` behavior also
   agree.
8. Recurrence pair readouts and source-conditioned continuous statistics come
   from the accepted component belief. `Learnables.replace` consumes those
   statistics through the existing declared family contract without an
   rSLDS-only parameter-update path.
9. If the statistic meaning differs from D9, the learnable semantic manifest,
   executable identity, checkpoint, and CFFGZ learning-state identity receive
   a new version. D9 state is rejected or migrated explicitly before mutation;
   no shape-compatible state is silently resumed under the new meaning.
10. Three composition fixtures retain the component approximation while adding
    through ordinary equality/fan-out structure: (a) an extra Gaussian
    measurement or constraint on an interior continuous state, (b) an extra
    categorical likelihood or constraint on an interior regime, and (c) a
    non-rSLDS factor sharing a declared boundary variable. Renaming and
    reordering every factor leaves selection and results unchanged.
11. A composition that violates a derived assumption fails validation or
    compilation with the unsupported boundary or constraint named. It never
    runs D9, D10, or another objective as an unreported fallback.
12. Literal zero categorical support remains exact; empty support, improper or
    vacuous Gaussian input, non-finite objective/residual, and invalid shapes
    fail loudly. Iterative execution distinguishes converged,
    budget-exhausted, and diverged and uses globalization unless cff-iur0
    proves it unnecessary.
13. Float32 uses dtype-derived guards. A compact hostile fixture and a case
    10–100 times its horizon or regime count stay within the complexity and
    storage bounds derived by cff-iur0.
14. The current D9 fixed-drive and source-homogeneous reductions remain covered
    by executable equivalence tests. No production whole-template D10
    classifier or model-owned engine dispatch is introduced.
15. README.md documents construction and unsupported compositions;
    docs/ARCHITECTURE.md records ownership and execution; docs/DERIVATIONS.md
    links the implementation anchors; docs/LIMITATIONS.md states the remaining
    approximation and scaling boundary.

## Methodological Suggestions — HOW

These suggestions are optional. The implementer may choose another sound
method that satisfies the outcomes.

- Keep the component solver in its own module and bind it through a generic
  plan/kernel protocol so `Engine.run` does not gain an rSLDS topology branch.
- Derive component membership from explicit local declarations and graph
  incidence. Keep names, model helpers, and learned-owner counts out of
  semantic eligibility.
- Reuse existing D9 factor messages at boundaries when the derivation proves
  them unchanged; avoid maintaining duplicate recurrence probability code.
- Apply **New factor or rule family** and **Runtime numerical or status
  invariant** as the minimum existing change paths. Apply **New IR field** and
  **Learnable or checkpoint feature** wherever criteria 3 or 9 make them
  applicable.

## Future work

cff-x0if owns the only end-to-end source-recovery run authorized by this epic.
Compiler fusion, balanced-tree transitions, destination-softmax transitions,
GFE/P-substitution inside this component, and whole-trajectory sampling remain
separate work unless cff-iur0 explicitly derives one of them as necessary.

## Definitions

- **Approximation identity:** A stable name and version for the mathematical
  belief/statistic semantics used by an executable component.
- **Component plan:** The compiled indexes and operations for one declared
  connected inference region.
- **Semantic fallback:** Running a different approximation or objective after
  the declared one becomes ineligible, without rejecting the model.
- **Separator correction:** The entropy-counting adjustment for a variable or
  marginal shared by neighboring regions.
- **Whole-template recognition:** Choosing semantics only when the compiler
  sees one complete model-specific graph layout.

## Notes

Initial change-path handoff: the present index has no row for this ticket
shape, which is why criterion 1 must land first. Until then, use **New factor or
rule family** and **Runtime numerical or status invariant**. **New IR field**
is applicable if selection needs a serialized declaration. **Learnable or
checkpoint feature** is applicable if source-conditioned statistic semantics
or persisted learning state changes.

Duplicated-surface handoff: host/JIT, flooding/scheduled,
single-instance/`vmap`, and current/legacy checkpoint behavior are all
applicable. The implementation handoff must also check compiler grouping,
slot addressing, parameter packing, `Arena`, `member_of`, `_make_step`,
scheduled batch/scan stages, `_make_bfe`, schedule classification, readouts,
serialization/bundle manifests, estimator impact, and public exports.

Intelligible-ticket review: complete. This ticket owns the generic component
plan/kernel protocol and its first regime/trajectory binding. cff-iur0 owns the
equations; existing factor modules retain their local probabilities and
messages; `Engine` retains public status and ledger assembly; `Learnables`
retains posterior replacement; cff-x0if owns empirical recovery evidence.

Wiggle-audit review retained clear contextual uses. `end_to_end` is the one
bounded cff-x0if workflow; `fixed` names the fixed-drive D9 reduction;
`generic` and `reusable` mean the model-independent component protocol;
`integration` is the rejected historical compiler/engine wiring; `N/A` is an
explicit change-path classification; `public` means package exports;
`same` means the existing canonical boundary representation or numeric
agreement across schedules; `expose` means declaring a boundary or forbidding
a second state authority; `implement` means this ticket's internal integration
and executable anchors; `support` and `exact` refer to literal categorical
support, except `exactly` also fixes objective ownership; `complete` means the
whole recognized graph; `land` means criterion 1 is committed before later
code; `remaining` means reconstructible state, retained ownership, literal
empty support, preserved reductions, or explicitly deferred work as stated;
`approximation` is the versioned mathematical identity; `material` means a
different objective semantics; `sufficient` means reconstructible from the
listed source of truth; `check`, `test`, and `validate` name the enumerated
review, executable, or input-validation actions; `converged` is `RunStatus`
success; `equivalence` is tolerance equality for named reductions; `estimator`
is the resource estimator; `match` is tolerance agreement with the objective
oracle; `oracle` is the independent tiny reference; `sound` means another
method satisfying acceptance; `stable` means the version identifier is not
silently redefined; `infer` means the component posterior update; `learn` and
`learnable` mean the existing declared posterior-statistic and persistence
contracts; `recover` names cff-x0if's withheld-function estimate; `scale` means
dtype tolerances or the derived complexity bound; `may` grants methodological
choice; `all`, `any`, `every`, and `only` are bounded by the explicitly named
surface, declaration, boundary, factor, and ticket sets; `current` names the
shipped D9 baseline or current/legacy persistence pair; and `existing` names
checked-in D9 contracts or architecture rows.
