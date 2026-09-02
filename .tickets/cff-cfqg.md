---
id: cff-cfqg
status: open
deps: [cff-ofv1]
links: [cff-ahps, cff-209l, cff-edoz, cff-ws3c]
created: 2026-08-13T03:59:30Z
type: epic
priority: 1
assignee: coopwilliams
external-ref: gh-108
tags: [architecture, trustworthiness, extensibility]
---
# Epic: Reduce the central extension surface

## Background in Plain English — WHY

After the rSLDS clean-port stack, CFFG can run a broad mix of discrete,
Gaussian, exponential-family, hybrid, recurrence, constraint, and GFE paths.
Adding or changing one family still requires careful edits across large central
modules. At commit e488df3, `src/cffg/compiler.py` has 2,368 lines and
`src/cffg/engine.py` has 3,973 lines. Both interpret factor types in several
places. That makes a local feature look like an engine-wide change and makes it
easy for synchronous execution, schedules, objective terms, readouts, and
serialization-derived plans to drift apart.

This epic makes the existing library easier to trust and extend. It first
finishes the registered same-arena factor path. It then gives compiler stages,
runtime execution, schedules, objective accounting, and readouts focused
owners behind typed internal contracts. The final gate proves the resulting
extension paths with executable evidence.

This is an architecture program. It does not repeat rSLDS scientific
validation, add a factor family, set a performance target, define a public
release, or change the mathematical meaning of an existing model.

## Outcomes / Acceptance Criteria — WHAT

1. The direct dependency `cff-ofv1` is closed because its acceptance criteria
   are satisfied. Its dependency chain closes `cff-209l`, `cff-edoz`, and
   `cff-ws3c` first.
2. Registered factors that use one discrete, Gaussian, or exponential-family
   arena have one documented extension path for structure, messages, energy,
   parameters, observations, readouts, and serialized implementation identity.
3. A representative same-arena factor can be added and exercised without
   adding its type name to central compiler, engine, validation, schedule, or
   objective dispatch.
4. Factors that need more than one arena, a GFE composite, or a structural
   constraint keep explicit specialized plans.
   4.1. Each specialized plan has one compiler-lowering owner.
   4.2. Each specialized plan has one runtime-execution owner.
   4.3. Each specialized plan has one objective-accounting owner.
   4.4. No universal callback is introduced by passing irrelevant state for
        unrelated factor families.
5. Compiler planning uses context-sized typed stages for topology, registered
   lowering, specialized lowering, data/learning resolution, and assembly after
   bounded characterization pins the e488df3 serialized-to-arena contract.
6. Engine execution, schedules, state/data/mu3 lifecycle, ordinary iteration,
   explicit outer/exact strategies, objective accounting, and readouts use
   focused owners after bounded characterization records the applicable
   numerical, support, status, and diagnostic contracts.
7. The refactor preserves public imports and call forms, template JSON,
   structure hashes, CFFGZ bundles, parameter and observation addresses,
   learnable plans, agent checkpoints, and migration behavior present at
   e488df3.
8. The refactor preserves each message rule, BFE or GFE ledger term,
   edge-entropy correction, approximation label, inference schedule, and
   convergence classification present at e488df3.
9. Literal zero support, improper or vacuous inputs, float32, and
   uncomfortable-scale cases retain their documented values or loud status.
   No child ticket weakens the numerical robustness contract to make a
   structural extraction pass.
10. `docs/ARCHITECTURE.md` names the final owners and gives contributors a
    bounded change path for same-arena factors and each specialized plan class.
    `README.md` and `docs/LIMITATIONS.md` change only when a documented
    import or extension boundary actually changes.
11. Completion is described as a trustworthy architecture and bounded
    extension baseline. It is not described as proof of paper faithfulness,
    rSLDS recovery, runtime speed, release readiness, or universal factor
    coverage.

## Methodological Suggestions — HOW

These suggestions are optional. An implementer may choose another sound method.

- Land the `cff-209l` same-arena kernel foundation before moving compiler
  types or stages.
- Reuse existing mathematical authorities and add old/new observable
  characterization only for an uncovered refactor seam. A newly discovered
  scientific gap gets a separate ticket unless it proves the move unsafe.
- Move one owner at a time and keep compatibility imports until consumers
  migrate.
- Prefer typed values named for their responsibility over broad utility
  modules or partially initialized arena objects.
- Keep same-arena registration and specialized-plan lowering as separate,
  explicit extension classes.
- Use the **New factor or rule family**, **New IR field**, **Runtime numerical
  or status invariant**, and **Learnable or checkpoint feature** rows in the
  architecture change-path index as preservation checklists. Each child ticket
  states which duplicated execution surfaces apply.

## Future work

The following tickets remain outside this epic:

- `cff-xabn -> cff-enhs -> cff-jilz -> cff-cz6v` is the parallel persistent-
  learning/checkpoint chain.
- `cff-j7mm` owns resource-estimator correctness.
- `cff-wbhu` owns compiled-loop and Colony throughput.
- `cff-5edo` owns long-rollout convergence and warm starts.
- `cff-3kdk` owns readout vectorization.
- `cff-cj5h` owns epistemic diagnostics.
- `cff-o7hs` is the immediate nonlinear-family implementation consumer and
  waits for this epic. Research and derivation that do not implement a factor
  may continue independently.
- `cff-ahps` remains the downstream factor-breadth roadmap and waits for this
  architecture gate.

## Definitions

- **Central dispatch:** Code in a shared compiler, engine, schedule, validation,
  or objective module that selects behavior by a factor type name.
- **Extension surface:** The production modules and contracts that must change
  to add one factor or feature.
- **Focused owner:** One module responsible for one named interpretation or
  operation.
- **Same-arena factor:** A factor whose variable messages all use one discrete,
  Gaussian, or exponential-family representation.
- **Specialized plan:** An explicit compiled plan for behavior that cannot use a
  same-arena kernel contract, such as a hybrid factor, GFE composite, or
  structural constraint.
- **Typed stage:** A compiler or runtime step with a named input and output
  contract instead of hidden shared state.

## Notes

**2026-09-02 architecture-program handoff**

Source of truth: `tk dep tree --full`, `tk ready`, and `tk blocked` at this
revision.

Legend: prerequisite --> dependent

```text
[ ] cff-209l (ready)
  --> [ ] cff-ecw0
  --> [ ] cff-544x
  --> [ ] cff-8t4d
  --> [ ] cff-ejib
  --> [ ] cff-kfo7
  --> [ ] cff-bfr9
  --> [ ] cff-ddsq
  --> [ ] cff-edoz
  --> [ ] cff-sf29
  --> [ ] cff-ny7f
  --> [ ] cff-nyad
  --> [ ] cff-5xdy
  --> [ ] cff-ii9t
  --> [ ] cff-jrbw
  --> [ ] cff-gl89
  --> [ ] cff-3ola
  --> [ ] cff-ws3c
  --> [ ] cff-ofv1
  --> [ ] cff-cfqg
        |
        +--> [ ] cff-o7hs --> [ ] cff-1o72 --> [ ] cff-ahps
        |
        +-------------------------------------> [ ] cff-ahps

Parallel or downstream consumers outside this epic:
[ ] cff-xabn --> [ ] cff-enhs --> [ ] cff-jilz --> [ ] cff-cz6v
[ ] cff-j7mm
[ ] cff-wbhu --> [ ] cff-5edo
[ ] cff-3kdk
[ ] cff-cj5h

Parallel source-recovery research/specification:
[ ] cff-iur0 --> [ ] cff-ocxl --> [ ] cff-x0if --> [ ] cff-zov5
```

Capability ownership:

- `cff-209l` is the cross-track owner of executable same-arena discrete and
  Gaussian kernel registration.
- `cff-ecw0` and `cff-544x` own compiler characterization and compiled values;
  `cff-8t4d`, `cff-ejib`, `cff-kfo7`, `cff-bfr9`, and `cff-ddsq` own topology,
  registered lowering, specialized lowering, data/learning resolution, and
  final assembly; `cff-edoz` closes that track.
- `cff-sf29` owns bounded engine characterization. `cff-ny7f`, `cff-nyad`,
  `cff-5xdy`, `cff-ii9t`, `cff-jrbw`, `cff-gl89`, and `cff-3ola` own execution,
  schedules, lifecycle, ordinary control, explicit outer/exact control, the
  objective ledger, and readouts; `cff-ws3c` closes that track.
- `cff-ofv1` owns the measured extension-surface proof. It uses the
  `cff-209l` test-only factors to prove same-arena registration without
  central type-name edits and an explicit matrix for multi-arena recurrence,
  GFE/P-substitution, and constraints.
- `cff-o7hs` is the first new-family implementation consumer. The learning,
  resource, throughput, rollout, vectorization, diagnostics, and
  source-recovery tickets shown outside the epic retain their own scopes.

Architecture change-path audit:

- Compiler characterization and extraction use **New factor or rule family**,
  **New IR field**, and **Learnable or checkpoint feature**. Moving compiled
  values also checks **Runtime numerical or status invariant** because runtime
  consumers move imports.
- Engine characterization and extraction use **Runtime numerical or status
  invariant** and **New factor or rule family**. Restored-agent and
  learning-statistic consumers also use **Learnable or checkpoint feature**.
- The final proof records all four rows in its extension matrix.

Duplicated-surface and robustness audit:

- Compiler tickets keep flooding/scheduled, direct/`vmap`, and
  latest-schema/supported-legacy consumers on one compiled contract. Host/JIT
  compilation is N/A because compilation is host-side.
- Engine tickets pin flooding/scheduled, direct/one-instance `vmap`, host/JIT
  status, and latest-schema/supported-legacy checkpoint pairs.
- Compiler hostile values are runtime N/A; dtype-stable structure and a
  32-group fixture are required. Engine evidence covers literal zero support,
  vacuous and improper inputs, non-finite updates, float32, and a domain or
  horizon at least ten times its smallest fixture.
- Specialized multi-arena, GFE, evaluator, and constraint plans remain
  explicit. This program does not force a universal callback.

Scope boundary: no child owns scientific recovery claims, a performance
target, release readiness, or new factor-family behavior.
