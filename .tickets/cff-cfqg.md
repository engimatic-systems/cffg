---
id: cff-cfqg
status: open
deps: [cff-ofv1]
links: [cff-ahps, cff-209l, cff-edoz, cff-ws3c]
created: 2026-08-13T03:59:30Z
type: epic
priority: 1
assignee: coopwilliams
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
5. Compiler planning and lowering use typed, one-way stages after
   characterization pins the serialized-to-arena contract at baseline commit
   e488df3.
6. Engine execution, schedule construction, objective accounting, and readouts
   use focused owners after characterization pins the numerical, support,
   status, and diagnostic contracts at its recorded baseline.
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
- Commit characterization evidence before each extraction. Move one owner at a
  time and keep compatibility imports until consumers migrate.
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

- `cff-xabn` owns versioned declared-learning semantics and persistent arrays.
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
