---
id: cff-edoz
status: open
deps: [cff-8t4d]
links: [cff-209l, cff-ws3c]
created: 2026-08-13T04:26:16Z
type: epic
priority: 1
assignee: coopwilliams
parent: cff-cfqg
tags: [architecture, compiler, extensibility]
---
# Compiler track: Give planning and lowering typed owners

## Background in Plain English — WHY

At commit e488df3, `src/cffg/compiler.py` has 2,368 lines. It defines the
compiled plan records and `Arena`, then performs topology layout, registered
and specialized factor classification, per-arena lowering, GFE composite
recognition, learnable resolution, parameter and observation packing, and
final assembly. The file contains 86 references to factor or group type
selection. A change to one family can therefore cross several compiler
responsibilities.

This track preserves compiler behavior while separating those responsibilities
behind typed, one-way boundaries. It starts only after `cff-209l` finishes
the registered discrete and Gaussian factor path. Three context-sized child
tickets first record the e488df3 contract, then move immutable plan values, and
finally extract lowering stages.

## Outcomes / Acceptance Criteria — WHAT

1. The direct dependency `cff-8t4d` is closed because its acceptance criteria
   are satisfied. Its chain closes `cff-ecw0`, `cff-544x`, and
   `cff-209l` first.
2. Characterization evidence pins the compiled plans, addresses, schedules,
   hashes, and failure diagnostics at its recorded baseline before
   implementation code moves.
3. Immutable compiled plan and arena values have a focused owner below compiler
   orchestration and runtime consumers.
4. Topology validation, registered factor lowering, specialized-plan lowering,
   learnable resolution, array resolution, and final assembly communicate
   through named typed values.
5. `compile_template` remains the public entry point and becomes readable
   orchestration below 1,000 physical lines. No extracted source file exceeds
   the repository's 1,000-line limit.
6. Dependency direction is one way: core IR and kernel declarations feed
   compiler stages; compiler stages produce compiled values; engine, schedule,
   estimator, and agent consumers do not import compiler implementation
   helpers.
7. The track changes no serialized model meaning, factor rule, objective term,
   runtime status, learnable cadence, checkpoint schema, or public compiler
   call form.
8. Final evidence preserves HMM, Kalman, fixed-A GFE, hybrid GFE, recurrence,
   structured recurrence, functional `mu3`, hard-constraint, declared-learning,
   AgentSpec, plate, and external-kernel compilation.
9. Compile-time hostile cases remain loud, including unsupported factors,
   constraints, roles, parameters, evaluator metadata, and malformed shapes.
   Runtime sparse-support, improper-input, float32, and scale behavior is not
   redefined by this track and remains covered by the affected integration
   suites.
10. `docs/ARCHITECTURE.md` names each final compiler-stage and compiled-value
    owner after the extraction lands.

## Methodological Suggestions — HOW

These suggestions are optional. An implementer may choose another sound method.

- Follow the child order. Keep the characterization commit independent from
  plan-value movement and stage extraction.
- Use the **New factor or rule family**, **New IR field**, and **Learnable or
  checkpoint feature** change-path rows as preservation checklists.
- Treat flooding versus scheduled execution, single-instance versus `vmap`,
  and latest-schema versus supported legacy checkpoints as consumers of one
  compiled contract.
  Do not add separate compiler interpretations for those pairs.
- Prefer small modules named for topology, registered lowering, specialized
  lowering, learning, arrays, and assembly over a general helpers module.

## Future work

`cff-ws3c` consumes the final compiled-value boundary while extracting engine
owners. New nonlinear-factor implementation remains blocked through
`cff-cfqg`. Performance, resource estimation, and scientific validation
remain with their existing tickets.

## Definitions

- **Arena:** Immutable compiled arrays and address plans derived from a
  serialized template.
- **Characterization evidence:** Tests that record observable behavior at a
  named baseline commit before a refactor.
- **Compiled plan:** Immutable metadata describing how one factor, constraint,
  schedule, evaluator, or learnable executes.
- **Lowering:** Conversion from validated serialized declarations into compiled
  plans and addresses.
- **One-way dependency:** A module relationship without a reverse import or
  ownership cycle.
