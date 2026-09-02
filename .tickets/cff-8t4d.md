---
id: cff-8t4d
status: open
deps: [cff-544x]
links: [cff-ejib]
created: 2026-09-02T04:02:00Z
type: chore
priority: 1
assignee: coopwilliams
parent: cff-edoz
tags: [compiler, architecture, refactor]
---
# Extract topology validation and slot layout

## Background in Plain English — WHY

At authoring snapshot e488df3, `compile_template` spans
`src/cffg/compiler.py:632-1936`. This is a source-region anchor, not a promised
future line map. Public orchestration first checks the semiring, expands plates,
and calls the existing core `validate`. The remaining structural front end
builds endpoint/degree/port facts and assigns discrete, Gaussian,
exponential-family, observation, and parameter-edge slots through approximately
line 870. Later lowering code consumes these facts.

This ticket extracts only that structural front end. Registered factor
lowering, specialized plans, learnables, arrays, and final `Arena` assembly
remain for later tickets.

## Outcomes / Acceptance Criteria — WHAT

1. `cff-544x` is closed with compiled values behind a one-way boundary.
2. One topology owner accepts the already expanded, core-validated `Template`
   and returns a named immutable `TopologyLayout`.
3. `TopologyLayout` contains endpoint order, degree/port facts, declared
   parameter-edge/constraint/role facts, edge records, own/opposite slot maps,
   and data-slot candidates. It contains no registered-versus-specialized
   classification, packed parameter value, or partially built `Arena`.
4. Public `compile_template` owns checking the accepted semiring option, plate
   expansion, and the existing core `validate` call as thin orchestration. The
   topology owner owns executable endpoint, degree, exterior-port, and slot-
   layout invariants only. `cff-ejib` and `cff-kfo7` are the sole registered
   and specialized semantic classification/lowering owners, including family-
   specific semiring compatibility.
5. The **New factor or rule family** and **New IR field** change-path rows are
   the affected rows. **Learnable or checkpoint feature** is checked only for
   parameter-edge addresses carried forward. **Runtime numerical or status
   invariant** is N/A because this stage does not execute numeric messages.
6. Flooding/scheduled and direct/`vmap` consumers receive the same final layout
   downstream. Host/JIT is N/A because topology construction remains host-side.
   Latest-schema/supported-legacy checkpoints reconstruct the template before
   this one path.
7. Sparse/vacuous/improper runtime messages are N/A. Float32 does not change
   membership or slot order. The 32-group characterization fixture from
   `cff-ecw0` retains deterministic addresses.
8. Existing focused compiler tests protect HMM, Gaussian, hybrid, recurrence,
   GFE, constraint, external-kernel, and malformed-topology paths through
   old/new observable characterization. No new scientific oracle is required.
9. Endpoint/degree/port/slot failures retain their `cff-ecw0` baseline
   `CompileError` item and reason. Registered/specialized semantic failures
   remain for their respective later owners. No serialized meaning, rule,
   objective, or runtime behavior changes.
10. The topology module is below 1,000 physical lines and the move fits one
    implementation context: one named input, one named output, and one source
    responsibility.
11. Focused compiler/IR/serialization/plate tests and the final short suite
    pass under the repository protocol.
12. `docs/ARCHITECTURE.md` names `TopologyLayout`, its owner, and dependency
    direction in the same commit as the extraction.

## Methodological Suggestions — HOW

These suggestions are optional. An implementer may choose another sound method.

- Use the e488df3 structural region only as a scope anchor. Leave semiring
  checking, plate expansion, and core validation in thin public orchestration;
  move endpoint/degree/port/slot work without moving semantic classification.
- Preserve diagnostic wording through structured errors or a thin compiler
  adapter.
- Keep the returned value immutable so later stages cannot reinterpret
  topology through hidden mutation.

## Future work

`cff-ejib` consumes `TopologyLayout` to lower registered same-arena groups.
Specialized plans remain explicit in `cff-kfo7`.

## Definitions

- **Endpoint table:** Deterministically ordered factor-role endpoints for each
  variable edge.
- **Slot layout:** Addresses for a factor's own message and the opposite
  message it reads.
- **TopologyLayout:** The named immutable output of structural compilation.
