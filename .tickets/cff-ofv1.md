---
id: cff-ofv1
status: open
deps: [cff-ws3c]
links: [cff-cfqg, cff-o7hs, cff-ahps]
created: 2026-09-02T04:02:01Z
type: task
priority: 1
assignee: coopwilliams
parent: cff-cfqg
tags: [architecture, extensibility, verification]
---
# Prove the bounded factor extension paths

## Background in Plain English — WHY

The compiler and engine tracks aim to reduce the number of central modules
that a future factor must change. File movement alone does not prove that
outcome. The final gate must exercise the new seams and fail if central
type-name dispatch returns.

This ticket verifies two intentionally different extension paths. A same-arena
factor uses registered kernel metadata without central edits. A factor that
spans arenas, changes the objective through GFE, or represents structural
constraints uses an explicit specialized plan with one owner at each stage.
The ticket does not pretend that both paths fit one universal callback.

## Outcomes / Acceptance Criteria — WHAT

1. `cff-ws3c` is closed with the compiler and engine architecture tracks
   satisfied.
2. The test-only discrete and Gaussian factors delivered by `cff-209l` prove
   the same-arena path.
   2.1. Each registers outside production compiler and engine modules.
   2.2. Each compiles, runs, reports its objective term, exposes its declared
        readout, and survives JSON and CFFGZ reconstruction.
   2.3. Each agrees with its independent finite-enumeration or analytic oracle.
   2.4. Each declares sparse-support, improper-input, float32, and
        uncomfortable-scale behavior.
3. A source-inspection test proves that the test-only factor type names do not
   appear in central compiler orchestration, public engine orchestration,
   validation maps, schedule builders, objective assembly, or readout dispatch.
4. The shipped same-arena built-ins use the same registered path as the
   test-only factors. They do not retain a second type-name implementation for
   messages, energy, parameter packing, observations, or readouts.
5. A compact extension matrix covers representative specialized paths:
   5.1. `recurrence_link` represents a multi-arena factor;
   5.2. one P-substitution composite represents GFE-specific lowering and
        objective behavior; and
   5.3. one hard constraint represents structural lowering.
6. For each specialized representative, the matrix names exactly one
   classification and lowering owner, one runtime-execution owner, and one
   objective owner. Source checks fail if its semantic selection is copied
   back into unrelated central orchestration.
7. The extension matrix states which change-path surfaces remain required for
   a new same-arena factor and for each specialized plan class. A surface is
   marked not applicable only with a concrete reason.
8. The HMM, Kalman, fixed-A GFE, hybrid GFE, recurrence, structured
   recurrence, functional `mu3`, hard-constraint, learning, agent,
   serialization, and checkpoint fixtures named by the architecture tracks
   retain their existing results. Their existing mathematical authorities are
   reused; this proof does not add a new recovery experiment or oracle for
   those shipped paths.
9. Flooding and scheduled, direct and one-instance `vmap`, host and JIT
   status, and latest-schema and supported-legacy checkpoint pairs retain the
   applicability recorded by the architecture tracks.
10. Literal zero support, vacuous and improper inputs, non-finite values,
    float32, and uncomfortable-scale tests retain their documented values or
    loud statuses.
11. `docs/ARCHITECTURE.md` contains the final extension matrix and module
    ownership map. `README.md` and `docs/LIMITATIONS.md` distinguish the
    supported same-arena registration path from explicit specialized plans and
    the remaining unsupported combinations.
12. This ticket adds no production factor family, new message or objective
    rule, performance target, scientific validation, or public-release claim.
13. Focused kernel, compiler, runtime, schedule, objective, readout,
    serialization, model, and hostile-input tests pass. The dashboard-backed
    short suite passes after the final code change.
14. The ticket fits one implementation context because production behavior is
    frozen: it adds only the two cff-209l extension proofs, source ownership
    checks, and the compact specialized-plan matrix.

## Methodological Suggestions — HOW

These suggestions are optional. An implementer may choose another sound method.

- Reuse the external test kernels required by `cff-209l` to demonstrate that
  later extraction did not weaken their path.
- For shipped families, reuse existing authority tests and old/new observable
  characterization from the architecture tracks. A newly noticed scientific
  gap becomes a separate ticket unless it shows the extension-path proof is
  structurally unsafe.
- Make source checks target ownership invariants rather than a total ban on
  every type string in family-local modules.
- Use the **New factor or rule family** row as the primary extension matrix.
  Also record the **Runtime numerical or status invariant**, **New IR field**,
  and **Learnable or checkpoint feature** surfaces that apply to each
  representative.
- Keep specialized-plan checks explicit and small. Do not invent a general
  multi-arena plugin contract in a verification ticket.

## Future work

Closing this ticket closes `cff-cfqg` and unblocks implementation ticket
`cff-o7hs` plus the downstream factor-breadth roadmap `cff-ahps`.
Research and derivation work that does not implement a runtime factor remains
independent.

## Definitions

- **Extension matrix:** A table that maps one extension class to the production
  owners, derived surfaces, and required evidence it must change.
- **Ownership invariant:** A testable rule that one semantic decision belongs
  to one named module.
- **Same-arena path:** Registration for a factor whose messages all use one
  supported representation.
- **Specialized path:** Explicit compiler, runtime, and objective handling for a
  factor or constraint that does not fit the same-arena contract.
