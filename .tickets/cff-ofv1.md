---
id: cff-ofv1
status: in_progress
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
   2.5. The fixtures retain the nonbuiltin role names, parameter keys, unequal
        role sizes/new signature, and observation-plus-packing cases required
        by cff-209l; they cannot be renamed built-in kernels.
   2.6. The existing external EF factor remains in the proof. All three arena
        bindings reconstruct through their declared sources/initializers.
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
   6.1. It maps original factors/constraints, absorbed goals/priors, edge
        exclusions, and shared-parameter terms to the cff-kfo7 claims and
        existing mathematical authorities. No factor or shared parameter is
        executed or charged twice because it appears in several drafts.
7. The extension matrix states which change-path surfaces remain required for
   a new same-arena factor and for each specialized plan class. A surface is
   marked not applicable only with a concrete reason.
   7.1. It explicitly includes initializer shape inference, array/blob
        precedence, named parameter writes, learning addresses, Colony,
        persistent-byte estimation, manifests, and checkpoint consumers.
   7.2. Same-arena registration does not imply new variational methods:
        each cell names accepted constraints and semirings. Generalized
        moment matching and new P-substitution combinations remain downstream.
8. The HMM, Kalman, fixed-A GFE, hybrid GFE, recurrence, structured
   recurrence, functional `mu3`, hard-constraint, learning, agent,
   serialization, and checkpoint fixtures named by the architecture tracks
   retain their existing results. Their existing mathematical authorities are
   reused; this proof does not add a new recovery experiment or oracle for
   those shipped paths.
9. Flooding and scheduled, direct and the cff-sf29 multi-instance/multi-structure
   Colony fixture, host and JIT
   status, and latest-schema and supported-legacy checkpoint pairs retain the
   applicability recorded by the architecture tracks.
   9.1. The cff-ii9t independent-stop/invalid-neighbor batch case retains
        per-instance status and state through the shared termination policy.
   9.2. Engine and Colony use one result finalizer, and lower runtime value
        owners do not import the facade.
   9.3. The existing discrete/Gaussian chain records retain flat bytes,
        registerized local carry and one final scatter, and traced-executable
        reuse at unchanged shapes/dtypes/static controls. No speedup is inferred.
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
14. This gate reuses extension factors and the batch, failure, and execution-
    structure checks delivered by its prerequisites. Its new work is the
    cross-seam ownership checks and matrix documentation, not another set of
    fixture implementations or scientific experiments.

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

## Notes

**2026-09-09T03:29:39Z**

PR #108 research — question: can factor registration alone prove objective accounting? Answer: composite ownership and exclusions also need a checked map. Hybrid GFE excludes absorbed goals/output entropies; functional mu3 charges one shared KL across slices. Sources at 7885bf5: engine.py:3761, 3950; tests/test_mu3.py:131; docs/DERIVATIONS.md:1220; Part I equations (8) and (25). Confidence: high. Reuse existing authorities; the map is a refactor preservation gate, not new scientific recovery evidence.

**2026-09-10T09:14:50Z**

Pre-implementation handoff prepared while cff-3ola is under its final gate; cff-ws3c remains the blocker. Applicable ARCHITECTURE rows: New factor or rule family (primary), Runtime numerical or status invariant, New IR field and Learnable or checkpoint feature. New work is targeted cross-seam source/import/call ownership checks plus an owner/capability/consumer matrix; reuse all three existing external factors and their independent oracles. Reviewed duplicated pairs: flooding/scheduled, direct/multi-instance and multi-structure Colony, host/JIT facts, latest/supported-legacy restoration all retain their prerequisite applicability. New ownership checks have numerical hostile-input behavior N/A because no rule, solver or numerical guard is added; existing sparse support, improper/non-finite, float32 and 10x-scale authorities remain required. Preserve the corrected global-goal, P-substitution validation, per-instance constraint-data and failure-diagnostic baselines; do not restore pre-repair goldens. Prepared source checks distinguish arena and transport labels from original factor-type dispatch. Matrix must make scalar-EF initialization compatibility, unsupported new learning/projection combinations, deterministic-only Colony and specialized non-BP scheduling explicit. No implementation has begun on this ticket branch yet.

**2026-09-10T09:32:13Z**

Implementation complete on the final runtime prerequisite 80d6afe (#133), pending the final full gate. Added 30 cross-seam checks in test_extension_ownership.py: test-only identities absent from every production source; no original factor-type interpretation in facades/transport; no duplicate ordinary built-in numerical selection; acyclic lower owners with no facade imports; exactly owned specialized lowering/update/objective calls; shared flat/scheduled registered executors. Existing fixtures were reused, not reimplemented. ARCHITECTURE now has semantic-owner, accepted-mode, required-consumer and composite-ownership matrices, with README/LIMITATIONS links and scope boundaries. Initializer/precedence/write/learning/Colony/byte/manifest/checkpoint surfaces and concrete N/A reasons are explicit. Focused final selection: 161 passed in 61.98 seconds, covering new checks, all three external factors, compiler joins, runtime contracts, common finalization and Gaussian scan/batch contracts. Ruff passes; canonical-doc test links resolve; no production source file changed. The numerical/serialized preservation authorities and hostile cases remain the prerequisite tests. Starting one dashboard-backed final short suite after the final test change; first output check at 20 minutes, allowance 1,470 seconds if needed. No independent agent review or runtime speedup is claimed.
