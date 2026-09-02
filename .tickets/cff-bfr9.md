---
id: cff-bfr9
status: open
deps: [cff-kfo7]
links: [cff-ddsq]
created: 2026-09-02T04:17:42Z
type: chore
priority: 1
assignee: coopwilliams
parent: cff-edoz
tags: [compiler, architecture, refactor]
---
# Extract learning and array resolution

## Background in Plain English — WHY

At authoring snapshot e488df3, blob resolution near
`src/cffg/compiler.py:871` and `_resolve_learnables` at lines 1938-2063 are
source-region anchors, not promised future line numbers. The implementation
starts from the exact `cff-ecw0` baseline. At e488df3, the same source values
flow through several lowering branches before learning addresses are known.

After structural plan drafts exist, this ticket gives numeric array loading,
shape checking, parameter/observation packing, member addresses, and declared
learning resolution one owner. It does not assemble the final arena.

## Outcomes / Acceptance Criteria — WHAT

1. `cff-kfo7` is closed with registered and specialized plan drafts passing.
2. One data-resolution owner accepts the expanded `Template`, root/array
   sources, `TopologyLayout`, `RegisteredPlanDrafts`, and
   `SpecializedPlanDrafts`; it returns immutable `ResolvedCompileData`.
3. `ResolvedCompileData` contains packed parameter arrays, observation address
   tables, member/runtime parameter addresses, writable-member facts,
   functional evaluator identity/state metadata, and resolved `LearnPlan`
   values. It contains no schedule or partially initialized `Arena`.
4. Blob precedence, dtype/shape validation, probability-to-log and
   covariance-to-precision transforms, observation rows, and learnable
   source/target pairing have one owner.
5. The affected rows are **New IR field**, **Learnable or checkpoint feature**,
   and the packing portion of **New factor or rule family**. **Runtime numerical
   or status invariant** applies only to preserving dtype-derived arrays; no
   inference status is produced here.
6. Flooding/scheduled and direct/`vmap` consumers receive one resolved data
   value. Host/JIT is N/A for resolution. Latest-schema/supported-legacy
   checkpoints reconstruct the same template and declared learning addresses.
7. Sparse/vacuous/improper runtime messages are N/A. Exact zero probabilities
   retain the repository log-zero representation. Float32 uses dtype-derived
   transforms and guards. The 32-group fixture verifies address/array order.
8. Existing array, bundle, serialization, learnable, agent-spec, Gaussian, EF,
   recurrence, mu3, and compile-failure tests protect old/new observable
   values. No new scientific oracle or learning-recovery experiment is added.
9. Missing blobs, wrong shapes, invalid probability/covariance values,
   unsupported learnable pairs, and unknown evaluator identity fail loudly
   with the named source or model member.
10. The move changes no bundle/checkpoint schema, learning cadence, factor
    statistic, objective, inference rule, or parameter meaning.
11. The data-resolution module is below 1,000 physical lines and fits one
    context: all inputs are complete plan drafts and its one output is resolved
    numeric/address data.
12. Focused array/serialization/learning/compiler tests and the final short
    suite pass.
13. `docs/ARCHITECTURE.md` names the resolution owner and data boundary in the
    extraction commit.

## Methodological Suggestions — HOW

These suggestions are optional. An implementer may choose another sound method.

- Make packers consume typed `ArrayRequirement` values from both lowering
  stages.
- Keep external source loading separate from pure shape/transform functions.
- Preserve the `cff-ecw0` baseline parameter keys and address order field by
  field.

## Future work

`cff-ddsq` combines complete topology, plan drafts, and resolved data into the
immutable arena.

## Definitions

- **ResolvedCompileData:** Immutable numeric arrays and addresses ready for
  final arena construction.
- **Source precedence:** The rule that explicitly supplied arrays override
  file-backed blob paths.
- **Transform:** A declared representation change such as probability to log
  probability or covariance to precision.
