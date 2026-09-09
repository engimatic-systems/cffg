---
id: cff-bfr9
status: open
deps: [cff-ejib]
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

After structural plan drafts exist, this ticket gives array loading and the
execution of declared validators/packers, member addressing, and declared
learning resolution one owner. Family owners retain the meaning of their
numeric transforms and constraints. It does not assemble the final arena.

## Outcomes / Acceptance Criteria — WHAT

1. `cff-ejib` is closed with registered and specialized plan drafts passing.
2. One data-resolution owner accepts the expanded `Template`, root/array
   sources, `TopologyLayout`, `RegisteredPlanDrafts`, and
   `SpecializedPlanDrafts`; it returns immutable `ResolvedCompileData`.
3. `ResolvedCompileData` contains packed parameter arrays, observation address
   tables, member/runtime parameter addresses, writable-member facts,
   functional evaluator identity/state metadata, validated hard-constraint
   payloads, and resolved `LearnPlan` values. It contains no schedule or
   partially initialized `Arena`.
4. Source resolution preserves supplied-array overrides, file blobs, and
   declared initializers in their baseline precedence. It invokes validators,
   shape providers, packers, probability-to-log and covariance-to-precision
   transforms declared by the registered or specialized owner. It owns
   observation/member addresses and learnable source/target pairing, without
   duplicating factor-type selection or reinterpreting a constraint payload.
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
- Include an external-kernel initializer/override/blob case and a hard
  constraint whose k/table payload comes from a blob. Both must reach assembly
  without a new family branch in the resolver.
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

## Notes

**2026-09-09T21:47:49Z**

Input-contract handoff from cff-ejib review: common ArrayRequirement/ArrayInput/ArrayDestination/ArrayRecipe and ShapeRequirement now live in array_requirements.py. Registered ShapeRequirement retains the trusted ParameterSpec and shared immutable role sizes without invoking the provider; callbacks must stay at initializer/packing gates. Registered callbacks currently receive the complete resolved mapping of all model-referenced blobs and list-valued members. Preserve that read surface: narrowing to the current group’s declared role blobs breaks accepted external callbacks (confirmed baseline reproduction reading another factor’s spin_prior). Requirements declare ownership and destinations, not an authorization boundary. Legacy EF pack callbacks and key checks must remain as-is, not silently upgraded to discrete/Gaussian validation. ObservationIntent distinguishes unobserved from observed without rows; ObservationSource.port may be None only when deferred conditioning diagnostics reject the model before row allocation. Gaussian and EF row counters stay independent. Reuse published structural FrozenArrays and role-size maps, do not copy them on final plan construction. Question -> answer -> source: modularity/performance-agent bounded review and baseline/current reproductions in cff-ejib; see tests/test_registered.py full-map, missing-blob/conditioning, and plan-identity cases. Confidence high. Start only after cff-ejib final gate and publication; retain all diagnostic checkpoint ordering and explicit cff-n183/cff-gvch corrected baseline.

**2026-09-09T21:49:11Z**

Correction to optional-port handoff: None does not guarantee an earlier conditioning problem. Existing family=any EF registration can reach row resolution with a discrete output. Preserve the original late KeyError(source.variable) there; never publish an observation address under None. tests/test_registered.py parameterizes the malformed Gaussian/discrete output cases and preserves missing-blob precedence for both.
