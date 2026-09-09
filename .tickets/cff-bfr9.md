---
id: cff-bfr9
status: closed
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
   payloads, immutable slot/observation prefix arrays, and resolved `LearnPlan`
   values. It contains no schedule or partially initialized `Arena`.
4. Source resolution preserves supplied-array overrides, file blobs, and
   declared initializers in their baseline precedence. It invokes validators,
   shape providers, packers, probability-to-log and covariance-to-precision
   transforms declared by the registered or specialized owner. It owns
   observation/member addresses and learnable source/target pairing, without
   duplicating factor-type selection or reinterpreting a constraint payload.
   4.1. Complete drafts retain source and address declarations for every
        referenced factor, including absorbed members and invalid prior claims
        whose diagnostics occur after blob loading. The resolver does not need
        the mutable discovery state to reconstruct those facts.
   4.2. Registered packers retain the full resolved-blob mapping and list-valued
        members. Legacy EF validation is not silently replaced by the ordinary
        discrete/Gaussian validator.
   4.3. Prefix conversion runs once at its existing late checkpoint, before
        learning validation. `cff-ddsq` consumes these immutable arrays instead
        of moving their possible overflow failures after learning checks.
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
   9.1. Existing failure classes and ordering are preserved, including
        source-dtype covariance checks before float64 widening, unresolved
        observation ports, and prefix overflow before learning validation.
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
- **Prefix array:** Cumulative offsets that map logical slots into flat arrays.
- **EF:** Exponential family, the representation used by the existing scalar
  natural-parameter kernel adapter.

## Notes

**2026-09-09T21:47:49Z**

Input-contract handoff from cff-ejib review: common ArrayRequirement/ArrayInput/ArrayDestination/ArrayRecipe and ShapeRequirement now live in array_requirements.py. Registered ShapeRequirement retains the trusted ParameterSpec and shared immutable role sizes without invoking the provider; callbacks must stay at initializer/packing gates. Registered callbacks currently receive the complete resolved mapping of all model-referenced blobs and list-valued members. Preserve that read surface: narrowing to the current group’s declared role blobs breaks accepted external callbacks (confirmed baseline reproduction reading another factor’s spin_prior). Requirements declare ownership and destinations, not an authorization boundary. Legacy EF pack callbacks and key checks must remain as-is, not silently upgraded to discrete/Gaussian validation. ObservationIntent distinguishes unobserved from observed without rows; ObservationSource.port may be None only when deferred conditioning diagnostics reject the model before row allocation. Gaussian and EF row counters stay independent. Reuse published structural FrozenArrays and role-size maps, do not copy them on final plan construction. Question -> answer -> source: modularity/performance-agent bounded review and baseline/current reproductions in cff-ejib; see tests/test_registered.py full-map, missing-blob/conditioning, and plan-identity cases. Confidence high. Start only after cff-ejib final gate and publication; retain all diagnostic checkpoint ordering and explicit cff-n183/cff-gvch corrected baseline.

**2026-09-09T21:49:11Z**

Correction to optional-port handoff: None does not guarantee an earlier conditioning problem. Existing family=any EF registration can reach row resolution with a discrete output. Preserve the original late KeyError(source.variable) there; never publish an observation address under None. tests/test_registered.py parameterizes the malformed Gaussian/discrete output cases and preserves missing-blob precedence for both.

**2026-09-09T21:53:23Z**

Preparation during cff-ejib final gate at 758d43e; implementation not started. Bounded modularity review identified required draft additions: ordered parameter-prior intents including failures, complete prior names for orphan diagnostics, deferred composite/semiring facts, parameter-edge tags/exclusions, ordered source declarations for all factors including absorbed members, and member/runtime learning address intents. Canonical packing shape is not always the historical initializer fallback. Requirements must preserve both without source inference by runtime key suffix. Suggested resolution package: contracts, sources, addresses/learning, explicit ordered orchestration; specialized numeric packers move beside named family owners, registered math stays kernels/kernel_runtime. Keep every extracted file under1000 physical lines. Question -> answer -> source: compiler.py:215-254,262-267,305-361,421-482,564-640,644-851 and specialized contracts/structural/discovery at701c10f; confidence high.

**2026-09-09T21:53:23Z**

Boundary correction adopted within this extraction plan: prefix conversion currently precedes _resolve_learnables because Python evaluates Arena keyword arguments in order (compiler.py:606-640). A resolve-then-assemble pipeline with prefix conversion reserved for assembly would reverse overflow/learning failures. Resolve immutable prefixes once at the original final checkpoint, before learning; assembly consumes them. Both bfr9 and ddsq acceptance criteria now agree. No new scientific choice or public behavior is introduced. Research source: modularity-agent direct control-flow inspection at701c10f; confidence high. Dependency companion (prerequisite --> dependent): [x] cff-kfo7 --> [in_progress, final gate] cff-ejib --> [blocked] cff-bfr9 --> [blocked] cff-ddsq --> cff-edoz --> engine track --> cff-cfqg. There is no parallel prerequisite within this compiler chain; unrelated ready tickets remain outside scope. Built from tk dep tree --full cff-bfr9, tk ready and tk blocked; dependency records are authoritative.

**2026-09-09T21:53:23Z**

Family-semantics research handoff: retain exact existing validators/tolerances during extraction, not stricter named-write APIs. Dirichlet and learned GFE/TM arrays stay positive raw concentrations; fixed A/c use existing normalization and exact-zero log encoding, with serialized selector/out/in ordering unchanged. Mixture/hybrid owners retain covariance interpretation and absorbed goal_m/goal_W normalization terms. Precision mean accepts any size-one blob before reshape; recurrence owns finite weights/offsets and their shapes. resolve_constraint retains payload meaning and exception distinctions. Resolved output carries exact runtime keys, observation rows/counts, MemberAddress/ParameterAddress transforms/axes/permutations, writable/source/target eligibility, validated constraints and LearnPlans. Mu3 and global-goal identity/ownership pass through unchanged; no particles or posterior initialization and no ownership inferred from equal arrays. Source-dtype covariance checks precede float64 widening; callbacks retain initializer/packing/abstract-shape checkpoints. Question -> answer -> source: formalism-agent bounded inspection of compiler.py:217,270,778,909,954,971; specialized/constraints.py:50; array_requirements.py:25 at701c10f. Confidence high; no fresh paper derivation required.

**2026-09-09T21:58:20Z**

Prefix/learning regression verified on current baseline in 0.32 seconds with no large allocation: one equality factor eq with roles/variables a,b,c and GLUE ports on all three; one declared but unreferenced blob unused; LearnableSpec("bad", "dirichlet", ("unused",)). At cardinality 2**63, compile_template raises OverflowError: Python int too large to convert to C long from prefix_offsets. At cardinality2, it instead reaches CompileError: learnable bad has no factor reading blob unused. The unused blob is never loaded; only small address arrays are constructed. Add this paired mechanism regression before extraction, pinning the literal original learning diagnostic from the baseline. Question -> answer -> source: modularity-agent isolated current compiler probe at83c3e98; confidence high.

**2026-09-09T22:07:02Z**

Ready baseline: cff-ejib is closed. Source commit 701c10f passed 1,755 default tests, with 41 slow deselections and four existing warnings, in 1,005.304 seconds measured wall time. The gate began at 758d43e and is recorded at its ticket-only descendant 2c29d5c. All 19 registered stage cases and 90 final compiler-boundary checks pass; 14 complete arenas equal the pre-extraction compiler.

The registered lowerer is 335 lines; compiler.py is 1,024 lines. Start from the published agent/cff-ejib-registered-lowering branch after PR publication and normal base update. No source changed after the gate. This ticket needs one fresh final gate after its own implementation, not another run of the unchanged baseline.

**2026-09-09T22:08:44Z**

Published base: PR #119, https://github.com/engimatic-systems/cffg/pull/119, branch agent/cff-ejib-registered-lowering. It is stacked on #118 and has the passing final gate described above. Update this base before creating the array-resolution work branch.

**2026-09-09T22:10:17Z**

Starting on agent/cff-bfr9-array-resolution from updated PR #119 branch at 0d8bfe9; validated source remains 701c10f (1,755 passing default tests). Change-path rows: New IR field copies existing declarations only; Learnable/checkpoint owns addresses and source/target pairing; New factor/rule owns declared numeric packing; Runtime numerical/status applies only to preserving source dtype and transforms, with no solver/status change. Flood/scheduled and direct/vmap share one resolved output. Host/JIT compiler pair is N/A. JSON and current/legacy bundles reconstruct through the same compiler; runtime checkpoint data and learning cadence stay unchanged. Sparse/improper messages are N/A; exact-zero log encoding, float32 source covariance checks, 32-group order and existing malformed-input gates are required. Resolution will return immutable data once all old gates pass; private numerical builders stay mutable so premature publication cannot introduce an earlier callback-payload exception. Complete draft metadata must replace discovery-state dependencies and family-name/suffix inference. Prefix conversion occurs before learning, as now specified in both resolution and assembly tickets. Preserve compatibility helper exports and legacy EF callback behavior. First pin the tiny prefix-versus-learning mechanism before moving code, then focused tests throughout and one final dashboard-backed gate.

**2026-09-09T22:11:18Z**

Pinned four pre-extraction regressions in tests/test_resolution.py; all pass in 0.01 seconds on unchanged implementation. The paired equality case checks prefix overflow before invalid learning. A paired legacy EF callback case proves malformed object-array publication occurs after learning validation: an invalid unused learnable wins first; without it, immutable publication raises TypeError. Keep numeric builder results private and mutable until the final resolved-output publication checkpoint. New test file is enrolled in the dashboard; final suite is deferred until after implementation.

**2026-09-09T22:27:40Z**

Declaration foundation implemented before moving orchestration. Completed specialized drafts now carry prior claims/layouts, prior names, deferred composite facts, excluded edges and historical initializer fallbacks. Registered drafts carry ordered SourceRequirements for all referenced factors, including absorbed members; shape callbacks remain unevaluated and covariance source checks are declared. ArrayRequirements carry explicit member/runtime write and learning interpretation; hybrid goals retain their auxiliary kernel handle. AddressBuilder and resolve_learnables consume that metadata without factor-name or key-suffix inference. Across all 14 characterization models, declared addresses, writable membership and LearnPlans match the existing integrated compiler. Source catalog order intentionally follows the input template, while JSON may reorder declarations; the existing registered reconstruction test now compares catalog content separately and keeps exact comparison for every pre-existing draft field. Nineteen resolution tests pass, including the four pre-move gate regressions. Source loading and numeric orchestration still use the compatibility compiler until the next extraction commit. Architecture records this boundary; no final suite yet.

**2026-09-09T22:47:59Z**

Integrated resolution now returns immutable ResolvedCompileData through one ordered pass. Source loading consumes captured requirements; named registered/specialized packers retain mathematical meaning and legacy callback validation. ObservationBuilder and AddressBuilder own row/member facts; prefixes precede learning, and final publication follows learning. Compiler facade shrank from 1,024 to 206 lines; largest new packing module is 294 lines, resolver 198. Final edge tagging and Gaussian/EF interior tables remain assembly work. Architecture change-path rows and boundary sections updated. Fourteen complete arenas match the pre-extraction compiler at 0bcda82 field by field, including arrays, addresses and callback identities. The integrated broad focused run passed 217 tests with two slow deselections; after boundary-test additions and annotation correction, all 125 resolution/registered/specialized/topology/compiler checks pass. The new alias test explicitly copies shared fixture inputs before mutating them, so it does not contaminate later tests. Total new resolution cases: 22. Ruff and diff checks pass; final default gate is next.

**2026-09-09T22:47:59Z**

Bounded reviews: formalism question -> are probability/concentration/covariance semantics and global-goal ownership preserved? Answer: yes by comparison to the prior compiler; fixed normalization, raw learned concentrations, mixture inversion, recurrence/precision validation, source-dtype covariance checks and learning eligibility/layout are unchanged. Modularity question -> are source/callback/error checkpoints retained without rediscovery? Answer: no blocker found; absorbed-source traversal, mutable per-factor shape dictionaries, initializer fallbacks, full-map callbacks and prefix-learning-publication order preserved. Corrected semiring_problem annotation to the completed draft type. Performance question -> is publication duplicated or a whole-template scan added per group? Answer: no; frozen arrays/maps/plans are reused, role/runtime metadata shared. Initializer lookup per blob and factors per learnable retain baseline scaling limitations; no speedup is claimed. Sources: agent comparisons against 0bcda82 in resolution/resolve.py, sources.py, addresses.py and named packing modules; direct 14-model arena parity and boundary identity/alias tests. Confidence high for reviewed paths; no new performance benchmark or scientific oracle was run.

**2026-09-09T23:05:14Z**

Final gate passed: 1,777 tests, zero failures, 41 opt-in slow deselections, four existing Gym deprecation warnings. Pytest986.60 seconds; measured wall999.044 seconds. Source339c46a; runtime record at ticket-only descendant5b3606f, completed2026-09-09T23:04:39.153983+00:00. Dashboard refreshed with22 new cases and30 diagrams in23 galleries. No source/test change after the gate began. All acceptance criteria are satisfied: typed complete inputs, ordered validation, immutable output reused by Arena, numerical/serialization/runtime behavior retained, focused/full checks and architecture updates. Source work is complete; publish the stacked PR on#119, then cff-ddsq consumes this result for final assembly. No additional unchanged suite is needed for publication.
