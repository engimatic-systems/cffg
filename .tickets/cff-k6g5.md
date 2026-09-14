---
id: cff-k6g5
status: open
deps: [cff-ocxl]
links: []
created: 2026-09-14T01:11:24Z
type: feature
priority: 2
assignee: coopwilliams
parent: cff-zov5
tags: [rslds, inference, hybrid]
---
# Declare serialized hybrid regions and approximation identity

## Background in Plain English — WHY

D12 accepts hybrid-regime-ec-v1 as an explicit approximation. The runtime still executes D9, and no template currently selects the new region ownership. The serialized template must select the mathematics before compilation can derive a plan.

## Outcomes / Acceptance Criteria — WHAT

1. Serialized declarations identify the D12 hybrid scopes, ordinary identity adapters, factor ownership, fixed active-support semantics, and approximation version without inspecting model classes or factor names.
2. Validation rejects unsupported source/pair-selected channels, nonlinear or GFE constraints, overlapping objective owners, and any association-producing factor cut across separate ordinary regime/state ports, naming the offending declaration.
3. JSON round-trip, structure hashing, compact plate expansion, supported flat edits/composition, and recursive bundle blob rewriting preserve every declaration field and reference. Reordering leaves canonical identity unchanged; scoped renaming preserves meaning while changing the name-sensitive hash. Existing D9 templates retain their meaning. Unknown declaration versions and premature D12 compilation or bundle publication fail by name without replacing an existing destination. Successful compiled and learning-state bundle round-trips remain mandatory in cff-7sw5 and cff-czdq, respectively.
4. The Derived component coordinate and New IR field change paths classify validation, serialization, plates, edits/composition, topology, specialization, registration, resolution, assembly, and persistence; runtime host/JIT, flooding/scheduled, and single/vmap are N/A until the execution ticket. Current/legacy semantic identity is applicable.
5. Sparse support, empty declarations, invalid shapes, and support-changing edits are validated; float32 and numerical scale are N/A because this ticket adds no numerical solve.
6. README explains declaration construction and unsupported compositions; ARCHITECTURE records authority. Focused serialization/composition tests and one final short-suite gate pass.

## Methodological Suggestions — HOW

These suggestions are optional; another sound method meeting the outcomes is welcome.

Keep declarations independent of derived array addresses. Apply the existing New IR field path to every semantic field.

## Future work

cff-7sw5 consumes these declarations, transports their identity into compiled plans, and verifies fixed-parameter bundle reconstruction. cff-czdq verifies successful public D12 bundles with learning identity and state resume after cff-f3jz. These downstream outcomes preserve the complete bundle/executable guarantee while keeping this ticket independent of compiler implementation. Direct composition of unexpanded compact templates remains the existing separate cff-b2fz work. Local kernels (cff-ffsg) and execution (cff-f3jz) remain separate.

## Definitions

- D12: the hybrid-regime-ec-v1 mathematical contract in docs/DERIVATIONS.md.
- EC: expectation consistency, matching declared moments between regional beliefs.
- Hybrid separator: one boundary carrying regime probabilities and regime-conditioned state moments.
- Adapter: an explicit equality region connecting a hybrid boundary to an ordinary categorical or Gaussian copy.
- JJ / xi: the Jaakkola–Jordan logistic quadratic bound and its squared auxiliary coordinate.
- FE / ledger: free energy and the single assembly of its owned energy/entropy terms.
- GFE: generalised free energy with P-substitution, excluded from D12.
- IR: serialized intermediate representation of model semantics.
- Arena / member_of: compiled array storage and the map from factors to its members.
- JIT / vmap: JAX compiled and batched execution surfaces.
- CFFGZ: the project bundle format containing template, data, and optional learning state.
- Gauge: redundant numerical coordinate that leaves a belief unchanged.
- Krylov: a matrix-free iterative linear solve using bounded stored vectors.
- D9 / D10: the current factor-local recurrence approximation and the retired whole-model candidate.
- Moment / covariance: an expected feature or product, and its centered second moment.
- Semantic identity: the versioned meaning of a declaration, executable, or persisted statistic.



## Notes

**2026-09-14T11:02:30Z**

Independent declaration research — Question: what existing transport/identity boundaries does this ticket cover? Answer: add a frozen spec and additive Template field, explicitly encode every semantic field, canonicalize unordered collections, reject unknown/current-malformed identities, and omit absent fields so D9 JSON/hash remains unchanged. Scope references through plates.expand_plates/_scope_copy/_expand_one, compose.scoped/merge/bond, plates._rename_variable, transactional edits and edit codec where actions are added, and bundle._rewrite_blob_paths. Snapshot declarations without mutable aliases in TopologyLayout; named premature-execution rejection prevents silent D9 lowering. No mathematical choice requiring Cooper was found. Sources core/ir.py:94/494, core/serialize.py:87/195/346/353/369, core/learning_identity.py:11/34/43, core/plates.py:50/87/141/266, core/compose.py:38/49/123/148, core/edits.py:180/207/226/265, bundle.py:93, topology.py, D12.2-6. Confidence high.

Criterion3 boundary clarified without removing the eventual guarantee: structure_hash deliberately retains names, so renaming changes identity while preserving scoped meaning; insertion-order reordering does not. Direct compact scoped/merge is already rejected and belongs to cff-b2fz; cover compact JSON and expansion then supported flat composition. Public save_bundle verifies through load_bundle, whose _load_archive calls compile_template. K6G5 therefore owns JSON/hash/blob-rewrite preservation and named premature-publication rejection, 7SW5 owns successful fixed-parameter empty-learning bundle reconstruction and executable identity, and CZDQ owns full learned D12 public save/load/resume. Loading itself does not run Engine inference. Sources bundle.py:181/288/336, learn.py:666, core/serialize.py:346, compose._require_flat; confidence high. This is an implementation-boundary correction, with no supported-scope, approximation or numerical-gate change.

Focused authorities: test_serialize.py, test_learning_schema.py identity/hash/migration and nested-scope tests, test_plates.py, test_compose.py, test_edits.py, test_edit_codec.py, test_topology.py immutable declaration snapshots, test_bundle.py blob rewriting/transactional failures. Current/legacy declaration identity applies. Runtime host/JIT, flooding/scheduled, single/vmap and numerical dtype/scale remain N/A; resolved data support belongs to compiler resolution.

Dependency handoff from tk dep tree --full cff-czdq, tk ready and tk blocked; tk dep cycle clear.
Legend: prerequisite --> dependent; conjunctions require both paths.
[x] qualified kernels/references --> [ ] 8OIS (ready repair) --> Y15D --> A3WZ
[ ] K6G5 (independent ready declarations) -----------------------------+
                                                                      |
                                              A3WZ + K6G5 --> 7SW5
                                                              |
                                                     IDHK --> BRCF
                                                              |
                                                            F3JZ
                                                        /           \
                                                     CZDQ           ULBZ
                                                        \           /
                                                            KUL2
                                                              |
                                                            X0IF --> ZOV5
Implementation still waits for the early numerical checkpoint; only independent research ran alongside the repair. TK records remain authoritative; the diagram abbreviates extra direct prerequisites.

**2026-09-14T18:30:55Z**

Independent preflight research (read-only at 3e52251; ticket contract at 037496a). Question: which entry points can lose a D12 declaration, and where should premature compilation/publication fail? Answer: cover Template/core exports and validate (including recursive plates); to_dict/from_dict, JSON/file wrappers and both structure hashes; expand_plates; direct scoped/merge/bond plus compose/unroll; transactional apply_edits/check_edits; topology snapshots; compile_template; save_bundle/load_bundle. Exact constructors in serialize.from_dict, plates.expand_plates, compose.scoped/merge and bundle._rewrite_blob_paths enumerate Template fields and otherwise drop an added declaration. Plate _scope_copy and _expand_one must rename/collect/collision-check references and preserve tied versus per-copy blob semantics. Chain wiring delegates to bond; single-copy broadcast uses a distinct _rename_variable path. Both need declaration-aware rewriting or named rejection.

Validation trap: compact expansion validates before wiring but has no final validation; direct scoped/merge/bond also lack the wrapper validation used by compose/unroll. Validate unsupported ownership/association cuts at these direct surfaces after rewiring. Existing edits deep-copy then validate transactionally; declaration-specific codec actions are conditional, while destructive/replacing/constraint/port/support-changing edits must preserve or reject semantic references. TopologyLayout/build_topology must snapshot the declaration immutably: _immutable.freeze does not recursively freeze arbitrary dataclasses merely because frozen=True.

Identity recommendation: explicitly encode all semantic fields; canonicalize only semantically unordered collections; omit absent declarations so D9 documents/hashes keep their meaning. Name-sensitive structure hashing should change on scoping while recursive blob-data removal preserves relocation identity. Decode unknown/malformed versions by name without assigning D12 meaning through legacy learning migrations. Place the named premature-compilation guard after expansion/core validation and before topology/specialized discovery in compile_template; valid declarations remain editable/loadable. Apply rejection during bundle initial preflight and retain archive verification before sibling temporary-file creation or os.replace. save_bundle's current load_bundle verification already reaches compile_template, and load wraps the named error in BundleError.

Affected: IR/validation/JSON/hash/legacy identity, compact plate and supported flat composition/edit transport, topology snapshots, compiler/bundle rejection, exports/docs. Conditional: new edit codecs only if adding new action vocabulary. Deferred to 7SW5+: executable specialized/registered ownership, resolution/packing/assembly/Arena and successful compiled bundles. Runtime host/JIT, schedules, single/vmap, numerical dtype/scale, learning/checkpoint resume are N/A here; unexpanded compact composition remains B2FZ. K6G5 is coherent as one declaration-preservation change. If the implementing worker cannot fit it in one context, first split a behavior-preserving transport refactor; do not expose a public D12 field while some copies can drop it.

Sources (all src/cffg/ unless marked tests): core/ir.py:494; core/__init__.py:8; core/validate.py:38; core/serialize.py:330,346; core/plates.py:50,266; core/compose.py:49; core/edits.py:180; topology.py:86; _immutable.py:187; bundle.py:93,181,336; compiler.py:103; core/learning_identity.py:43. Regression anchors: tests/test_learning_schema.py:234,273 and tests/test_topology.py:33. Confidence: high for existing transport/validation/save-order facts and scope map; guard placement is a prospective recommendation, and whether to split is a medium-confidence context-size judgment. No tests, compilation, solver, source edit or mathematical-schema decision occurred. Root will retain implementation sequential after the numerical checkpoint.
