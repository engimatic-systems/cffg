---
id: cff-k6g5
status: closed
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

**2026-09-14T20:26:29Z**

Cooper's 2026-09-14 steering: users must be able to avoid D12 compilation when they do not select the feature. Preserve the explicit declaration/approximation activation boundary and existing D9 meanings. Existing models without D12 declarations must not enter hybrid lowering/solve compilation. cdc-ogj9 measures early cost and separates fixed-parameter inference from learning/reporting. A cheaper mode within D12 is not yet implemented; do not promise that a no-learning flag removes its inference compilation cost.

**2026-09-15T04:19:09Z**

Pre-implementation handoff: applicable ARCHITECTURE rows are Derived component coordinate and New IR field. This change covers core declaration validation, explicit JSON and canonical hash identity, compact expansion and both variable-scoping paths, supported flat scoped/merge/bond and transactional edits, immutable topology snapshots, recursive bundle rewriting, and named compiler/publication rejection. Specialized/registered lowering, resolution, assembly, Arena/member_of and executable persistence are deferred to 7SW5; no execution is enabled. Host/JIT, flooding/scheduled, single/vmap, learning resume, float32 and numerical scale are N/A here. Current/legacy identity is applicable. Hostile cases: empty component/separator/support, duplicate labels, wrong domains/references, overlapping factor owners, source/pair-selected or nonlinear channels, constraint and association cuts must fail by declaration name; fixed support remains structural through edits, while resolved blob-support checks belong to 7SW5. All supported transport surfaces will ship together in one coherent declaration change. Focused tests and atomic author commits precede an explicit pause for root-owned independent review and final short-suite gate.

**2026-09-15T04:31:33Z**

Implementation handoff for independent review: added the opt-in Template.hybrid_components mapping and frozen component/separator/region/adapter specs. core/hybrid.py owns explicit hybrid-regime-ec-v1/fixed-support identity, named structural admission, and complete reference mapping. R/H/D roles are checked against explicit factor/equality incidence; ordinary categorical/linear-Gaussian adapters are checked through equality wiring. Overlapping owners, source/pair-selected channels, missing or malformed scopes/support, unsupported constraints and association cuts fail by declaration name. JSON/file round-trip and both hash paths canonicalize unordered collections while retaining port order; D9 omission/legacy meaning is preserved. Plates (including chain and single-copy broadcast), supported direct flat scoped/merge/bond, existing transactional edits, immutable topology snapshots and recursive blob rewriting preserve the declarations. Failed direct D12 bonding is atomic. Compiler and bundle save/load guards reject before lowering/array resolution/destination replacement. No execution, numerical thresholds, model builders, learning semantics or recovery fits changed. README, ARCHITECTURE and LIMITATIONS explain authority and unsupported production activation; the new test file is registered in dashboard EPIC_FILES. Final focused validation: uv run pytest -q tests/test_hybrid_declarations.py tests/test_ir.py tests/test_serialize.py tests/test_plates.py tests/test_compose.py tests/test_edits.py tests/test_edit_codec.py tests/test_topology.py tests/test_learning_schema.py tests/test_bundle.py tests/test_hygiene.py -> 191 passed, 1 deselected in 5.18 s (log /tmp/cff-k6g5-final-focused.log). git diff --check passed; tk dep cycle found no cycles. Root owns independent review and the required final short-suite gate, closure and PR publication; no full-suite result is claimed. Implementation is ready for the explicit committed author pause. Remaining resolved-array shapes/support, executable ownership and successful compiled bundles belong to 7SW5; learned resume belongs to CZDQ and compact direct composition to B2FZ.

**2026-09-15T04:36:58Z**

Independent review of author head 84ee7d7 is complete. Root inspected transport, identity, publication guards and documentation; the fresh reviewer inspected schema and structural admission only after the author committed and paused. Two P2 findings require ordinary validation repairs before the final short-suite gate.

1. A source-selected affine channel can pass if the declaration is changed with the factor. In the declared() fixture, add z0d/z0p categorical copies and an owned equality of z0/z0d/z0p, move pz to z0p, select dynamics d by z0d, change right.regime to z0d, change channel ports to (left,right), and expose z1 through an adapter on middle. validate and JSON round-trip incorrectly accept it although recurrence is z0 -> z1 and dynamics selects z0. core/hybrid.py checks each factor against its chosen ports but not the inter-region R/H/D relationship. Enforce the accepted D12.2 destination-selected construction, and add a regression that changes both factor and declaration together.

2. A detached separator can reuse physical regime/state coordinates without a consistency route. In declared(), turn observe into a Gaussian unary on y, remove observation/sensor, clear adapters, add detached_site=(z0,y), and assign observe to a unary region on detached_site. validate and JSON round-trip incorrectly accept this disconnected repeated coordinate. Nonzero separator incidence alone is insufficient. Validate the structural consistency of shared/equality-identified coordinates under D12.2 and add a mechanism regression; do not substitute global connectedness for the required per-coordinate consistency check if connected counterexamples remain possible.

Both findings are supported by the accepted D12.2 scopes and K6G5 acceptance, not new mathematical or numerical requirements. Do not demand universal exactness or add restrictions on unrelated ordinary exterior graphs. Nine additional malformed decoding probes rejected by component name, with no additional finding. Root found no transport loss in explicit Template constructors, both scoping paths, edits, nested bundle rewriting, canonical identity or immutable snapshots; the ordinary-path/early-publication guards are coherent. Compatibility is assessed within the ticket's current-reader/legacy identity contract; no new old-reader forward-compatibility gate is introduced.

Return these repairs to the original K6G5 author, then run focused structural/transport compatibility validation after the final change and commit a clean revised handoff. Pause again for independent re-review. No full-suite gate, D12 numerical solve, compilation benchmark or registered fit has run during review. Root owns the one final short-suite gate after repairs and review acceptance.

**2026-09-15T04:41:47Z**

Independent-review repair handoff: both P2 structural admission gaps are fixed. D12.2 intermediate separators now pair exactly one recurrence output with one dynamics input, preventing a source-selected channel from bypassing admission by changing both its factor selector and its declared ports. A per-coordinate induced region graph now requires all separators sharing a physical/equality-identified coordinate to have a consistency route through regions that retain that coordinate. This does not rely on global connectedness, separator naming, or ordinary exterior connectivity. Coordinate adjacency and recurrence/dynamics incidence are indexed once; separator-domain lookup is also direct rather than scanning all separators per region. Regressions reproduce the exact source-selected rewrite and detached unary from review, plus a connected R/D/R/D chain that drops and reintroduces one physical state; restoring a distinct final state is an accepted positive control with an ordinary exterior adapter. README and ARCHITECTURE document these existing D12 structural requirements. Final focused validation after all edits: uv run pytest -q tests/test_hybrid_declarations.py tests/test_ir.py tests/test_serialize.py tests/test_plates.py tests/test_compose.py tests/test_edits.py tests/test_edit_codec.py tests/test_topology.py tests/test_learning_schema.py tests/test_bundle.py tests/test_hygiene.py -> 194 passed, 1 deselected in 5.18 s (log /tmp/cff-k6g5-review-repairs-focused.log). git diff --check passed; tk dep cycle found no cycles. Author will commit and explicitly pause for re-review. Root still owns the one final short-suite gate, closure and PR publication. No numerical solves, benchmark, recovery fit or full suite ran; no approximation, numerical threshold or ordinary exterior scope changed.

**2026-09-15T04:43:35Z**

Root independent re-review accepted author repair commit 8663b20 with no remaining findings. The R/H/D pairing now rejects the selector-plus-declaration bypass, and indexed coordinate connectivity rejects both disconnected and globally connected broken-consistency examples. The positive two-transition chain and ordinary adapters remain valid. Scope filtering protects malformed declarations before indexing; the direct separator-domain lookup preserves validation while avoiding a per-region global scan. Documentation matches accepted D12.2. Root independently ran the six repair/composition/guard regressions: 6 passed, 29 deselected in 0.03 seconds; /tmp/cff-k6g5-root-rereview.log and .xml. git diff --check and tk dep cycle passed. The author was notified only after re-review completed and has ended edits. Root now owns the one final short-suite gate under the existing 1200-second cap; ticket closure and publication remain pending that result. No numerical qualification or recovery run occurred during review.

**2026-09-15T05:08:21Z**

Final code gate passed at c2af46ece957bb76783540419ae619959f156931: 2,867 passed, zero failures, 16 warnings; pytest 1,144.23 seconds and measured wall 1,145.056 seconds. The unchanged 1,200-second hard cap retained 54.944 seconds of headroom. Four workers used the unchanged not-slow selection with the executable cache empty at start and bounded at 1 GiB. Root announced the launch and waited quietly for the full recorded 1,200-second wait budget without polling. This is the only final short-suite run for K6G5. The timing audit found 8,601 phase records, 2,867 distinct call cases, no duplicates and no non-passing phases. Runtime history and ~/agent-share/cffg-tests.html were refreshed by that run. Evidence: /tmp/cff-k6g5-short-suite.log, /tmp/cff-k6g5-short-suite-summary.json and /var/folders/kg/pmjxhgy506n699dj2g96f2wh0000gn/T/tmp3g9rh4df.timings.jsonl.

Independent review and re-review are accepted; all ticket outcomes pass. Close K6G5 and publish its stacked PR on PR150. Subsequent changes are ticket/runtime-evidence only and do not require another unchanged suite. Production execution remains guarded, and no registered source-recovery fit has run. The next fresh sequential implementation author takes 7SW5 compiler plans, including fixed-parameter bundle reconstruction and prevention of D9 fallback before runtime integration. Current declarations preserve ordinary-model semantics and the explicit D12 activation boundary.

Dependency companion from tk dep tree --full cff-7sw5, tk ready and tk blocked; arrows mean prerequisite --> dependent, and TK retains additional direct dependencies:

[x] A3WZ numerical checkpoint --+
[x] K6G5 declarations ----------+--> [ ] 7SW5 (ready; next author)
[x] OCXL contract -------------+          |
                                      [ ] IDHK --> [ ] BRCF --> [ ] F3JZ
                                                                  /   \
                                                            [ ] CZDQ [ ] ULBZ
                                                                  \   /
                                                                [ ] KUL2
                                                                    |
                                                                [ ] X0IF --> [ ] ZOV5

The two independent ready roots have completed sequentially. 7SW5 was immediately blocked by K6G5 and is now ready; the remaining runtime, learning and composition work still precedes the registered three-fit gate. No mathematical contract, supported scope, approximation or qualification threshold changed.

**2026-09-15T05:13:00Z**

Published as PR151: https://github.com/engimatic-systems/cffg/pull/151, base cff-a3wz-moderate-qualification (PR150), head cff-k6g5-hybrid-declarations. Code, independent review/re-review and the final short gate pass; ticket closed. A bounded read-only compiler-boundary study justified extracting ready cdd-ziiw before 7SW5 implementation, preserving every original compiler/runtime/recovery requirement. Root will assign a fresh sequential author after publishing this ticket-only handoff. No merge performed; all three registered recovery fits remain unused.
