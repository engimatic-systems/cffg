---
id: cdhd-mfc9
status: in_progress
deps: [cdpf-gzix, cff-a3wz]
links: []
created: 2026-09-15T08:02:15Z
type: task
priority: 2
assignee: coopwilliams
parent: cdhd-hzw0
tags: [hybrid, numerics, qualification]
---
# Define general D12 incidence and evaluator input contracts

## Background in Plain English — WHY

The compiled declaration drafts preserve general D12 connections, but the qualified Coordinate and evaluator use two message banks along a homogeneous chain. That layout cannot represent a three-branch shared boundary, a unary-only component, or a regime-selected unary factor on the intermediate H boundary of an R/H/D pair. Live ordinary adapters also need explicit port and objective inventory. Resolving parameter arrays before defining this boundary risks publishing plans that cannot express the declared approximation.

This bounded ticket defines inspectable unbound numerical layout and evaluator-input contracts from existing declarations. It preserves the complete admitted graph and identifies precisely when the unchanged qualified chain evaluator is applicable. It executes no numerical solve and makes no new qualification claim.

## Outcomes / Acceptance Criteria — WHAT

1. Immutable records describe region/separator incidence, ordered ports, support and origin specialization requirements, active message coefficients with one common-constant gauge per directed message, and auxiliary ownership for each recurrence region. No time-axis assumption determines membership.
2. Recurrence, dynamics, unary and ordinary identity-adapter inputs have explicit ownership. Objective inventory preserves each regional energy/entropy, degree-based separator correction and adapter port/count. Ordinary incoming/outgoing slot intentions remain distinct from held unary evidence and future accepted runtime state.
3. A structural constructor consumes admitted declarations or reviewed cdpf-gzix drafts without loading arrays or calling numerical kernels. It preserves the graph's existing scope and produces named errors only for malformed contracts. It does not reject a valid declaration simply because it is outside the current chain evaluator.
4. A separate applicability result identifies a lossless mapping to the unchanged qualified chain layout when available, including endpoint site ownership, R/H/D elimination conditions, ordering, support and fixed-shape assumptions. Otherwise it records the explicit general-evaluator requirements; no fictitious transition, component concatenation, moved H unary or dropped adapter makes a false chain mapping pass.
5. Structural regressions include the actual admitted three-branch graph, one-label unary-only graph, K>=2 unary-only graph, an H-boundary regime-selected unary, multiple independent components, both adapter domains, renaming/reordering, and sparse fixed support. Sizes vary incidence, label count and state dimension independently without allocating numerical runtime buffers.
6. The evaluator boundary specifies inputs/outputs needed to reuse the existing iterative policy and its complete certificates. It names derivative forward/transpose actions and coordinate partitions without copying policy, executing callbacks or implementing numerical formulas. Stored state, temporary scratch and currently unknown inventory remain distinct.
7. Existing qualified numerical source files, Coordinate.from_sites, solver policy, source inventories and all accepted/specification artifacts remain unchanged. Metadata is not declared to be a qualified numerical implementation. Public compiler/bundle/runtime guards remain active; no recovery fit runs.
8. Before implementation the author records the architecture row and duplicated-surface inventory below. Canonical ownership documentation and focused structural tests pass, then independent review follows a clean committed pause. Root runs one final short suite under the unchanged 1200-second cap.
9. Before closure, the resulting concrete contract is used to create bounded dependent evaluator, policy-binding/tiny-qualification and moderate-qualification tickets under the parent checkpoint. Each future numerical assignment has explicit dependencies and acceptance criteria. This ticket does not close its parent checkpoint by defining types alone.

## Methodological Suggestions — HOW

Use declaration-derived incidence tables and existing immutable plan records. Keep general metadata distinct from the historical chain Coordinate. Read only the applicable D12.3-.5 equations and local API boundaries; delegate any needed paper-level question as bounded research. Another sound method meeting the outcomes is welcome. In particular, prove the exact chain mapping conditions rather than recognizing model names.

Architecture: Derived component coordinate applies at the declaration/draft-to-unbound-numerical-layout boundary. Checked pairs: serialization identity is consumed unchanged; no resolution/assembly/publication, host/JIT arithmetic, synchronous/scheduled execution, single/vmap evaluation, lifecycle/state/ledger/readout or current/legacy checkpoint activation. Future numerical and production consumers are explicitly listed in the contract. Hostile inputs: singleton/exact sparse support and invalid/empty support are structural cases; signed/vacuous/improper numeric values are deferred to named numerical children; dtype-dependent coefficient/storage descriptions include native float32 and float64 without performing arithmetic. Larger graph/K/d metadata must remain bounded.

## Future work

The parent numerical checkpoint owns general evaluation, derivatives, feasible initialization, complete iterative solve and independent tiny/moderate evidence. cff-7sw5 uses the qualified interface for actual parameter/support resolution and assembly. cff-idhk/brcf/f3jz own accepted state, production objective/readouts and public activation; cff-ulbz owns the separate exterior-mixture EC binding. Recovery remains cff-x0if.

## Definitions

- R/H/D: a recurrence region, its intermediate hybrid separator H=(destination label, previous state), and destination-selected affine dynamics region.
- Unary: a factor acting on one declared hybrid separator.
- Incidence: one ordered region-to-separator connection.
- Gauge: the common additive message constant removed without changing a belief.
- Auxiliary: a recurrence's squared logistic-bound coordinate, later solved jointly with messages.
- Evaluator: the unbound numerical interface that supplies beliefs, residuals, energy, learning statistics and certificates.
- Derivative actions: a residual Jacobian applied to a vector and its transpose applied to a vector.
- Fixed support: the exact labels allowed to carry positive mass.
- JIT/vmap: JAX compiled and batched execution.



## Notes

**2026-09-15T08:03:56Z**

Bounded read-only audit at 9f7597dd288ae382fea372414f8239f78bed6ef3 by the completed discovery author, before the separate unary repair. Question -> can the qualified chain solver consume every already admitted declaration by compiler indexing alone? Answer -> no. Three-branch shared Y needs more than two adjacent-time message banks and separator correction degree-1=2. Unary-only K=1 or K>=2 needs no transition, while Coordinate.from_sites requires T>=2,K>=2. A core-admitted regime-selected unary on intermediate H=(j,x) has no input in the grouped TransitionParameters and cannot be moved to endpoint Y sites without changing scope. Live adapters require explicit incidence, ordinary-slot I/O and objective inventory; held exterior cavities folded into sites do not certify the coupled ordinary objective or stationarity. Multiple chains must remain separate, and mixed layouts need explicit compatible grouping rather than false transitions.

Sources -> tests/test_hybrid_discovery.py branched and one_label_unary regressions; core/hybrid.py:_scope_consistency_problems/_component_problems accepts graph incidence and validates recurrence/dynamics pairing. hybrid_joint.py:Coordinate.from_sites fixes [T,K,d], two [T-1] banks and recurrence auxiliary indexes; hybrid_separator.py:evaluate uses shifted consecutive cavities and interior-chain entropy corrections. DERIVATIONS D12.3 gives the exact destination-selected elimination conditions; D12.4 uses general degree-based counting; D12.7 gives generic region-port stationarity; D12.5 owns explicit ordinary adapter entropy and moment equations. hybrid_natural supplies individual adapter kernels; hybrid_joint_solver.cgls already accepts derivative callbacks but the surrounding policy invokes the chain evaluator/linearizer and assumes auxiliary coordinates at the tail. Confidence -> high for these concrete reproduced interface gaps and source assumptions. Existing tiny/A3 results remain valid for their recorded layouts; this audit does not establish a mathematical defect or numerical failure on the new layouts.

Adopted action -> cdhd-mfc9 defines immutable general incidence and evaluator inputs first. Parent cdhd-hzw0 then requires bounded evaluator, policy-binding/tiny and moderate-qualification children, created from the concrete interface before implementation. cff-7sw5 now depends on this numerical checkpoint, preserving Cooper's early-feasibility-before-production order. Reuse the existing iterative policy and fixed certificates; new evaluator behavior needs its own qualification. Preserve prior artifacts/source inventories and all three recovery attempts. Do not narrow declarations, claim a generic wrapper is already qualified, or rewrite historical evidence to pass source checks. Ordinary diagnosis and repairs remain authorized; actual mathematical/scope/threshold decisions or unresolved numerical inability come back to Cooper with evidence. No new runtime or numerical program was executed by this research.

**2026-09-15T08:27:36Z**

Published as PR153, https://github.com/engimatic-systems/cffg/pull/153, stacked on PR152 with base cdd-ziiw-hybrid-plan-foundation and head cdpf-gzix-hybrid-discovery. Discovery cdpf-gzix and repairs cdhd-fyq6/cdhd-0ios are closed. Final source gate at8d71a97 passed all2946tests in1128.324seconds, with no test selection or threshold changes; subsequent commits only record tickets/runtime evidence. No unchanged short-suite rerun is required for publication.

Next ready D12 root is cdhd-mfc9 under numerical checkpoint cdhd-hzw0. A fresh sequential author should use the published PR153 branch as its intended base, update it before branching, record the pre-edit architecture/hostile-input handoff, and define only the bounded immutable incidence/evaluator contract. Existing numerical sources/qualification evidence and all execution guards remain unchanged. No real numerical evaluator, State, compiler activation or recovery fit is authorized within that metadata ticket. Its concrete contract must feed bounded numerical children before closure; defining metadata cannot close the checkpoint epic. Source-backed research, future work and qualification boundaries are in that ticket. All three registered recovery fits remain unused; nothing in this compiler-discovery step establishes recovery performance.

**2026-09-15T08:33:03Z**

Additional bounded read-only research at published base28a280d clarifies the chain applicability result. Question -> exactly which structural facts support a lossless mapping to the unchanged chain API, and which remain unresolved? Answer -> pair R second port to D first port while preserving declared direction. For the existing direct contraction, H is private to those two incidences. Contracted transitions must form one connected directed path with distinct endpoints, one incoming and one outgoing link at each interior Y. Undirected degree<=2 is insufficient: two outgoing branches require an unjustified transition reversal. Cycles and independent components cannot be concatenated into one evaluation.

Endpoint canonical unary factors may sum into sites with every original owner/normalizing constant retained; D12.4 cancels unary entropy against its extra separator incidence. Extra H incidence requires explicit treatment. A general canonical H potential U(j,x) does not inherently invalidate D12.3's conditional elimination identity, but the grouped potential/energy must include it and the unchanged TransitionParameters API has no input for it. A categorical-only U(j) or Gaussian-only U(x) may be transported to a shared endpoint coordinate by matched moments; that is an algebraic inference, not an implemented or qualified mapping. Therefore report an outstanding H-site mapping requirement rather than declaring the admitted graph invalid or silently moving it. Regime-selected Gaussian U(j,x) is generally joint in destination label and previous state and has no endpoint-site scope.

Adapters remain named equality regions with both port moments, incoming cavity/outgoing quotient and entropy ownership. Held incoming cavities define only a conditional internal solve. For example, an ordinary Gaussian factor connecting chain endpoints leaves its own messages/moment equations unsolved when cavities are frozen. Current joint certificates have no ordinary-adapter stationarity block. A structural chain mapping must expose this exterior obligation, not claim complete coupled qualification.

Metadata supplies owner identities, ordered ports, equality copies, K/d, declared support, adapter domains/slots and parameter intentions. It can establish incidence/coefficient counts but not resolved sites, fixed origin values, numerical validity, held-versus-joint exterior policy or certificates. Existing Coordinate requires T>=2,total K>=2,homogeneous d>=1; singleton active support is distinct from total K=1. Preserve original sparse label identities and named unknown numerical obligations.

Sources -> core/hybrid.py:_scope_consistency_problems; DERIVATIONS D12.3 (line2642), D12.4 (2664), D12.5 (2758); hybrid_separator.py:TransitionParameters/evaluate; hybrid_joint.py:Coordinate.from_sites and context/evaluation certificates. Confidence -> high for API/declaration conditions and explicit adapter ownership; algebraic transport of separable H unaries is an inference needing an implemented mapping and applicable tests. No files/tests/numerical programs or fits were used by research. All three recovery fits remain reserved. Use this result in cdhd-mfc9's precise applicability contract; do not expand that ticket into numerical implementation.

**2026-09-15T08:34:38Z**

Pre-implementation handoff: use ARCHITECTURE change-path row "Derived component coordinate" at the reviewed draft -> unbound incidence/evaluator contract boundary. Implement metadata in new modules outside the frozen hybrid_ numerical family. Consume existing HybridComponentDraft parameter intentions and declaration/copy/slot identity unchanged. Preserve general incidence, active support with original labels, ordered region ports, one common-constant gauge per outgoing region-port message, recurrence auxiliary ownership, region/adapter/separator objective ownership, and named chain applicability plus unresolved numerical obligations.

Duplicated surfaces: declaration/draft identities and both categorical/Gaussian adapter domains apply; multiple components remain separate; future evaluator inputs/outputs, derivative actions and coordinate partitions are inventoried. Serialization/JSON/hash/scoping identity is consumed unchanged, so edits there N/A. Resolution/assembly/publication, host/JIT arithmetic, synchronous/scheduled execution, single/vmap evaluation, lifecycle/accepted State/ledger/readout, and current/legacy checkpoint activation are N/A in this metadata change. Existing numerical producers, solver policy, source/spec/accepted evidence, public compiler/bundle/runtime guards are unchanged. Future numerical modules require explicit source inventories regardless of filenames.

Hostile inputs: empty/invalid support fails with named malformed-contract errors; exact sparse and singleton support retains label identities and gauge counts. Signed/vacuous cavities, improper local beliefs and nonfinite numeric values are named deferred evaluator obligations, with no array inspection. Native float32/float64 describe future coefficient bytes explicitly; no arithmetic guards are introduced. Incidence count, total K and d vary independently through shape-only metadata, without allocating dense coefficients, numerical runtime buffers or parameter arrays. Inputs/outputs, possible accepted state and temporary scratch are distinct; unresolved requirements stay None. Focused structural tests and dashboard EPIC_FILES registration precede an atomic committed pause; root owns review, final short-suite gate, numerical child tickets, closure and publication. No recovery fit runs.

**2026-09-15T08:49:27Z**

Research distilled from root's completed independent stable-source audits; neither audit ran numerical programs. Question -> which old solver assumptions must the future evaluator binding preserve besides evaluate/linearize callbacks? Answer -> a complete simultaneous target separate from the rectangular residual; current dynamic certificates with maximum(), finite primal FE, array-valued status replacement, fixed accepted/trial/terminal pytree shapes/dtypes, generated and explicit start semantics, accepted frozen residual/coordinate scaling, and an explicit nonnegative squared-xi partition. The old policy assumes every coordinate after message_size is squared xi, so exterior unknowns cannot be appended blindly. Empty auxiliary blocks appear reduction-safe but valid empty coordinates still need complete properness/energy/certificate behavior; this is not numerical qualification. Scratch includes evaluator payloads and derivative captures in addition to the old CGLS three N plus one M vectors. Sources -> hybrid_joint_solver.py:142-207,216-266,290-299; hybrid_joint.py:163-169,260-303; hybrid_joint_storage.py:29-60. Confidence -> high on API/source facts, medium on unexecuted empty-layout behavior. These obligations now appear in evaluator_contract metadata/protocol; none of the numerical sources changed.

Question -> which existing local kernels can implement separate general R/H/D scopes? Answer -> recurrence_potential already constructs the R potential and normalize/collapse_label/quotient supply its projections. Current auxiliary_statistics assumes a grouped [x;y] belief through source_region, and stable R energy is embedded in transition_expectation, so separate R auxiliary/energy extraction is needed. Destination-selected affine coefficients and whitened energy also exist there, but lift/marginal assume two label axes, requiring explicit same-j D construction/projection. gaussian_site can represent a selected Gaussian unary directly at H, retaining its amplitude without an invented 1/K prior. Both ordinary identity kernels return reduced beliefs and signed hybrid/ordinary targets. Sources -> hybrid_separator.py:62-88,114-142,226-238,273-315; hybrid_natural.py:360-384,507-590; DERIVATIONS D12.2-.5. Confidence -> high on these API facts, medium on future ticket sizing; no new missing-math claim or scope decision follows.

Root also verified source-integrity gates: test_hybrid_joint_solver.py:143-160 selects current tiny report004 with exact SOURCE hashes; d12_moderate_layouts.contract verifies prospective producer hashes. Future protected-source extraction/policy changes therefore require a new truthful tiny report and regenerated prospective descriptors before their final suite. Preserve historical report004/A3 evidence and do not claim it qualifies changed producers. This metadata change edits no numerical/source/spec/accepted evidence artifact.

**2026-09-15T08:49:27Z**

Implementation handoff for independent review: incidence_contract.from_draft consumes reviewed HybridComponentDraft records, retaining ordered region/adapter ports, original sparse labels, one common-constant gauge per hybrid directed message, shape-only coefficients and fixed-origin requirements, recurrence auxiliary blocks, parameter intentions, and region/separator/adapter/equality objective ownership. Ordinary incoming/outgoing slots and the adapter's ordinary degree increment stay explicit; the full ordinary correction is unknown. chain_applicability returns a conditional internal directed-path mapping or named general requirements, without narrowing declaration admission. Cycles, outward branches, independent paths, mixed shapes, unary-only K=1/K>=2 and additional H inputs are represented. Endpoint unaries and held adapter site placements keep original owners. Evaluator metadata/protocol names input/output payloads, simultaneous targets, rectangular residual partitions, complete certificate obligations, derivative actions and explicit auxiliary constraints. Coupled exterior dimensions, accepted State and scratch remain unknown. Neither contract executes arrays, kernels, callbacks, initialization, solve, derivatives, lowering or compilation.

Changed canonical ownership/scope documentation in ARCHITECTURE and LIMITATIONS; registered test_incidence_contract.py in dashboard EPIC_FILES. Final focused validation after the last code edit: uv run pytest -q tests/test_incidence_contract.py tests/test_hybrid_discovery.py tests/test_hybrid_plans.py tests/test_hybrid_declarations.py tests/test_dashboard.py -> 163 passed in 1.95s (19 new structural cases). These are focused results, not a fresh short-suite result. Root owns independent review, one final short suite under the unchanged 1200-second cap, dependent-ticket creation, closure and stacked PR. Existing numerical files, Coordinate.from_sites, policy, source inventories, qualification artifacts, and compiler/bundle/runtime guards are unchanged. All three registered recovery fits remain unused.

Concrete recommendations for root's bounded dependent tickets (not new assignments in this change):
1. Separate R-scope auxiliary/statistic and stable-energy kernel extraction. Depends on this contract. Make existing grouped callers delegate without changing their numerical interface or mathematical meaning; qualify independent R moments/energy/derivatives on sparse support and native float32. Include new source-bound tiny evidence/prospective descriptor regeneration if protected sources change.
2. Same-label D-scope construction, stable energy and projection. Depends on this contract and any shared primitives from 1. Represent D(j,x,y) directly; prove projection/energy and D12.3 parity against independent tiny references, retaining signed cavities, properness and centered arithmetic. Include applicable source-integrity updates rather than rewriting historical evidence.
3. General internal incidence evaluator and derivative actions. Depends on 1 + 2. Consume the exact region/port/objective/auxiliary records and satisfy the named internal output/residual protocol, including selected H unaries, unary-only, branches and disconnected components. Reuse adapter kernels for held incoming cavities, explicitly labeling conditional objectives and leaving full exterior stationarity unclaimed. Test objective counting, statistics, complete targets and forward/transpose agreement independently before any solver binding.
4. Coupled ordinary adapter/exterior boundary and certificates. Depends on 3. Specify the currently unknown exterior coordinate/residual/payload inventory for admitted categorical/Gaussian neighbors and bind both adapter port equations plus original exterior objective owners. Include an ordinary Gaussian factor coupling two internal endpoints to distinguish a held solve from a coupled one. Exterior mixtures remain cff-ulbz. If this exceeds one context, separate the boundary definition from its numerical implementation before assignment.
5. Shared iterative-policy binding and new tiny qualification. Depends on 3 + 4. Reuse the existing numerical policy through the evaluator interface, adapt complete targets/status replacement and explicit auxiliary partitions without copying policy, qualify generated and explicit starts plus rejection/exhaustion/divergence and returned-point complete certificates on newly supported scopes. Protected old-source changes require a new truthful tiny report and prospective descriptors; keep prior evidence immutable.
6. Prospective moderate float32 solve/storage qualification. Depends on 5. Predeclare independent graph/K/d cases and initialization variants, distinguish held/coupled exterior cases, exercise full properness/FE/statistics/certificates, and measure logical retained state/scratch, compiled memory and process peaks separately, including derivative captures. Keep the 1200-second gate and all thresholds/recovery allowances unchanged.
These children complete cdhd-hzw0 before production parameter resolution/assembly in cff-7sw5 and runtime integration in cff-idhk/brcf/f3jz; metadata alone cannot close the parent checkpoint. No numerical child or dependency has been created by this author; root owns that roadmap refinement before closure.

**2026-09-15T08:55:28Z**

Independent review completed after the author committed and paused at b77f94ae6d62778603fe986d4ba27ca8845295b8. Fresh reviewer checked immutable/type boundaries, adapter/exterior obligations, unknown storage and no-execution/source preservation. Root independently checked chain applicability, original-label gauges, degree ownership and auxiliary offsets. Root's 19 structural tests passed (log /tmp/cdhd-mfc9-root-review.log, XML /tmp/cdhd-mfc9-root-review.xml), and 12 additional metadata-only asymmetric source/destination support cases matched an independent enumeration of at-risk sticks. No new numerical program or recovery fit ran.

One P2 requires repair within this ticket. incidence_contract.from_draft checks parameter-member coverage and duplicate identities but not complete source/destination/runtime consistency. Starting with stages(declared()), replace recurrence r's ArrayRequirement.inputs with (): from_draft accepts it and describe_evaluator silently omits both recurrence inputs. Removing destinations or changing runtime keys to an undeclared key also passes. These are malformed intermediate contracts, not lawful missing data: root confirmed that an actual template omitting both recurrence blob roles is rejected by core.validate with named missing-weights/offsets errors. Preserve every required parameter-role intention and verify that source roles, member runtime mapping and parameter destinations agree; reject incomplete/inconsistent records with component-and-factor-named diagnostics. Add focused mechanism regressions for the three reproduced cases and partial omissions as appropriate. Reuse existing parameter-role authority/conventions; do not change declaration admission or introduce a second numerical policy.

The reviewer otherwise found the scope, immutable publication, adapter/exterior obligations, storage honesty and protected-source boundary sound; 19 independent focused tests passed. Both reviews are now complete. Root notifies the original author and requests only this bounded repair, final focused checks, updated handoff and a new committed clean pause; no review of live edits. Root will re-review after the pause, create the bounded numerical children from the documented handoff, and run one final short suite after all relevant edits. All three registered recovery fits remain unused.

**2026-09-15T08:59:34Z**

P2 review repair complete; ready for root's independent re-review after this committed pause. from_draft now checks each region/recipe against the existing core FACTOR_TYPES.param_roles signatures, requires exact source/member-runtime role coverage, and requires unique runtime keys matching named parameter-only destinations. This rejects omitted roles even when both source and runtime entries were removed. It preserves original source order, registered/specialized parameter conventions, shape providers and unresolved addresses; no numerical transforms, callbacks or array resolution run. Errors identify the component and factor; entirely missing factor requirements list the missing owners. ARCHITECTURE documents the validation boundary.

Regression mechanisms cover recurrence, dynamics, categorical unary, Gaussian unary and selected Gaussian H unary: empty/partial sources, partial runtime roles, matching source/runtime omissions, empty/partial destinations, payload destinations, undeclared runtime keys and aliased runtime keys. Final focused validation after the last code edit: uv run pytest -q tests/test_incidence_contract.py tests/test_hybrid_discovery.py tests/test_hybrid_plans.py tests/test_hybrid_declarations.py tests/test_dashboard.py -> 207 passed in 1.89s, including 44 new repair cases (63 total incidence-contract cases). git diff --check and tk dep cycle passed. These remain focused results; no full suite was run.

This repair changes only incidence metadata validation, its tests, canonical ownership documentation and this handoff. Existing numerical producers, source inventories, specification/accepted evidence and public compiler/bundle/runtime guards remain unchanged. All three recovery fits remain unused. Root still owns re-review, creation of the previously recommended bounded numerical children, the one final short suite, closure and stacked PR. The ticket remains in_progress until those landing steps finish.

**2026-09-15T09:02:12Z**

Root independently re-reviewed committed repair 6c6117003eb5cf5a5770e77961393460833681eb after the original author paused. Accepted with no remaining findings in this scope. Required source-role signatures come from existing core.ir.FACTOR_TYPES.param_roles, selected by the already declared region kind and array recipe. Source roles and member runtime roles must match; runtime keys must be named, unique and correspond one-to-one to named parameter-only destinations. Missing source/runtime pairs together are rejected as well as isolated omissions. Component/factor diagnostics preserve the ownership boundary; no array resolution, registry numerical callback or transform executes.

Root independently ran uv run pytest -q tests/test_incidence_contract.py tests/test_extension_ownership.py --junitxml=/tmp/cdhd-mfc9-root-rereview.xml: 93 passed in1.62seconds; log /tmp/cdhd-mfc9-root-rereview.log. The unchanged single-owner architecture checks pass. Base-relative audit /tmp/cdhd-mfc9-source-audit.json found no protected numerical, experiment, source/specification/evidence, public compiler/bundle or execution-guard changes. git diff --check and tk dep cycle passed. Earlier independent review also verified12 asymmetric source/destination at-risk inventories without numerical evaluation.

Root now notifies the original author after completed re-review. Before closure, root will create the documented bounded numerical children (separating coupled-exterior definition from numerical binding) and run one final short suite after all relevant source edits under the unchanged1200-second cap. Defining metadata alone does not close cdhd-hzw0. No numerical qualification program or recovery fit has run in this ticket; all three registered attempts remain unused.

**2026-09-15T09:03:52Z**

Adopted bounded numerical split from the reviewed cdhd-mfc9 interface and two independent stable-source audits, before closing the metadata prerequisite. Explicit children now exist under cdhd-hzw0:
- cdic-ikhw: separate R auxiliary/statistics and stable energy, depending on cdhd-mfc9.
- cdic-1w3r: same-label D construction, energy and projection, depending on cdic-ikhw.
- cdic-7y77: general internal evaluator/derivatives, depending on both R and D.
- cdic-3e79: coupled ordinary-port metadata and objective ownership, depending on the internal evaluator.
- cdic-nuel: numerical ordinary/exterior coupling, depending on that definition and the internal evaluator.
- cdic-za1z: one shared iterative policy and new complete tiny qualification, depending on internal and coupled evaluation.
- cdic-w79t: prospective moderate float32 graph solves and actual storage/compilation evidence, depending on the shared-policy qualification.

The coupled-exterior definition and numerical binding are separate bounded assignments. Moderate harness preparation must be split before implementation if the later concrete case/evaluator scope requires it. Every future author checks scope before editing; no entire checkpoint epic is assigned to one worker. Protected-source R/D or policy changes include a new truthful current tiny report and prospective descriptor regeneration before their final code gate, preserving historical evidence. New moderate source-bound evidence remains explicit downstream work. No coefficient, probability, approximation, stopping, qualification or recovery threshold is changed here.

Legend: prerequisite --> dependent. Built from tk dep tree --full cdhd-hzw0, tk ready and tk blocked; retained original direct dependencies remain authoritative. No cycles.

[x] FFSG + [x] A3WZ + [x] K6G5 + [x] cdd-ziiw --> [x] cdpf-gzix (PR153)
                                                               |
                                                    [~] cdhd-mfc9 (reviewed; gate pending)
                                                               |
                                                        [ ] cdic-ikhw (R)
                                                               |
                                                        [ ] cdic-1w3r (D)
                                                               |
                                                        [ ] cdic-7y77 (internal evaluator)
                                                               |
                                                        [ ] cdic-3e79 (ordinary contract)
                                                               |
                                                        [ ] cdic-nuel (coupled evaluator)
                                                               |
                                                        [ ] cdic-za1z (shared policy/tiny)
                                                               |
                                                        [ ] cdic-w79t (moderate evidence)
                                                               |
                                                        [ ] cdhd-hzw0
                                                               |
                                                        [ ] 7SW5 --> [ ] IDHK --> [ ] BRCF --> [ ] F3JZ
                                                                                              /   \
                                                                                        [ ] CZDQ [ ] ULBZ
                                                                                              \   /
                                                                                            [ ] KUL2
                                                                                                |
                                                                                            [ ] X0IF --> [ ] ZOV5

cdhd-mfc9 is the ready D12 root awaiting its final short suite; cdic-ikhw is next after closure. cdh-kvy5 remains an independent nonblocking audit/research branch. Implementation stays sequential; independent research/review can proceed in parallel only under the established author-pause rule for review. All three registered recovery fits remain unused. Prior A3 chain results remain valid for their exact source/layouts and do not certify these new graph layouts. This roadmap addition records missing implementation and qualification, not a mathematical defect or scope revision.
