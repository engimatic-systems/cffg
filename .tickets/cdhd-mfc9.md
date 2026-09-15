---
id: cdhd-mfc9
status: open
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
