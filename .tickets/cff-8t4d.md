---
id: cff-8t4d
status: closed
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
   variable/constraint/role facts, raw edge records, own/opposite logical slot
   maps, segment lengths, and data-slot candidates. It contains no
   registered-versus-specialized classification, inferred parameter-edge
   shape/axis tags, packed value, or partially built `Arena`. Specialized
   discovery supplies those semantic tags after topology construction.
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
   JSON and model bundles reconstruct templates before this one path.
   Latest-schema/supported-legacy agent checkpoints restore runtime data
   against an already compiled, builder-defined model.
7. Sparse/vacuous/improper runtime messages are N/A. Float32 does not change
   membership or slot order. The 32-group characterization fixture from
   `cff-ecw0` retains deterministic addresses.
8. Existing focused compiler tests protect HMM, Gaussian, hybrid, recurrence,
   GFE, constraint, external-kernel, and malformed-topology paths through
   old/new observable characterization. No new scientific oracle is required.
9. Endpoint/degree/port/slot failures retain their `cff-ecw0` baseline
   exception owner, item, reason, and reporting order. Core validation retains
   `ModelError` for malformed degrees/endpoints; compiler-owned executable-port
   failures retain `CompileError`. If topology runs before semantic admission,
   it returns ordered diagnostics for aggregation at the existing compiler
   gate rather than raising earlier. Registered/specialized semantic failures
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

`cff-kfo7` first discovers specialized ownership from `TopologyLayout`.
`cff-ejib` then lowers the unclaimed registered groups; the specialized owner
completes drafts using that membership without repeating classification.

## Definitions

- **Endpoint table:** Deterministically ordered factor-role endpoints for each
  variable edge.
- **Slot layout:** Addresses for a factor's own message and the opposite
  message it reads.
- **TopologyLayout:** The named immutable output of structural compilation.

## Notes

**2026-09-09T17:38:47Z**

Preparation during cff-544x gate (no topology implementation started). Question: what can move without semantic reclassification? Answer: compiler.py at 315b082:152-160 degree/port indexes; 162-169 constraint declarations only; 252-258 executable-port diagnostics returned for existing aggregation; 287-333 sorted endpoints, declared-domain raw EdgeInfo, own/opposite maps and logical counts; DATA candidates from 332-372, without conditioning decisions; declared-domain segment widths/prefixes from 1404-1416. Parameter carriers already declare flat cardinality, so shape/axis tags can be attached later by building a new edge mapping. Keep admission (170-251), shape inference (262-285), conditioning validation (334-369), absorption, classification, observation rows (1000-1013,1199-1207), DELTA checks and entropy exclusions outside topology. Observation row numbers follow lowered group order, not sorted-variable topology. Error-order caveat: factor semantic errors precede executable-port errors at the 259-260 gate; conditioning errors survive array resolution until the 462-463 gate. Preserve both boundaries and the distinction between core ModelError and compiler CompileError. Endpoint order sorts factor/role names; declared message-role order and hard-constraint numeric x0...xN order are separate conventions. Source: bounded read-only modularity-agent inspection of current compiler and core/validate.py; confidence high. TopologyLayout should snapshot declaration facts and hold ordered endpoints/degrees/ports, raw edges/own-opposite slots/counts/prefixes, ordered DATA candidates, and deferred port diagnostics. Use cff-544x publication rather than mutable nested IR aliases. Before implementation, update intended branch base and recheck ready status; these are planning notes, not completed extraction.

**2026-09-09T17:52:18Z**

Starting from updated agent/cff-544x-immutable-values at 39d5bf2 (PR #114). Architecture rows: New factor or rule family and New IR field; Learnable/checkpoint checked for carried parameter-edge addresses; Runtime numerical/status N/A. Duplicated surfaces: flooding/scheduled and direct/vmap consume the same final arena; host/JIT compiler N/A; JSON/bundle reconstruction and current/supported-legacy checkpoint consumers retain the one compiler and their existing restore owners. Sparse/vacuous/improper runtime messages N/A; dtype-independent topology and 32 distinct groups remain required. Use the recorded extraction/diagnostic-order boundary; do not move conditioning semantics or observation row numbering into topology.

**2026-09-09T18:06:23Z**

Implementation handoff: cffg.topology owns one immutable TopologyLayout with declared variable/constraint/port snapshots, Mapping-typed FactorDeclaration role bindings, ordered endpoints/degrees/ports, raw edges, own/opposite maps, three slot counts, Python segment-length tuples, ordered DATA candidates and deferred executable-port diagnostics. Compiler retains semantic admission, inferred Dirichlet tags, conditioning decisions and observation row numbers; tags create new EdgeInfo records instead of mutating topology. Int64 prefix allocation stays at final assembly. Review found and fixed an early OverflowError for a 2**63-cardinality declaration that should first report legacy/unterminated-port errors; the exact precedence regression is now a test. Independent bounded review confirmed the original error and order are restored. The tuple endpoint representation is explicitly rendered as the prior list in its existing error diagnostic. Focused compiler/topology/IR/serialization/plate/dashboard run: 121 passed; Ruff and diff/dependency checks pass. Architecture change-path rows now name topology. Final dashboard gate pending; no numerical/serialized/runtime behavior is intentionally changed.

**2026-09-09T18:11:29Z**

Final short-suite gate running at committed implementation head 8cd8919, after the 121-test focused pass. No source edits after launch. The publication PR will stack on #114; cff-kfo7 remains the next serial dependency, not an implementation started during this gate.

**2026-09-09T18:23:47Z**

Completed at implementation head 8cd8919: dashboard-backed default short suite passed 1,676 tests, 41 opt-in slow tests deselected, four existing Gym deprecation warnings. Pytest 979.15 seconds; measured wall 991.115 seconds. Dashboard refreshed with 13 new topology cases and zero failures. No source edits after the passing run. Focused run: 121 passed; existing 44 characterization cases unchanged. Topology extraction is complete and will be published stacked on PR #114. Next cff-kfo7 is blocked by separate pre-existing correctness tickets cff-n183 and cff-gvch; cff-n183 awaits Cooper’s choice of correct cross-group goal sharing versus explicit temporary rejection. No inference behavior was changed here.
