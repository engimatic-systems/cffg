---
id: cff-fbmr
status: closed
deps: [cff-6vgp]
links: []
created: 2026-08-22T15:57:07Z
type: task
priority: 1
assignee: coopwilliams
parent: cff-0oqo
tags: [rslds, inference, schedule]
---
# Integrate structured recurrence inference through engine schedules

## Background in Plain English — WHY

cff-6vgp provides the D9 structured recurrence rules without changing engine
behavior. The engine must now run those rules as one coherent inference path.
The integration must preserve the same fixed point and free-energy value under
synchronous and scheduled execution. It must also distinguish an exact
acyclic regime chain from a categorical loop, where local Bethe beliefs are not
one exact path posterior.

## Outcomes / Acceptance Criteria — WHAT

1. Compiler planning identifies each connected categorical recurrence
   component and records whether it is acyclic. The exact structured-chain
   claim is available only for an acyclic component.
2. The engine dispatches cff-6vgp's pair-potential and hybrid sum-product rules
   through both categorical roles and the Gaussian drive role.
3. Synchronous and scheduled execution converge to the same messages,
   singleton beliefs, pair beliefs, Gaussian messages, and BFE under the
   runtime's dtype-scaled certificate.
4. A scheduled acyclic component may fuse its local messages into one
   forward/backward pass, but the factor-local pair regions and ordinary edge
   corrections remain the belief and energy owners. K=8, T=100 requires
   `O(T*K^2)` work and memory for that categorical pass.
5. `_make_bfe` uses the structured recurrence node energy once per factor and
   the existing categorical/Gaussian edge corrections once per interior edge.
   A K=2/K=3 chain matches enumeration and `-log Z` for fixed drives.
6. Converged, budget-exhausted, diverged, and infeasible statuses remain
   distinct. `Engine.run` classifies an already-empty categorical support as
   infeasible. The direct factor/JIT update classifies that invalid local
   input, or a non-finite potential, as divergence. Neither path can return a
   coherent-looking pair belief.
7. Single-instance and `vmap` execution agree. Flooding and any fused schedule
   preserve literal zeros and produce the same result in float32 and x64 within
   their named tolerances.
8. P-substitution on recurrence still fails during compilation. Existing
   supported GFE composites can coexist and receive ordinary recurrence BFE
   terms in the mixed ledger.

## Methodological Suggestions — HOW

These suggestions are optional. The implementer may choose another sound
method.

- Keep pair-region ownership in the existing recurrence plan. Add scheduling
  metadata rather than a serialized monolithic chain factor.
- First land an unfused fixed-point path. Add a fused forward/backward stage
  only when its parity can be checked against that path.
- Reuse exact-support utilities instead of turning log zeros into finite
  floors.

## Future work

cff-ifbr will expose the factor-local pair derivation as a public readout and
use it for learning statistics. The Gaussian-drive inference message already
uses that pair in this ticket. cff-63dd will bind the behavior change to
persisted identities. cff-h0hx will run development-only recovery evidence
after both land.

## Definitions

- **Acyclic component:** A connected recurrence subgraph with no categorical
  cycle; its local sum-product beliefs are marginals of one global path belief.
- **Fused pass:** One scheduled forward/backward execution of the same local
  factor messages, used to avoid one whole-chain flooding diameter per update.
- **JIT path:** The JAX-compiled execution surface used by the engine.

## Notes

Architecture handoff: compiler grouping, slot addressing, `Arena`,
`member_of`, `_recurrence_sweep`, `_make_step`, scheduled batch/scan stages,
schedule classification, support checks, status, and `_make_bfe` are
applicable. IR factor spelling, parameter packing, plate expansion, editing,
and visualization are checked N/A because no serialized field changes.

Duplicated surfaces: host/JIT, synchronous/scheduled, single/`vmap`,
fixed/uncertain drives, both categorical roles, Gaussian writes, node/edge
energy, exact zeros, and both dtypes must agree.

**2026-08-22T17:24:07Z**

Implemented at 9657db4 from base 7ec4260. The compiler now derives connected recurrence component metadata in `Arena.recurrence_components` and maps each grouped factor through `RecurrencePlan.component_ids`; self-loops and parallel categorical edges are loopy, equality branches remain interior, and mean-field/GFE factors are unary-cavity boundaries. The unfused engine path rematerializes one factor-local D9 pair from the two categorical cavities plus current drive moments, sends sum-product messages through both categorical roles, and uses that same pair for the Gaussian write and structured node energy. Pair arrays are not persisted or publicly exposed in this ticket. Ordinary categorical/Gaussian edge corrections retain their prior ownership. Fixed-drive K2/K3 engine BFE agrees with independent `-log Z` to 4.44e-15/9.33e-15 in x64 and 5.72e-6/4.29e-6 in f32. Production sum-product message/pair errors are <=2.11e-15 in x64; the remaining singleton-product learning gap is 0.0119--0.0458 and belongs to cff-ifbr. One pass is O(P*K^2) work and transient pair memory; no K^3 allocation and no chain fusion were introduced. Public `Engine.run` reports an existing empty support as infeasible, while direct factor/JIT status is diverged for empty local support or non-finite parameters; vacuous Gaussian startup no longer bypasses either check. Flood/mixed, host/JIT, K2/K3, f32/x64, literal-zero, asymmetric-axis, self-loop/parallel/equality/MF/GFE, K8/T100, Gaussian-write, and full BFE ownership regressions pass. Final dashboard gate: 1,458 passed, 39 deselected in 995.25s; focused D9 set: 52 passed in 42.29s. cff-ifbr must expose the same deterministic pair derivation and feed its pair probabilities into public coefficient statistics without reconstructing an outer product; cff-63dd owns persisted semantic identity. No consumer experiment or artifact was run.
