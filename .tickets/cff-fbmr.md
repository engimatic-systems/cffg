---
id: cff-fbmr
status: in_progress
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
6. Converged, budget-exhausted, and diverged statuses remain distinct. Empty
   support or a non-finite potential is divergence in host and JIT paths and
   cannot return a coherent-looking pair belief.
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

cff-ifbr will expose the installed pair beliefs to Gaussian and learning
consumers. cff-63dd will bind the behavior change to persisted identities.
cff-h0hx will run development-only recovery evidence after both land.

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
