---
id: cff-f3jz
status: open
deps: [cff-ocxl, cff-7sw5, cff-ffsg, cff-a3wz]
links: []
created: 2026-09-14T01:11:24Z
type: feature
priority: 2
assignee: coopwilliams
parent: cff-zov5
tags: [rslds, inference, hybrid]
---
# Execute hybrid EC plans with one objective and truthful solver status

## Background in Plain English — WHY

D12 local kernels and compiled ownership are insufficient without a graph solve and a complete objective ledger. cff-a3wz first qualifies the complete message/xi solver without production binding. This ticket integrates that qualified solver and reports its actual terminal state through the engine.

## Outcomes / Acceptance Criteria — WHAT

1. Generic engine execution consumes the declared plan without an rSLDS model branch, selecting one approximation and one belief-array authority.
2. Production reuses the solver qualified by cff-a3wz for D12.7–8 jointly, preserving gauge removal, dtype-scaled properness checks, feasible initialization, bounded local/global/linear/line-search budgets, and globalization. Engine integration distinguishes converged, budget-exhausted, and diverged; non-finite objective/residual is divergence. It does not introduce a separate numerical policy or weaken the prerequisite's certificates.
3. The ledger owns precisely D12 regional energies, labeled entropies, hybrid separator corrections, adapter entropies, and ordinary exterior corrections. It matches the independent tiny objective and stationary dual at the returned coordinate.
4. Synchronous/scheduled, host/JIT, and single/vmap execution agree on messages, internal readouts, support, FE, and terminal status. Lifecycle initialization, binding, support checks, assignment scoring, schedule classification, and estimator/runtime consumers are included in the handoff.
5. Sparse support, support changes, inactive xi, signed/vacuous cavities, float32, and 10–100x scale obey D12.6. No dense trajectory covariance/Jacobian or silent budget-to-success conversion is introduced. New long globalization regressions are slow and opt-in.
6. The Derived component coordinate and Runtime numerical or status invariant inventories include every duplicated execution surface. README/ARCHITECTURE/LIMITATIONS describe the approximation and solver limits; focused tests and one final short-suite gate pass.

## Methodological Suggestions — HOW

These suggestions are optional; another sound method meeting the outcomes is welcome.

Keep Engine responsible for public status and ledger assembly. Prefer local or matrix-free residual derivatives with explicitly bounded Krylov storage. Never use a residual as FE.

## Future work

cff-czdq owns learning semantics and persisted-state migration. cff-ulbz supplies the separately selected compatible exterior mixture binding; both feed the cff-kul2 composition gate.

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

**2026-09-14T02:44:32Z**

Cooper’s supervised sequence now extracts joint numerical feasibility into cff-a3wz, a prerequisite of this ticket and cff-7sw5. Reuse its local solver, fixed certificates, initialization/globalization evidence, and measured iteration storage. This ticket retains production engine integration, objective ownership, duplicated execution surfaces, lifecycle/readouts/status forwarding, documentation, and the full gate. Split that remaining integration into bounded tickets before assigning it if it cannot fit one worker context.
