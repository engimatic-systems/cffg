---
id: cff-f3jz
status: open
deps: [cff-ocxl, cff-7sw5, cff-ffsg]
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

D12 local kernels and compiled ownership are insufficient without a graph solve and a complete objective ledger. The NumPy experiment solves held auxiliaries only; production must solve the joint message/xi residual and report its actual terminal state.

## Outcomes / Acceptance Criteria — WHAT

1. Generic engine execution consumes the declared plan without an rSLDS model branch, selecting one approximation and one belief-array authority.
2. The solver handles D12.7–8 jointly with gauge removal, dtype-scaled properness checks, a feasible initialization strategy, bounded local/global/linear/line-search budgets, and globalization. It distinguishes converged, budget-exhausted, and diverged; non-finite objective/residual is divergence.
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


