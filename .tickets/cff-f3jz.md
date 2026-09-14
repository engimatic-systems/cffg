---
id: cff-f3jz
status: open
deps: [cff-ocxl, cff-7sw5, cff-ffsg, cff-a3wz, cff-brcf, cff-idhk]
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

D12 local kernels and compiled ownership are insufficient without a graph solve and a complete objective ledger. cff-a3wz first qualifies the complete message/xi solver without production binding. cff-idhk then supplies accepted-coordinate storage and transport, and cff-brcf supplies objective/readout ownership. This final integration ticket activates their public execution and reports the actual terminal state through the engine.

## Outcomes / Acceptance Criteria — WHAT

1. Generic engine execution consumes the declared plan without an rSLDS model branch, selecting one approximation and one belief-array authority.
2. Production reuses the solver qualified by cff-a3wz for D12.7–8 jointly, preserving gauge removal, dtype-scaled properness checks, feasible initialization, bounded local/global/linear/line-search budgets, and globalization. Engine integration distinguishes converged, budget-exhausted, and diverged; non-finite objective/residual is divergence. It does not introduce a separate numerical policy or weaken the prerequisite's certificates.
3. Public execution uses the cff-brcf ledger with precisely D12 regional energies, labeled entropies, hybrid separator corrections, adapter entropies, and ordinary exterior corrections. It matches the independent tiny objective and stationary dual at the returned coordinate, without repeating ledger construction or introducing another owner.
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

**2026-09-14T02:51:53Z**

Research question → after cff-a3wz, can production integration fit one worker context while retaining one solver and one belief-array authority? Answer → likely not. Suggested sequential split before implementation: (A) accepted component coordinate, lifecycle and transport; (B) owned objective and internal readouts; (C, retain this ticket) public scheduling/status activation. Each slice keeps its applicable host/JIT, synchronous/scheduled, and single/vmap pairs together. A consumes cff-7sw5 plans and the unchanged a3wz solver; ordinary boundary messages are derived views, solver scratch is temporary, and no public D12 execution activates before B/C. B excludes component-owned categorical/Gaussian entropy from ordinary ledgers and reads one accepted coordinate without solving again. C excludes D12 from exact two-pass eligibility and forwards complete terminal certificates through Engine and Colony. Source → independent integration_boundary_research report; src/cffg/runtime_types.py:29, lifecycle.py:45, execution/contracts.py:8, schedule_contracts.py:10, execution/transport.py, estimate.py:76, ledger.py:113, ledger_terms.py:30, readouts/recurrence.py:19, readouts/gaussian.py, readouts/views.py, execution/synchronous.py, schedule.py, schedule_program.py, iteration.py, run_results.py, colony.py. Confidence → high on existing ownership and duplicate surfaces; medium that three contexts suffice, pending actual solver and compiler-plan interfaces. This is a future split recommendation, not a change to current dependency records or a review of in-progress FFSG.

**2026-09-14T02:51:53Z**

Two concrete integration hazards from that research: runtime_checks.state_is_finite currently checks every leaf while D12 uses literal negative-infinity support; define a lawful representation/check and mirror it in schedule_program.make_support_check. D9 recurrence readouts reconstruct from ordinary inputs; D12 must explicitly select accepted component moments rather than rerun D9. Current/legacy learning and checkpoint identity remain cff-czdq across Learnables.replace, learning_checkpoint.py, learning_schema.py, core/learning_identity.py, and bundle.py. GFE/Mu3 mathematics remains outside D12, but shared state forwarding still needs review.

**2026-09-14T03:06:47Z**

Bounded split adopted before implementation from the independent ownership research. cff-idhk owns accepted-coordinate state, lifecycle and mechanical transport; cff-brcf owns objective exclusions and internal readouts; cff-f3jz retains public scheduling/status/consumer activation and final integration parity. These tickets add no parallel implementation authorization and all remain blocked behind cff-a3wz. Exact component types/fields are chosen only from the qualified solver and compiler plan when their implementer starts.

Legend: prerequisite --> dependent. Based on tk dep tree --full cff-x0if, tk ready, and tk blocked. Multiple inputs require all prerequisites; tk remains authoritative.

[x] cff-ocxl
  +--> [~] cff-ffsg (review repairs; only active implementation)
  |       --> [ ] cff-a3wz (joint numerical gate; next)
  +--> [ ] cff-k6g5 (ready declarations; independent research only in parallel)
         a3wz + k6g5 --> [ ] cff-7sw5 (compiler plans)
                             --> [ ] cff-idhk (state/transport)
                                   --> [ ] cff-brcf (objective/readouts)
                                         --> [ ] cff-f3jz (public execution)
                                               +--> [ ] cff-czdq (learning)
                                               +--> [ ] cff-ulbz (mixtures)
                                      czdq + ulbz --> [ ] cff-kul2 (composition)
                                                        --> [ ] cff-x0if (recovery)
                                                              --> [ ] cff-zov5 (epic)

Validation: no dependency cycles; documentation/ticket-only change with no runtime surface modified. Full-suite validation is not required for this split; git diff --check covers changed artifacts.
