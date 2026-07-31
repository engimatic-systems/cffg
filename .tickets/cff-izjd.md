---
id: cff-izjd
status: closed
deps: [cff-dfpi, cff-lv2w]
links: []
created: 2026-07-13T02:48:31Z
type: feature
priority: 2
assignee: coopwilliams
parent: cff-ahps
---
# Hybrid GFE composite: engine integration of the D5 regime-epistemics node

## Plain English

The engine will gain the D5 hybrid regime-epistemics composite so discrete
regime uncertainty can drive purposeful exploration in a
discrete-continuous model. It will reuse the proved local stationary solve and
the globalized mixed-GFE outer runtime while keeping Gaussian state messages on
the existing mixture-collapse path. The direct bounded GFE term and its
diagnostics remain first-class ledger outputs.

Implement the P-substituted hybrid mixture composite specified by DERIVATIONS.md D5 (cff-dfpi decision): compiler detection of a gaussian_mixture(_linear) node facing a Gaussian goal with a P_SUBSTITUTION bead; K-logit Newton solve of rho=softmax(r(rho)+mu_e) reusing the D1 globalized solver + status contract; selector message r_s(rho); z-edges keep D3 mixlin collapse messages (v1 scope); exterior substituted x-edge with goal absorbed (interior cavity = collapse(q_w) oslash mu_down deferred); ledger term = direct Ghat (never the Lagrangian value — cff-hp3y). Oracle: cffg/experiments/hybrid_epistemics.py + tests/test_hybrid_epistemics.py are the executable spec; engine values must match them. Include bound-direction diagnostics (sandwich + H(rho) cap) and the hostile-inputs checklist (zero-weight regimes, improper covariances, float32, K up to ~32).

## Design

D5 Prop 4 has the fixed point, messages, and ledger; Prop 3 the surrogate + sandwich; Prop 6 the closed-form NIW/known-V parameter-novelty channel (v2 candidate). Coupled stationarity over (rho, m_s, S_s) is explicitly out of scope until derived (S1h item 4).

cff-weff is a correctness prerequisite rather than a change to D5: this node
adds another locally solved P-substituted coordinate to a mixed plan, so it
must execute under a coupled outer method that certifies the original
stationarity equations instead of the known fixed-damping Picard loop.

## Change-path handoff

Applicable rows: **New factor or rule family** and **Runtime numerical or
status invariant**.

- Authority: `docs/DERIVATIONS.md` D5, `AGENTS.md` numerical contract,
  `src/cffg/compiler.py` composite plans, and `src/cffg/engine.py`
  (`RunStatus`, `RunResult`, `Engine.run`).
- Derived surfaces: P-substitution detection and parameter packing; hybrid
  selector/Gaussian slot ownership; synchronous `_make_step`; scheduled mixed
  propagation; `_make_bfe`; outer-chart support and certification; public
  diagnostics and README/ARCHITECTURE/LIMITATIONS descriptions.
- Duplicated pairs: host/JIT, flooding/mixed, and single-instance/`vmap` are
  affected and must share the composite rule. Current/legacy checkpoint
  decoding, exact two-pass execution, learning readouts, and serialization
  schema changes are N/A because the composite is derived from existing IR.
- Focused tests: `tests/test_hybrid_epistemics.py`,
  `tests/test_hybrid_engine.py`, `tests/test_gfe.py`,
  `tests/test_outer_gfe_formalism.py`, `tests/test_schedule.py`, and
  `tests/test_runtime_status.py`.

## Hostile inputs

- Exact zero selector support remains exact in the K-logit active chart;
  unsupported regimes contribute neither mixture moments nor entropy, and an
  empty intersection is infeasible rather than regularized.
- Vacuous selector input is a valid uniform cavity. Improper Gaussian state,
  noise, or goal precision produces inner failure/divergence and a non-finite
  ledger, never a finite converged result.
- Float32 uses dtype-derived probability, stationarity, positive-definiteness,
  and Newton/globalization guards. No absolute ridge, floor, or clamp is added.
- Scale coverage uses K=32 regimes with nontrivial Gaussian dimensions and
  verifies finite rules, bounded storage, and certified convergence without a
  dense whole-graph Jacobian.

## Acceptance criteria

1. A P-substitution bead on `gaussian_mixture` or
   `gaussian_mixture_linear` with substituted role `out` and a facing Gaussian
   goal compiles to one derived hybrid-GFE composite; unsupported roles,
   missing/ambiguous goals, and interior substituted outputs fail clearly.
2. Engine selector messages, stationary regime beliefs, direct G-hat ledger
   term, and entropy-bound diagnostics match the independent D5 executable
   oracle without changing the existing D3 Gaussian collapse messages.
3. `CONVERGED` retains the independent coupled outer-root, local-solve,
   exact-support, feasibility, and finite stationary-FE certificates. Failure
   reasons remain machine-readable under both host and JIT execution.
4. Sparse selector support, vacuous input, improper covariance/goal, float32,
   and K=32 cases satisfy the hostile-input contract; a `vmap` batch matches
   sequential execution.
5. The probe-versus-idle engine fixture reproduces the research oracle:
   regime epistemics selects the risk-dominated probe, while the otherwise
   identical non-epistemic ablation does not.

## Notes

**2026-07-31T20:47:14Z**

Implemented and validated the D5 exterior hybrid-GFE composite for gaussian_mixture and gaussian_mixture_linear factors facing fixed Gaussian goals. The compiler derives HybridGfePlan groups; the runtime shares the gauge-reduced globalized simplex solve, preserves exact selector support and D3 Gaussian state messages, accounts direct G-hat, and exposes entropy-bound diagnostics. The O(K^2) Hershey-Olsen sandwich runs only on explicit diagnostics, not inference or ordinary scoring. Hostile coverage includes improper state/goal precisions, float32 K=32, forced flooding versus mixed scheduling, and vmap parity. Validation: focused 109 passed / 3 deselected in 103.78s; slow 38 passed / 1261 deselected in 414.37s; final dashboard-backed not-slow suite 1261 passed / 38 deselected in 328.46s. Remaining scope is intentionally separate: interior substituted cavities, learned Gaussian goals, and parameter novelty.
