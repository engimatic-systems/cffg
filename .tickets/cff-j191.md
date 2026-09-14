---
id: cff-j191
status: open
deps: [cff-ffsg]
links: []
created: 2026-09-14T04:40:24Z
type: task
priority: 2
assignee: coopwilliams
parent: cff-a3wz
tags: [rslds, inference, hybrid, numerical]
---
# Build an independent tiny joint hybrid solver reference

## Background in Plain English — WHY

The early numerical checkpoint cff-a3wz must compare a complete message and auxiliary solve with independent mathematics. The existing NumPy experiment fixes the auxiliary values, while exact temporal path enumeration represents a different approximation. An independent tiny reference is therefore its own bounded implementation ticket before the candidate solver is written.

## Outcomes / Acceptance Criteria — WHAT

1. An independent float64 reference solves the complete T=2 and coupled T=3 expectation-consistency (EC) message equations together with the squared-bound auxiliary equations. It imports neither production mathematics nor the candidate solver. Exact path integration is used only where the mathematical reduction justifies it; T=3 uses independent EC closure.
2. A machine-readable specification is committed before reference checkpoint solves. It freezes tiny fixtures selected from the cff-a3wz cases, initializations, reference budgets/globalization, reporting fields, and arithmetic versus propagated stopping-error accounting. The fixed moderate formulas and numeric tolerances in cff-a3wz are carried forward unchanged. Candidate solver budgets are frozen separately before candidate execution in the next ticket.
3. Reference solutions meet tau(float64)=64*eps(float64)**0.75 at one returned coordinate for natural, unweighted conditional mean/covariance, auxiliary, stable finite free-energy, and primal/dual certificates. Proper active beliefs and explicit converged/exhausted/diverged outcomes are checked. Inactive auxiliaries are exactly zero; a nonzero stationary residual is not success.
4. Executable cases cover both D11 source-mean twins and variance assignments, D9 fixed-drive/drive-independent reductions, boundary messages, conditional cross-moments, full free energy, and source-conditioned learning increments. The existing D12 temporal falsifier and fixed-auxiliary approximation ceilings are reported separately and unchanged.
5. Same-coordinate reference evaluators expose sufficient arrays to compare all message blocks, centered and physical moments, objective, auxiliary targets, and learning statistics independently of the reference solve. Arithmetic uses cff-a3wz's fixed atol=rtol=512*4*eps and FE absolute tolerance 512*max(4,5*T)*eps; any solved-coordinate stopping allowance has a prospective justification rather than being fit to discrepancies.
6. Before implementation, record Derived component coordinate and Runtime numerical or status invariant change paths. Reference execution and test/dashboard surfaces apply; production JIT/vmap iteration, compiler, schedules, lifecycle, persistence, public learning activation, and recovery fits are N/A. Plan literal zeros, empty support, inactive auxiliaries, improper complete beliefs, invalid input, and tiny-positive underflow explicitly. Float32 and moderate scaling remain the candidate/qualification responsibilities; state this boundary instead of claiming reference coverage for them.
7. Focused reference tests, independent review, canonical derivation anchors, dashboard registration, and the required final short-suite gate pass after final relevant changes. Keep each new source file below the repository size limit. Record reproducible commands/results and hand off the frozen specification and reference API. Closing this ticket does not qualify the candidate or activate production inference.

## Methodological Suggestions — HOW

- Reuse only independent calculations from tests/hybrid_separator_reference.py and the mathematical D12/D11/D9 equations. The fixed-auxiliary experiment can explain conventions but cannot become the supposedly independent implementation.
- Keep test-only independent reference mathematics separate from the candidate package. Tiny dense reference solves are permitted; this does not authorize dense trajectory or Jacobian storage in the candidate.
- Split before implementation if the bounded reference itself exceeds one worker context.

## Future work

The next bounded ticket implements the unbound candidate solver against this reference. cff-a3wz remains the unchanged moderate-horizon numerical and storage qualification gate. Compiler integration and cff-x0if registered recovery remain blocked downstream.

## Definitions

- D12: the hybrid-regime-ec-v1 mathematical approximation.
- EC: expectation consistency, matching specified moments of neighboring regional beliefs.
- Auxiliary xi: the squared logistic-bound coordinate, tied per source label and stick.
- Stick: a binary decision in the ordered categorical link.
- Natural coefficients: constant, linear, and precision coefficients of a quadratic log potential.
- Primal/dual: two expressions for the same stationary free-energy objective.
- Epsilon: dtype machine resolution near one.
- Oracle/reference: independently implemented mathematics used to check a candidate.
- JIT/vmap: JAX compiled and batched execution.



## Notes

**2026-09-14T04:41:18Z**

Downstream handoff: cff-y15d is the separately implemented candidate solver. cff-a3wz remains the qualification gate; none of its numerical, fixture, or recovery requirements are removed by this split.

**2026-09-14T06:51:59Z**

Independent planning research (read-only; no solves): Which existing fixtures/APIs provide an independent joint reference? Answer -> tests/hybrid_separator_reference.py imports only NumPy/SciPy and supplies enumerate_chain (scalar K2, T2..6 fixed-auxiliary integration), recurrence_increments, and the separate reflected_mixture_gap falsifier. Reuse these independent APIs where justified, but copy fixture VALUES into the frozen specification: importing tests/test_hybrid_separator_derivation.py also imports candidate mathematics. Existing oracle c means sqrt(xi); freeze that conversion. Coupled T3 requires new independent local EC projection/quotient, D12.4 primal and D12.9 dual evaluations; its exact temporal path -logZ is not the EC objective. Analytical Gaussian identities are the arithmetic authority: existing quadrature tolerance2e-11 is larger than the fixed float64 same-coordinate budget (~4.55e-13), so quadrature is corroboration only with declared error.

Prospective fixture-to-claim map: (1) T2 coupled fixture(2) values for complete messages+xi, exact held-xi path moments/cross-covariance/FE/learning; (2) T3 coupled fixture(3) values for independent EC closure and all returned arrays; (3,4) mean twins (2,-2)/(-2,2), unit variances, zero slopes/offsets, destination log support[0,-inf], vacuous destination Gaussian site; (5,6) variance assignments(1,9)/(9,1), zero means, same support, pin conditional second moments and precision increments; (7) T3 independent=True values for separated categorical/continuous D9 reduction, xi=offset^2, exact path FE and nonzero temporal cross covariance; (8) K3,d2 fixed-drive values in test_regime_pair_derivation.py:118 for conditioned categorical algebra with xi=(Wx+offset)^2 and augmented learning; (9) source-masked T2 baseline for zero exposure/inactive xi/zero learning; (10) empty source-support companion for explicit failure. Twins mask a destination and cannot substitute for the inactive-source case.

The fixed-drive check CONDITIONS OUT the observed drive before Gaussian entropy evaluation. It is an algebraic D9 reduction, not a singular zero-covariance continuous EC solve; supported D12 scope is unchanged. Reference validation must distinguish intentional -inf support from NaN/+inf, and expose active conditional moments independently of weighted statistics: existing conditional_moments and path skipping are not sufficient for these hostile-input checks. All fixture numbers, gauges/origins, initializations, stopping propagation and bounded solve strategies remain to be frozen by the implementer BEFORE executing checkpoint solves.

Source -> independent b0s3 research report; tests/hybrid_separator_reference.py:14, tests/test_hybrid_separator_derivation.py:23, tests/regime_trajectory_reference.py:65, tests/test_regime_pair_derivation.py:118, docs/DERIVATIONS.md D11.1 and D12.2-.4/.9. Confidence -> high for formulas, independence boundaries and coverage. This note is implementation planning, not reference/candidate qualification evidence. No reference solve, moderate checkpoint or registered recovery fit has run.
