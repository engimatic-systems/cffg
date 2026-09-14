---
id: cff-j191
status: closed
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

**2026-09-14T08:48:29Z**

Implementation plan and prospective numerical specification: tests/fixtures/hybrid-joint-reference-spec.json freezes copied tiny inputs, both xi starts, zero messages plus one local quotient initialization sweep, fixed zero and nonzero origins, active-label gauges, 200 iterations, 24 backtracking trials, fixed-point direction followed only on failure by dense finite-difference Newton, and every final certificate. The same-coordinate arithmetic and moderate formulas from cff-a3wz are unchanged. Stopping error is accounted for separately by the frozen local residual-sensitivity propagation formula; no discrepancy determines a tolerance. No reference checkpoint solve has run before this specification commit.

Applicable architecture paths: Derived component coordinate and Runtime numerical or status invariant. Affected surfaces: independent NumPy/SciPy reference evaluation/iteration/certification, focused tests, dashboard registration, and canonical D12 anchors. Checked N/A pairs: production host/JIT and single/vmap, synchronous/scheduled runtime, compiler/IR/serialization, lifecycle, persistence, runtime ledger ownership, public learning activation, and registered recovery fits. Float32 and moderate K/d/T qualification remain in cff-y15d/cff-a3wz.

Hostile-input plan: exact -inf support masks remain masks, absent message coordinates are omitted, inactive xi and learning are exactly zero; empty support and invalid active objectives fail explicitly. Signed/vacuous cavities are allowed if the complete active Gaussian blocks and returned aggregate beliefs are proper. Improper trials are rejected without precision clamping. NaN/+inf, malformed inputs, negative xi, and tiny-positive mass underflow cannot report success. Masked-source and empty-source companions are distinct from the D11 destination-masked twins. Fixed observed drive uses conditioned categorical entropy, never a singular Gaussian EC solve. Existing temporal falsifier and fixed-auxiliary approximation ceilings stay separate and unchanged.

**2026-09-14T08:58:58Z**

First checkpoint evidence: all 18 valid frozen case/start records converged in 0–8 updates, without rejected steps or Newton fallback; worst returned certificate 5.274e-15. The four frozen empty-support/underflow records explicitly diverged. Full first-attempt records and original spec hash are retained in tests/fixtures/hybrid-joint-reference-attempt-1.json.

First focused command (uv run pytest -q tests/test_hybrid_joint_reference.py) reported 44 passed and one tuple/dictionary test-wiring failure in 1.23s. An extra diagnostic test solved coupled_t2 with site_constant[0,1]=-600 before that literal input was registered. That result is explicitly excluded from frozen qualification evidence. Its exact modification, command context, observed assertions and missing per-solve detail are preserved in hybrid-joint-reference-diagnostics.json; the separate 22 checkpoint records retain their original provenance. The revised test will only evaluate a held frozen solution with that diagnostic modification. Newly chosen held-coordinate diagnostics are committed prospectively before their next execution. Solver strategy, numerical tolerances, original fixtures, D12 semantics and recovery requirements are unchanged.

**2026-09-14T09:04:34Z**

Implementation complete and paused for independent review; ticket remains in_progress. The independent files are tests/hybrid_joint_reference.py (local/chain evaluation and conditioned fixed-drive algebra) and tests/hybrid_joint_solver_reference.py (fixed-support Coordinate, bounded solve, comparison arrays, local stopping propagation, JSON reporter). load_case reads the copied committed inputs. evaluate(model, messages, xi) recomputes all natural blocks, regional and separator physical/centered moments, both FE expressions, xi targets, residual certificates and source-conditioned learning without running a solve. Coordinate.pack/unpack supplies the gauge-reduced symmetric unknown vector; solve returns explicit converged/exhausted/diverged status and failure reasons. output_arrays and stopping_allowance are the candidate-comparison handoff.

Final focused validation: uv run pytest -q tests/test_hybrid_joint_reference.py tests/test_hybrid_separator_derivation.py passed all 82 tests after the final relevant source edit. This includes 48 new tests and 34 unchanged D12 derivation tests. The new file is registered in dashboard EPIC_FILES. Canonical ARCHITECTURE/D12 anchors explain the API and remaining qualification boundary. tk dep cycle and git diff --check pass. No full short suite was launched; the supervisor owns one final run after independent review.

Reproduction: uv run python tests/hybrid_joint_solver_reference.py --output /tmp/cff-j191-final.json. tests/fixtures/hybrid-joint-reference-results.json retains the final machine report, original spec SHA256 and exact independent-source hashes. Its 22 frozen case/start records have 18 converged and four expected empty-support/positive-mass-underflow divergences. Maximum accepted updates: 8; rejected trials: 0; Newton steps: 0; maximum returned certificate: 5.274e-15. The independent local inverse-coordinate norm is at most 2.413, and maximum scaled output sensitivity is 9. Same-coordinate arithmetic and the prospective local stopping-propagation formula remain unchanged. The report separately includes the conditioned fixed-drive arrays, temporal falsifier values and unchanged fixed-auxiliary ceilings.

Evidence limits: ordinary backtracking/Newton fallback is implemented but was not needed by these frozen solves; no measured fallback-performance claim is made. The first unregistered -600 diagnostic solve remains excluded, with its exact input and evidence limits in hybrid-joint-reference-diagnostics.json. Subsequent tiny-active, signed-cavity, inactive-source and positive-auxiliary-underflow checks only evaluate held coordinates and passed. No new held diagnostic exposed a numerical failure. Float32/JIT/vmap complete iteration, moderate-horizon storage, candidate comparison and registered recovery fits remain downstream; no such solve ran here. Next: independent review, supervisor final short-suite gate, then cff-y15d can consume this reference. No thresholds, D12 approximation, or supported production scope changed.

**2026-09-14T09:10:21Z**

Independent review complete at 336a2bf with no actionable findings. Reviewer reran the focused 82 tests (all passed in 1.46s), checked imports and D12 equations, and independently integrated local densities with 96-point Gaussian quadrature at held perturbed T2, coupled T3 and centered T3 coordinates. Moments, auxiliary targets, learning and both objectives agreed within 2.08e-15 scaled error, below unchanged arithmetic budgets. Conditioned K3 fixed-drive logistic products agreed within 2.23e-16; Gaussian entropy is correctly excluded. The original spec is byte-identical to 2ee434c and source hashes match; all 18 successful saved coordinates recertify, alongside four expected divergences, max 8 updates and worst certificate 5.274e-15. An independent larger finite-difference stencil corroborated local stopping sensitivities; no global bound or uniqueness claim is made.

The disclosed extra -600 diagnostic remains excluded and distinguished from all 22 pre-specified records. Frozen valid cases did not need rejection/Newton fallback; branches were reviewed structurally without performance claims. Reviewer made no edits and finished all processes. Completed review was sent to the implementer, who remains paused. Supervisor now runs the single final short suite; candidate, float32/moderate iteration, storage and registered recovery qualification remain downstream.

**2026-09-14T10:18:45Z**

Final short-suite gate PASSED at 75ac1d2 (reference implementation 336a2bf): uv run python tools/dashboard.py --run-suite; 2537 passed, 41 deselected, 4 existing gym deprecation warnings in 3176.36s, exit 0. Log /tmp/cff-j191-short-suite.log; dashboard ~/agent-share/cffg-tests.html and tracked runtime history refreshed. Supervisor waited the full recorded 4020-second budget quietly before checking completion; no unchanged suite rerun. Independent review and the 82-pass focused result are recorded above. No reference numerical source changed during review or final gate. Closing J191: the independent reference, frozen specification, exact held-coordinate checks, local stopping sensitivity API and transparent provenance are complete. cff-y15d is now the next ready implementation; its candidate and cff-a3wz moderate/storage qualification remain unrun. Registered recovery remains untouched.
