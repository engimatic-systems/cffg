---
id: cff-d8jj
status: closed
deps: [cff-u9un, cff-aqpz]
links: []
created: 2026-09-25T09:24:47Z
type: bug
priority: 1
assignee: coopwilliams
parent: cff-xa6z
tags: [pcgym, numerics, float32]
---
# Avoid cancellation in Gaussian-mixture selector evidence

## Background in Plain English

The retained three-step float32 hybrid model converges with a fresh undamped message residual of zero, but its first selector posterior differs from an independent reference by 6.04e-5. The current joint Gaussian log masses are about 9849; subtracting their large common constants loses precision. The resulting log-odds error is 3.13e-4. A centered scalar-likelihood calculation on the same incoming beliefs reduces the posterior error to 3.36e-9. Joint matrix condition numbers are modest, so this is cancellation rather than a near-singular model.

Numerical convergence certifies the implemented update, not the accuracy of these subtractive scores. cff-u9un records the distinction instead of silently relaxing a production status threshold.

## Outcomes / Acceptance Criteria

1. Supported Gaussian-mixture selector updates use a numerically stable equivalent evidence calculation when proper incoming Gaussian information permits it.
2. The retained float32 case agrees with an independent centered likelihood oracle within a preregistered dtype-relative bound without changing parameters, objective, selector support or covariance floors.
3. Both plain and linear mixture selector paths are assessed; any unaffected path is explicitly justified. Improper but supported incoming messages retain truthful existing fallback semantics.
4. Sparse and exact-zero support, vacuous inputs, float32/float64, strongly displaced means, and at least tenfold dimensions or components have explicit outcomes.
5. Proper-branch JIT, vmap and forward/reverse derivatives remain valid. Invalid computed intermediates cannot become finite evidence through a safe inverse placeholder.
6. Existing posterior, node-energy and learning-statistic authorities remain consistent with the documented D3 approximation. Clarify which selector-independent normalizers cancel.
7. Tighten the retained planning posterior check after repair; do not claim that numerical accuracy repairs the separate partial-goal planning limitation in cff-c5qf.

## Methodological Suggestions

Another sound method is acceptable. Before implementation use the New factor/rule family and Runtime numerical/status architecture rows and list all execution pairs. For proper input, consider integrating the predictive Gaussian against the output information potential after centering at its predictive mean, dropping only selector-independent normalizers. Keep a mathematically justified fallback for cases outside that envelope. Compare stable local scores, not only a converged engine result. Full final short suite required.

## Future work

- cff-wt1c uses the accuracy repair for matched float32 comparisons.
- cff-c5qf separately decides the partial-goal planning contract; this ticket does not revive deferred expectation-consistency work.

## Definitions

A selector is the categorical variable choosing a Gaussian component. Log odds compare two selector log probabilities. Cancellation is loss of meaningful digits when subtracting large nearly equal values. A condition number measures sensitivity of a linear system. JIT is compiled execution; vmap batches instances; dtype is numeric precision.


## Notes

**2026-09-25T09:26:33Z**

Measured research, high confidence: H3 float32 auto converges in 4 sweeps; fresh undamped message residual is exactly zero. Max reference discrepancies: selector6.03704558e-5, mean5.81475583e-7, covariance1.00973462e-9, FE2.05200246e-4. Canonical dyn1 masses9849.69335938 and9848.24902344 have float32 spacing0.0009765625; stable same-cavity likelihood log odds differ by3.12619906e-4 and explain essentially all posterior error. Stable same-cavity posterior differs from independent oracle by3.36e-9. Joint condition numbers7.31,38.16,51.89 rule out near-singular input as the dominant cause. Sources: read-only pcgym_numerics_review probe, mixg._log_z/joint selector assembly, tests/hybrid_planning_fixture.py. Current integration test separates strict moment/FE/residual checks from a fixture-specific selector allowance: two canonical-score log-odds ULPs times sigmoid derivative bound1/4, plus ordinary rounding. ULP means the gap between adjacent representable numbers. This is not a universal accuracy theorem; tighten/remove the cancellation allowance in this repair.

**2026-09-25T09:26:33Z**

Dependency snapshot from tk dep tree --full, tk ready and tk blocked; no cycles. Legend: prerequisite --> dependent.

[x] arithmetic repairs --> [in progress] cff-u9un
  +--> [blocked] cff-d8jj stable selector evidence --+
  +--> [blocked] cff-c5qf scope decision -----------+--> cff-wt1c --> cff-xa6z

The two follow-ups can proceed independently after the integration baseline. cff-c5qf additionally awaits Cooper decision. Parallel ready roots include cff-7kpx, cff-qu2j, cff-ees0 and cff-nd1q.

**2026-09-25T09:57:57Z**

Pre-implementation handoff: New factor/rule family and Runtime numerical/status rows. Assess both linear and prior-mixture selector updates plus their duplicated responsibility calculations in node energies and public learning statistics; shared scoring must preserve their mathematical agreement. Host/JIT, flood/scheduled, Engine/Colony, single/vmap and forward/reverse derivatives apply. Checkpoint/serialization schema N/A. Preserve mathematically valid improper-input fallback, exact-zero support and existing D3 approximation. Drop only selector-independent normalizers; no covariance floors or tolerance relaxation. Hostile cases planned: displaced means and tight incoming precision, sparse selector support, vacuous/improper cavities, f32/f64, K>=32/d>=20, invalid computed covariance and gradients. Derive a centered Gaussian overlap/information-potential calculation before patching; stage stable score ownership separately from integration if necessary.

**2026-09-25T10:02:13Z**

Research owner map (pcgym_numerics_review): six relative-score consumers are mixlin_to_selector/mixlin_stats/mixlin_energy and mix_to_selector/mix_stats/mix_energy. Only their responsibility scores may drop component-common constants; node entropies and expected log factors remain absolute. Forward/predictive weights stay incoming selector probabilities. Backward collapse log_c+log_Z is a distinct absolute-mass reconstruction and is not replaced piecemeal. D5 hybrid GFE is different mathematics and N/A. Neural differentiation indirectly consumes update/ledger. Source: mixg.py, execution/cross_domain.py, ledger_terms.py, readouts/gaussian.py; high confidence read-only inventory.

Derivation review (failure_provenance): for predictive N(m,C), C=L L.T, common finite anchor a, r=eo-Lo*a, delta=m-a, G=I+L.T Lo L, solve u=G^-1(L^-1 delta+L.T r), t=G^-1 L.T(r-Lo delta), p=L u. Relative log evidence r.T p-.5 p.T Lo p-.5 t.T t-.5 logdet G is exact up to potential(a), common to all components. Compute u and t with two RHS to avoid cancellation in either posterior location or movement. Choose stopped-gradient output pseudomode clipped to component mean bounds: an unbounded mode can be far from a weakly observed predictive cloud and erase component differences. Gauge invariance preserves normalized-weight derivatives. Proper C/G and all computed intermediates are required. For improper input Li, retain integrable-joint fallback: scalar Li=0,Lo=W=1,A=1vs2 has evidence ratio2:1 via Jacobian1/abs(A). Source: algebra and bounded local counterexample; high confidence. Sanitize unused branches before decompositions/products, not only after.

**2026-09-25T10:09:56Z**

Implementation note: stationarity further reduces the prior research formula to score=.5*r.T*p+.5*r.T*delta-.5*delta.T*Lo*p-.5*logdet(G), avoiding subtraction of large movement quadratics and requiring one solve. All six relative-score consumers share it; absolute backward masses and Gaussian entropy/factor terms remain unchanged. The original improper-input integrable-joint branch is preserved and sanitized when unused. Independent overlap tests include sparse/empty support, singular potentials with linear nullspace terms, integrable negative precision, displaced means, K32/d20, JIT/vmap and derivatives. Retained H3 float32 now converges in four sweeps with selector max error5.049e-7, log-odds max error1.962e-6 and FE error5.261e-5. Removed canonical-cancellation allowance; log-odds bound is64*eps. The new guarantee is bounded numerical accuracy, not a partial-goal planning repair.

**2026-09-25T10:10:56Z**

Independent review (failure_provenance): singular/indefinite integrable potentials and improper-input Jacobian branch agree with direct integration, with finite reverse derivatives. An extreme-scale case C=Lo=1e20 in float32 (1e200 in float64), means0 and sqrt(C), overflows G despite finite proper inputs. This implementation deliberately reports invalid evidence and public NaN messages rather than publishing safe-solve placeholders. Tests pin that explicit outcome and LIMITATIONS documents the envelope; arbitrary-scale solvability is not claimed. Question: do accepted scores preserve the integral? Answer: yes on reviewed bounded cases; extreme whitening overflow is rejected. Source: direct scalar overlap and current helper; high confidence.

**2026-09-25T10:11:40Z**

Final independent review: no accepted-output discrepancy found. Moderate singular and integrable-indefinite potentials matched independent information-form integrals; eta/precision/covariance derivatives agreed within about1e-9 float64 and1e-6 float32. Both linear-input branches had finite matching forward/reverse derivatives. Source: failure_provenance bounded read-only probes. Persisted local direct-integral derivative tests pass at1e-12 float64.

**2026-09-25T10:13:00Z**

Focused validation after integration: test_mixg plus original planning checks24passed; tightened evidence/planning45passed2deselected; additional cavity/covariance derivative checks3passed; hybrid_readouts, hybrid_learnables, mixture_collapse, mixture_rank, neural_gfe and switching_kf94passed207.47s. Exact selector support, learning weights, D3 ledger, improper-input fallback and independent derivatives passed. Full final short-suite gate remains before PR publication.

**2026-09-25T10:41:30Z**

Final gate after the last numerical edit passed:2595 tests,103 deselected,4 third-party Gym deprecation warnings; wall1272.832s. cff-aqpz moved15 unchanged expensive replays to slow using saved timings, estimating1190.517s; gate reused without numerical changes. Dependency and whitespace checks pass. Completed stable evidence across six consumers, independent overlap/derivative/hostile-input regressions, strict retained float32 accuracy checks and canonical docs. Remaining partial-goal scope is cff-c5qf, not part of this numerical repair.
