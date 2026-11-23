---
id: cfr-yuhl
status: closed
deps: []
links: []
created: 2026-11-23T03:03:35Z
type: bug
priority: 1
assignee: coopwilliams
parent: cff-xa6z
tags: [premerge, numerics, directed]
---
# Certify the actual float32 engine update before returning a directed fixed point

## Background in Plain English — WHY
On the cumulative #195 tip (5c4c43b), #190 can return a converged state even though one ordinary engine step changes its messages more than the allowed tolerance. The directed certificate checks a separate host implementation of the message update. This matters to anyone using the default JAX float32 mode.

## Outcomes / Acceptance Criteria — WHAT
1. A successful directed certificate bounds the residual and free-energy change of the actual public update to the documented tolerance. An unverified candidate follows the existing fallback or exhaustion contract within the original budget.
2. Reproduce on CPU with JAX_ENABLE_X64=0 and PYTHONPATH=tests. Call cstr_planning_fixture.planning_fixture(past=0,horizon=2,affine=True,dtype=np.float32,process_scale=.02), compile, bind observations, run with one iteration and tol=1e-6, and compare engine.step(result.state) against the returned state. At this tip the reported residual is 7.2719e-6. The effective tolerance is 7.6294e-6. The actual maximum relative changes are 2.0463e-5 for g_eta and 1.6257e-5 for g_lam.
3. Retain correct float64 decisions, certified energies, failure origins and budget accounting. Check single precision across scales and against higher precision. Exercise nonfinite inputs and vacuous messages through the existing guards.

## Methodological Suggestions — HOW
Use the runtime numerical or status invariant row in docs/ARCHITECTURE.md. Compare the host directed certificate with the compiled local rules on the same state. Check automatic CPU dispatch, one-iteration failure and the ordinary solver fallback. Investigate a conservative fast check if it can avoid material startup cost. Keep the residual definition fixed.

## Future work
cfp-okq2 records the original directed qualification. The reactor campaign in cpcc-2dc3 used float64. Remeasure the affected timing boundary if the chosen fix changes execution cost.

## Definitions
The residual here is the maximum of abs(new-old)/(1+abs(old)) over message entries. A fixed point has stable messages under their actual update rule.

## Notes

**2026-11-23T03:17:39Z**

The certificate now verifies an actual Engine.step update and both production Bethe ledger values against the independent innovation evidence, in both float32 and float64. The bounded region also reuses this check when it revalidates live data. On the reported float32 fixture the public step exceeds tolerance and the one-iteration run returns uncertified. Focused directed/region tests passed (11). The retained 9 plus 590 held-out float64 decisions passed preliminary replay, including all 59 stock references. The compiled check adds cold startup cost; the integration review records final timings on a clean code head.

**2026-11-23T03:21:28Z**

The final integration review identified an architectural boundary test that rejects array conversions inside Engine. Moved those conversions into the numerical certificate owner. The directed suite and the engine boundary check passed together (8 tests).
