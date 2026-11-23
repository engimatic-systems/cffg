---
id: crbs-2xf6
status: closed
deps: []
links: []
created: 2026-11-23T04:26:14Z
type: task
priority: 2
assignee: coopwilliams
parent: crbs-duxy
tags: [benchmark, rxinfer, extraction]
---
# Qualify native CFFG and RxInfer filters and planners on a common multivariable contract

## Background in Plain English — WHY
Both reactor adapters are specialized to four augmented coordinates, one action and a particular reactor transition. Each rebuilds the starting belief for its planning window. Sharing an external parameter estimator or recycling goal-conditioned beliefs would hide the difference this benchmark should measure.

## Outcomes / Acceptance Criteria — WHAT
1. Implement one goal-free filter per library. It consumes the marginal from the last observation, the action actually applied, and the next outlet reading once. On the adaptive arm its joint vector has ten concentrations and log(Kla), with full covariance including state–parameter correlations. Fit a shared prior and coefficient random-walk scale on development data only. Use each library’s own graph updates. The harness may copy moments between ticks, transport inputs and score outputs. Label that boundary.
2. Implement planning from a copy of that filtered belief. Both planners get the same outlet preferences, action costs, action limits, integration and preview horizon and emit both physical flow commands. Future preferences cannot feed back into the filter. Test by changing only future goals and by replaying an identical observation under changed past proposals but identical applied actions.
3. On an independent linear-Gaussian fixture, recover the dense Kalman filter means, covariance blocks and innovation evidence, and the two-action reference for the matched unconstrained planning problem. Exercise cross-covariances, missing whole readings, partial-state projections and non-diagonal noise. Cover changing applied actions and proper fallback/status. Check a held-out checkpoint or in-memory restart boundary against uninterrupted causal filtering.
4. Check the nonlinear filter against an independent high-accuracy reference on short development windows, reporting reference stability as its numerical budget grows. Report native method and approximation, including precision, timing, any clipping, covariance validity, unsupported cases, and fallback. Include a matched Taylor comparison and keep extra approximation methods separately identified. A short, dense reference need not be run for every tick of a long episode.
5. Measure cold setup, warm filter and plan time, readout and transport, any compilation, peak process memory and deadline accounting on the same CPU. Qualify the numerical validity and dimensionality of both ports before either arm is timed competitively.

## Methodological Suggestions — HOW
Architecture change paths are runtime numerical or status invariant and learnable/checkpoint boundary if a core change proves necessary. Try benchmark-local carry first. The continuous directed CFFG route admits projected linear DATA on a fixed-dimensional Gaussian chain. Qualify actual dispatch for this larger model. Fallbacks, zero information, nonfinite transitions, float32 smoke and the production float64 run should be explicit. The archived RxInfer slice already uses dense Gaussian messages and Julia Linearization, but its shared-history planning adapter restarts m0/P0. Construct a persistent observation-only stream using the installed RxInfer version. Confirm both can read a full joint posterior rather than separate one-dimensional beliefs. Apply matched first-action bounds or identical postprocessing and log raw and applied commands. The existing bounded region has a two-coordinate limit per dense action block.

## Future work
If general rolling state or categorical mode inference becomes necessary, scope a separate library change with cff-3nr4, cff-ees0 or cff-c5qf. The primary experiment proposes continuous tracking of log(Kla).

## Definitions
A filter uses only observations available at the current tick. Planning uses possible future actions and desired outputs. A covariance block measures how uncertain quantities vary together. A Kalman filter provides the exact answer for the linear-Gaussian qualification case. Innovation evidence is the probability density assigned to the next reading before seeing it.


## Notes

**2026-11-23T04:26:57Z**

Compatibility review against merged 51ae6a7: src/cffg/directed_chain_plan.py admits a fixed-dimensional continuous Taylor chain with projected data. Current tools/pcgym_timing/cffg_adapter.py resets from its base, and rxinfer_worker.jl:132 starts each planning solve at m0/P0. The Julia slice already carries dense covariances internally. Both adapters need a new plant model, goals and two-action readout, plus a separate causal filter. Confidence high for the current APIs. Full native nonlinear posterior quality and performance on this plant require the pilot.

**2026-11-23T05:01:17Z**

Qualified both benchmark-native implementations against dense correlated 11-dimensional filters and independent two-flow control references. CFFG uses the same Affine and Natural message rules as its production directed solver; a separate public Engine template agrees. RxInfer uses persistent Julia graph updates through a binary pipe. Both share the nonlinear tangent computation and the observer/planner split. Missing readings, changing goals, two-flow output, restart and likelihood calculations passed. Each arm differed from the dense command oracle by less than 3e-12 flow units. A short four-transition nonlinear importance-sampling reference with 262144 particles passed the declared budget and repeat checks. Both arms were within 0.056 posterior standard deviations in mean and 0.100 in scaled covariance. Qualified PCGym mean and Jacobians across 180 operating points. Raw checks in ~/agent-share/pcgym-extraction-rxinfer/qual/.
