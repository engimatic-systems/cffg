---
id: crbs-jqe2
status: closed
deps: [crbs-4d73, crbs-2xf6]
links: []
created: 2026-11-23T04:26:41Z
type: task
priority: 2
assignee: coopwilliams
parent: crbs-duxy
tags: [benchmark, rxinfer, extraction]
---
# Pilot repeated plant changes and freeze the long-episode RxInfer comparison

## Background in Plain English — WHY
More ticks are valuable when they reveal drift, slow loss of calibration, or recovery after a familiar condition returns. A long stable plateau can make an average look good while hiding an estimator that fails every time the plant changes.

## Outcomes / Acceptance Criteria — WHAT
1. Use three development seeds, disjoint from evaluation. Give both libraries the same goal-free probing tapes in both stationary and changing conditions. Test the nominal frozen, informed and native adaptive filters. Use short affine and nonlinear references from the qualification work. Train or select methods only on these development cases with the same resource allowance.
2. Run a short feasibility pilot of approximately 600 transitions per seed with a complete change and return, with dwell times justified in model hours. Include at least one uninterrupted 6,000-transition development run per library to expose numerical drift, memory growth, file size and realistic wall time. Freeze the proposed evaluation at 20 fresh paired seeds, or record a revised seed count based on feasibility and expected interval width before opening any held-out outcomes.
3. On the same tape, score the primary hidden-state error, one-step outlet predictive log density before seeing the reading, joint and marginal hidden-state calibration, Kla error, and per-change recovery time. Preserve censored recovery and failed outputs. Test a full covariance against the same mean and diagonal-only covariance on the same truth. Report counts and failures for every method. A stronger score from the full covariance measures the value of correlations on this fixture.
4. Separately run both controllers in closed loop with each arm’s own causal history. Compare X5 tracking, flow effort, physical reliability and compute. Record matched seeds, disturbances, initial truth and sensor noise. Use an advance-declared feasible hold on failure/deadline and score its consequences. Freeze a causal substitute for unavailable filter means (for example, last finite prediction) and a rule for undefined densities so failures cannot improve the headline by dropping difficult periods.
5. Freeze the sensor contract, physical sample time, preview length, action treatment, regime schedule and stationary twin, signal scales, practical difference, per-seed score and 95% interval procedure, arm selection, deadline, wall-time estimate and hashes. Use a one-way development/held-out boundary. Stop before the full campaign if the informed reference cannot usefully separate the task from frozen dynamics, sensor recovery is implausible, either library fails its affine reference, or the expensive arm cannot complete at the frozen operating point. Report whichever criterion failed.

## Methodological Suggestions — HOW
Start with two noisy outlets. Use recurring A→B→A changes with a later drift excursion, independently varied setpoints, and random initial profiles that admit the same observation but different interiors. Preserve fixed physical-time event windows and compare the first and later visits at matched perturbation size. Publish whole-run and phase-specific errors, 90% interval coverage and width, joint Gaussian log score for finite proper beliefs, recovery in ticks and model hours, cold and warm latency, maximum memory and holds. A 6,000-transition run with 20 seeds costs 120,000 transitions per library per arm per condition. Measure and price the full matrix before calling it feasible. Preserve native diagnostics even when a command is held.

## Future work
A green pilot unlocks the fresh-seed campaign under the parent epic. Discrete switching beliefs and nonlinear continual learning of multiple unknown parameters are separate questions.

## Definitions
A development seed can inform design decisions. A held-out seed is first used after those decisions are frozen. A stationary twin experiences the same input and noise design while its transfer coefficient stays nominal. Censored recovery means the filter never recovered within the observation window. Calibration checks whether reported uncertainty matches observed errors.


## Notes

**2026-11-23T04:27:10Z**

If native methods match at the same approximation, preserve that result. The pilot may qualify a second method per library using the same resource budget. Predeclare the primary competitive configuration for each library before fresh seeds. For long-run repeatability, separate a second abrupt visit to the same B plateau from a gradual excursion. Compare first and second abrupt visits with matched amplitude, dwell and setpoint phase; report the ramp separately.

**2026-11-23T05:06:13Z**

Three development seeds have completed, along with uninterrupted 6000-transition adaptive runs in both libraries. The first development run identified no numerical failures. Independent review then corrected the failure-at-reset policy, interrupted worker receipts and lock inheritance; the original pilot evidence is preserved under attempts/pre-freeze-runner-review, and all development cases have been replayed with the reviewed runner. The next decision uses only the repeated pilot and full qualification attestations. The fresh seeds are still unopened.

**2026-11-23T05:07:17Z**

Final pilot passed all gates, including native affine parity, independent nonlinear reference, identical pre-change adaptive histories, headroom, reliability and 6000-step endurance. The 72 short runs contributed 43200 updates and two 6000-transition runs measured retention of state and memory. The short twin histories and all 43200 sensor records passed independent replay. The signed-off sources and outputs are bound by file digests under ~/agent-share/pcgym-extraction-rxinfer/qual/attestations.json and freeze.json. Frozen design chooses all 20 reserved independent seeds, 6000 transitions each, with a practical primary-error margin of 0.0002 concentration units. Matched Taylor filters are compared on tapes and in separate closed loops, with the frozen and informed controls included.
