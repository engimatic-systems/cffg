---
id: crbs-duxy
status: closed
deps: [crbs-jqe2]
links: []
created: 2026-11-23T04:25:21Z
type: epic
priority: 2
assignee: coopwilliams
tags: [benchmark, rxinfer, extraction]
---
# Compare CFFG and RxInfer on a partially observed, changing extraction column

## Background in Plain English — WHY
In the previous 599-decision reactor episodes, CFFG and RxInfer produced practically identical tracking accuracy. Both planners received the same external estimator for one heat-transfer coefficient. A five-stage PCGym extraction column lets us ask a different question. Can a library infer the eight concentrations between two noisy outlet measurements, keep track as the plant changes, and use those beliefs to control the outlet over thousands of decisions?

## Outcomes / Acceptance Criteria — WHAT
1. Qualify both native filters on the same applied flows and outlet readings. Each carries the joint uncertainty about ten concentrations and one positive transfer coefficient. Each refreshes that belief from sensors once per tick. Log the approximation and any external code. Report the matched first-order Gaussian comparison and distinguish any separately qualified alternatives.
2. Freeze a single primary comparison before opening fresh seeds. On common, pre-generated action and sensor histories, compute each seed’s root mean squared error over the eight hidden concentrations and the changed-period ticks (from departure from nominal through the last non-nominal transition of each excursion, with the state scored after the resulting observation). Average the paired CFFG-minus-RxInfer differences across seeds. Report a whole-seed 95% interval, the individual differences, and every failed tick.
3. Measure practical consequences in an independent closed-loop comparison. Score X5 tracking error, effort and saturation of both flows, constraint violations, failure/hold/deadline counts, first-action and warm decision times, and memory over time. Each controller receives its own applied actions and observations. Both get the same control objective, horizon, limits, deadline, hardware budget and fallback. Preserve the objective and both implementations in the archive.
4. Propose 6,000 uninterrupted transitions per long episode, ten times the earlier 599-decision episode. Keep the look-ahead horizon fixed across duration comparisons. Include stationary and changing plants, abrupt change, gradual drift, and a repeated visit to the changed condition. Freeze independent seed-level change times, sensor streams and setpoint schedules. Score per phase, recovery, the first 600 ticks and matched earlier/later visits. Dwell extensions with matched forcing isolate duration if a causal claim about duration is made.
5. Start with three development seeds, then target 20 fresh paired seeds. Use development variability and measured runtime to freeze sample size, meaningful effect and interval precision before opening held-out seeds. If an interval spans meaningful advantages for either method, report the comparison as inconclusive. Keep seeds intact during resampling.
6. Archive the frozen protocol, sensor-only and truth records, source and environment digests, workers, fallback events and per-seed results. A complete report can distinguish practical accuracy, uncertainty calibration, recovery, and computational cost.

## Methodological Suggestions — HOW
Compare adaptive results with two controls per library. A frozen-coefficient filter holds the nominal value. An informed filter gets the true current coefficient while still having to infer the eight hidden concentrations. Use these to quantify which part of the task parameter adaptation can improve. Establish an identical Taylor approximation as a parity reference. If a library has another supported approximation, qualify it on development data under the same predeclared resource and tuning budgets and publish its result separately. Fixed tape isolates inference, and closed loop tests whether the resulting actions help. Predeclare a causal failure policy that keeps every scheduled tick in the denominator, with coverage and failure counts reported alongside errors.

## Future work
A discrete hidden operating-mode posterior and explicit memory of previous regimes would require a separate experiment. A repeat visit in this study measures readaptation. Finite-hypothesis product support is tracked in cff-ees0, partial-goal hybrid planning in cff-c5qf, and reusable persistent rolling boundaries in cff-3nr4.

## Definitions
X5 is the liquid outlet from stage five. Y1 is the gas outlet from stage one. Kla is the rate at which material transfers between phases. A joint belief includes the uncertainties of individual quantities and their correlations. An episode is a continuous simulated plant run. A planning horizon is the short future interval considered at each decision. A paired seed gives both libraries the same underlying experimental conditions.


## Notes

**2026-11-23T04:27:10Z**

Prerequisite --> dependent. The two development roots can proceed in parallel.

[ ] crbs-4d73  Plant, hidden truth and sensor design (ready) ----+
                                                                +--> [ ] crbs-jqe2  Pilot and freeze --> [ ] crbs-duxy  Fresh-seed comparison
[ ] crbs-2xf6  Native filter and planner qualification (ready) --+

At the proposed scale the full closed-loop matrix spans 1,440,000 decisions (20 seeds x 6,000 transitions x 2 conditions x 3 arms x 2 libraries). Common-tape filtering adds another 1,440,000 filter updates over 40 distinct tapes if all arms use the full design. Forecast/reference audits can be restricted to prespecified windows. The pilot sets sample size and forecasts wall time from measured rates. No held-out runs are part of scoping.

**2026-11-23T06:01:55Z**

Completed all 480 fresh-seed runs, with 6,000 uninterrupted transitions per run and 20 independently paired seeds. Each library estimated eight hidden concentrations from two outlet sensors. On the common recordings during changes, both adaptive filters averaged 0.0020072 hidden-state error. The mean paired difference was 2.99e-17, well inside the prespecified practical margin of 0.0002. Compared with fixing the rate at its nominal value, learning reduced hidden-state error by 81.4% and closed-loop outlet-tracking error by 24.3% during changes. On the stationary plant, fixing the correct rate won both comparisons. Adapting increased hidden-state error by 23.9% and tracking error by 1.1%. Both filters regained the changed rate in a median eight decisions on both abrupt visits. The measured routes use the same local Gaussian approximation. CFFG used its directed Gaussian operations, and RxInfer used its cached Julia graphs. CFFG reached the first real action faster in one cold launch per library (2.33 vs 11.28 s). RxInfer was faster once warm (1.01 vs 1.15 ms median). All 1,440,000 closed-loop commands met the deadline with zero fallback commands.

The HTML report, frozen protocol, 480 raw archives and checksums are in ~/agent-share/pcgym-extraction-rxinfer. The independent audit replayed 2,880,000 sensor records, checked 2,640 sampled state transitions and dense filter outputs, and reproduced the primary effect to rounding. To reproduce scoring, run uv run --frozen --no-sync python tools/extraction/analysis.py --output ~/agent-share/pcgym-extraction-rxinfer --stage heldout, then use audit.py with the same arguments and report.py with the output directory as its positional argument. Isolated benchmark code and tickets are the only changed paths. Focused runner_check.py, analysis_check.py, report_check.py, native filter and planner references, nonlinear reference convergence, and full-archive verification cover this comparison. The default core-library suite was outside this change. Discrete operating-mode memory remains a separate research question.
