---
id: cff-xfnx
status: in_progress
deps: []
links: []
created: 2026-11-24T03:52:06Z
type: bug
priority: 1
assignee: coopwilliams
parent: car-zieb
tags: [axiom, learning, numerics]
---
# Retain small mean and count updates across an online mixed-learning stream

## Background in Plain English — WHY

A single batch can correctly combine 1,024 readings even when updating the same model one reading at a time silently loses the accumulated change. The within-component covariance is available and correct. The new failure occurs later when a small change is written into a large absolute float32 posterior mean or count. This threatens the continual-learning contract.

## Outcomes / Acceptance Criteria — WHAT

1. Reproduce the public mixed_streaming_probe with prior mean 10000, kappa 4096, scale 3, and 1,024 observations of mean 10001 and variance 0.125. Require equal evidence across one batch, singly streamed observations and restart. Compare the final mean against 10000.2 and the learned scale against 950.2. The candidate returns valid at every tick but the float32 stream ends at 10000 and scale 1154.7755. Origin zero and float64 are working controls.
2. Cover soft counts independently. For prior count 2^22, four updates of (0.25,0.75) should add (1,3). The current float32 split path adds (0,4). Test active and inactive components, missing observations, and evolution of all shared fields.
3. Choose and document a persistent representation, including what callers with JAX x64 disabled can use. Qualify full streams in float32/float64, single/batched/JIT execution, checkpoint before and after a sub-ULP increment, and loading the original snapshot. If a configuration cannot preserve the promised accuracy, expose that status distinctly from a successfully accumulated update.
4. Compare the accuracy, synchronized time and persistent bytes of viable representations on a fixed observation sequence. Name what changes for the 10,000-step AXIOM experiment and what remains an adversarial generic-library case. Report any new tradeoff before adopting a default.

## Methodological Suggestions — HOW

The first diagnostic separates inference, within-observation covariance and persistent posterior storage by supplying the same fully specified observation and responsibilities every time. Use the analytic weighted NIW sum as an oracle. Compare a wider persistent accumulator with a saved local origin and residuals for means, and compensated accumulation for counts. Keep posterior storage precision independent of scoring precision where supported. A change to the stored fields follows the Learnable or checkpoint feature architecture path, including legacy restore and single/batch equivalence. Flooding and scheduled execution are N/A for this standalone online update.

## Future work

car-9ccf can qualify its native streaming API after this accuracy boundary is resolved. Component lifecycle and the Gameworld agent remain downstream. cff-6wpk covers a different loss at the conventional Gaussian inference-to-learner boundary.

## Definitions

An ULP is the gap between adjacent representable numbers at a given magnitude. A persistent accumulator stores evidence across real observations. A posterior scale carries information about the variance the learner is estimating.


## Notes

**2026-11-24T03:55:16Z**

Independent review supplied a two-record regression in either dtype: let m=4096 and h=spacing(m), with kappa=1 and psi=h*h. Two point observations at m+h give 5*h*h/3 in a batch but 13*h*h/6 one at a time, also after restart. Storing a wider mean moves the limit. Compare keeping the fractional offset across checkpoints as well as a wider accumulator; quantify the supported range.

**2026-11-24T05:16:09Z**

Cooper chose compact snapshots by default, exact continuation opt-in. Round only at reload by saving the primary fields from a live compensated bank; ordinary saves leave learning untouched. Preserve exact count and mean carry when requested, reload old layouts with zero carry, reject damaged exact layouts. Architecture paths: Learnable or checkpoint feature and Runtime numerical or status invariant. Check public score/update and paused/restarted streams in eager/JIT/scan/vmap and both precisions; solver sweeps, graph templates and CFFGZ declared-family layout are outside this standalone operation. Benchmark recurring score/update speed and compact file cost against fd8bfd1. A required double-digit performance or file tradeoff needs a human decision; optional exact bytes already authorized.

**2026-11-24T05:25:00Z**

A second accumulated array also needs a remainder. With float32 Gaussian scale Psi=2^24, sixteen observations contributing 0.25 each must add 4. Uncompensated scalar updates add zero. Exact mode therefore includes the full scale matrix as well as the mean and counts; compact stays in v2. Development tests cover all three mechanisms and exercise optional exact restore at every tick. An overstrict precision guard in the draft rejected some normal soft assignments including Gameworld patch tick 37 and 28/40 seeded scan steps at N=1, K=5000. Repairing and independently replaying before timing.

**2026-11-24T05:57:10Z**

Independent checks on 089f377 preserved all three precision corrections, 480 recurrent updates and archival compatibility. The real-frame RGB patch corpus saw zero changed MAP labels across 23,040 assignments, and exact file reload preserved output bits. A compact restart on every translated scalar observation still loses 21.53%; once per 128 loses 0.029%. First default compute candidate and improved candidate both exceed Cooper’s threshold. Experimental result and the open default decision live in caoc-0dvd and http://localhost:8080/axiom-online-precision/index.html.

**2026-11-24T07:18:58Z**

The final integrated short suite passed 2798 tests in 1163.637 seconds wall time at f74543f. The corrected compensated update and opt-in exact continuation are implemented and qualified for both working precisions. The remaining blocker for the default is Cooper's decision in caoc-0dvd (optimized recurring float32 time increases 21.7–54.2%). The disk cost of the compact format for the same visible arrays is unchanged. The full review artifact is ~/agent-share/axiom-online-precision/index.html.
