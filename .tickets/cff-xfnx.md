---
id: cff-xfnx
status: open
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

