---
id: cad-1bat
status: open
deps: []
links: [cff-6wpk]
created: 2026-11-24T05:24:57Z
type: bug
priority: 2
assignee: coopwilliams
tags: [learning, numerics]
---
# Retain regression precision when translated predictors obscure small directions

## Background in Plain English — WHY

The MNIW regression learner stores a dense matrix describing confidence in the fitted coefficients. Two large input coordinates can nearly agree. Squaring those coordinates can round away the evidence about their difference, even when the residual noise in their joint prediction is accurately fitted. The centered learner rejects an unresolved stored matrix.

## Outcomes / Acceptance Criteria — WHAT

1. With identity prior column precision, identity joint covariance, output mean zero and two predictor means equal to 2^27, retain precision 4 in the unnormalized contrast (1,-1,0) across save, resume and another update.
2. Keep the corresponding translated 4096 case correct and distinguish ill-conditioned coordinates from ordinary differences in physical units.
3. Specify recovery for old checkpoints and measure setup, update and storage costs before changing the public representation.

## Methodological Suggestions — HOW

Consider keeping a factored precision or expressing regression coordinates relative to a retained input origin. Compare against a high-precision centered design. The regression in test_centered_mniw_rejects_rank_loss_in_stored_precision reproduces the current boundary.

## Future work

Online mixed-feature accumulation has a separate representation decision under cff-xfnx.

## Definitions

MNIW stands for matrix-normal inverse-Wishart. Column precision describes confidence in the slope and intercept coefficients. A contrast is a weighted difference of coefficients.


## Notes

**2026-11-24T05:25:20Z**

The fail-closed regression is in tests/test_conj.py. The baseline contrast 4 is recovered exactly at offset 4096; the stored-precision guard refuses the unresolved result at 2^27.
