---
id: cao-ar2d
status: open
deps: [cao-24tn]
links: []
created: 2026-11-24T05:42:38Z
type: task
priority: 2
assignee: coopwilliams
parent: car-k6oi
tags: [axiom, numerics]
---
# Reduce ordinary-scale cancellation refusals in compiled component merges

## Background in Plain English — WHY

A compiled component merge can refuse an ordinary, valid proposal when subtracting nearly equal learned and prior statistics. The exact host operation can still compute the result. A model that treats the device refusal as evidence against a merge would change its scientific decision rule.

## Outcomes / Acceptance Criteria — WHAT

1. Reproduce the refusal using a receiver centered at zero and a donor with prior mean 1, mean weight 2 and scalar NIW scale 1. Adding one point at zero gives the donor mean 2/3 and scale 5/3. When each value is stored as two floating-point words, an exact host merge succeeds while the current compiled kernel returns NUMERICAL_RANGE with every original field intact.
2. Establish a supported device accuracy bound for cancellation. Accept the fixture or document the residual numerical limit with the exact host result, on both float32 and float64. A small change in the prior-subtracted moments may not silently disappear into a successful result.
3. Verify single and batched proposals against the independent rational natural-parameter oracle, with translated coordinates, unequal counts, partial categories, and an unchanged neighbor bank. Keep explicit rejection and exact rollback for examples outside the supported bound.
4. Compare decision outcomes on a short interaction-rule merge sequence. When the scientific model requested the proposal, the exact selected-row host retry on NUMERICAL_RANGE supplies the candidate for its normal acceptance score. Record how often the fallback was needed and its cost separately from purely compiled merges.

## Methodological Suggestions — HOW

Consider a different local coordinate system, rearrangement of the prior subtraction or a tighter error bound. The Runtime numerical or status invariant and Learnable or checkpoint feature change paths cover the fixed-shape transaction and exact host replay. Check eager/JIT/vmap; graph sweeps, template fields and bundle migration are N/A. The executable starting case is test_unresolved_prior_subtraction_uses_explicit_range_status in tests/test_component_bank_device.py.

## Future work

car-k6oi owns complete lifecycle qualification and car-mps6 owns the model's proposal and acceptance rule. The present fallback lets those consumers advance while this numerical limit is investigated.

## Definitions

NIW means Normal–inverse-Wishart. Its scale summarizes Gaussian spread. A high/low pair stores a primary floating-point value and a smaller correction. Cancellation occurs when subtracting nearly equal numbers. NUMERICAL_RANGE means that the device cannot establish the required accuracy for the proposed change.


## Notes

**2026-11-24T05:44:15Z**

The failing device proposal and exact-host retry are both executable in test_component_bank_device.py. Both precisions passed the explicit status, unchanged source and identical retried candidate checks at d1b1953.
