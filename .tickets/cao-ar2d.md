---
id: cao-ar2d
status: closed
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

**2026-11-28T04:28:30Z**

Plan. Reproduce the stored two-word donor in both precisions, then compare the selected transaction with independently aggregated rational natural moments. Check the Runtime numerical or status invariant and Learnable or checkpoint feature change paths: host exact, eager/JIT/vmap, device result, retry and scientific scoring are affected. Graph schedules, templates, migration, hyperparameter banks and evaluation seeds are N/A. Keep the current compensated representation, invalid/stale status, exact rollback and scientific score comparison. Quantify residual refusal and retry cost separately with synthetic records.

**2026-11-28T04:37:06Z**

Qualification on a774a54. The prior and learned donor centers can share both stored words. In that case the device now multiplies the common moments by the donor excess count; the existing error test still compares against four dtype epsilons times the smaller of the represented result and change. For the translated center 32 + float(0.3), an eps/512 fractional count moved from NUMERICAL_RANGE on a774a54 to OK at both precisions. The independent Fraction oracle sums raw natural coordinates, including partially observed categorical heads. All accepted fields satisfy the change-relative bound. Eager/JIT/vmap, exact rollback, stale references, donor reset, unchanged neighbors and the 32/500/5000-slot checks are covered.

The 2/3 and 5/3 rounded donor still returns NUMERICAL_RANGE in pure compiled code. Rational receiver changes (mean, scale) are (−8.881784197001252e−16, −1.184237892933501e−15) in float32 and (−3.081487911019577e−33, −4.108650548026103e−33) in float64. The public host-boundary propose_merge_exact reads three selected rows, rechecks references and validity, and uses the host lifecycle rational operation. The candidate agrees bitwise with the full exact lifecycle. A host-only scientific reducer already uses that exact operation. The score remains mean held-out reward/switch log likelihood minus mixing KL; accept equal or larger. Across two synthetic three-request sequences, five requests were live, three needed retry, one was inactive. Score changes +0.02648051389 and −0.97351948611 determine the contested decisions, and a zero change is accepted. All six outcomes matched the shipped reducer.

Ran uv run --frozen --no-sync pytest -q -m "slow or not slow" -k "not 5000_rows_and_2000 and not full_bank_rescore_5000" tests/test_component_bank.py tests/test_component_bank_device.py tests/test_axiom_reduction.py tests/test_axiom_reduction_bounds.py: 58 passed, 2 large scientific rescoring studies deselected in 65.13 s. No evaluation data used. The shared CPU queue has a reproducible warmed synthetic comparison in ~/agent-share/cffg-backlog-gates/cao-ar2d/warm_cost.py. Parent owns the integrated short-suite gate and closure.

**2026-11-28T04:37:16Z**

With JAX_ENABLE_X64=0, jax.disable_jit and jax.jit returned identical status and every result field for both the shared-center transfer and the 2/3 refusal. Both selected candidates passed the rational oracle.

**2026-11-28T06:48:22Z**

The canonical integrated gate at fd9a0f7 passed all 2,844 selected tests in 455.037 s on four workers. The four code and test hashes in warm_cost.json match this reviewed branch. In 17 synchronized warmed host retries, the median selected-row vs full-snapshot costs were 3.347 vs 10.343 ms (4 slots, float32), 3.345 vs 10.298 ms (4 slots, float64), and 3.828 vs 11.459 ms (500 slots, float32). At 500 slots the corrected same-center device attempt took 219.17 µs vs 202.46 µs previously and now accepts the tiny transfer. The 2/3 range refusal took 211.71 vs 208.00 µs. Each device median uses 129 calls. Ordinary control timings varied by block. These data quantify the cost of the safeguarded attempt and the selected exact retry on these examples. Reproduction harness and raw blocks: ~/agent-share/cffg-backlog-gates/cao-ar2d/{warm_cost.py,warm_cost.json}, receipt SHA-256 1ea5f56f7f97c4faec82a783fec317c65d86c43e29f6ed8e6a53be3190e23745.
