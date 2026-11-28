---
id: cad-1bat
status: in_progress
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

**2026-11-28T04:49:53Z**

Planning on the sampled-parameter interface in PR #212 at 17c6d51. Learnable or checkpoint feature and Runtime numerical or status invariant are the change paths. Compare the retained QR precision root and a separate common-mean rank-one contribution with raw rational natural parameters. Check 4096 and 2^27, resume from exact new and legacy dense snapshots, a second update, f32/f64 sources, a change of units, and a larger predictor dimension. Inspect direct, ordinary and component MNIW folds, declared key/shape identity, rSLDS reconstruction and joint sampling. Graph message schedules, JIT/vmap and engine/bundle structure are N/A to the host representation study. No production representation change before cost and API review.

**2026-11-28T04:56:46Z**

Independent pre-implementation evidence on 17c6d51. With C=2^27, exact C²+2 rounds to the same float64 word as C², and two different positive contrast energies can produce the same old dense bytes. A public physical .lam ndarray cannot preserve that information. Resolve and convert a valid legacy precision; an old file that already lost rank must fail. In an actual declared 4096 MNIW/NIW NPZ checkpoint, the existing nine-key manifest and public Learnables.load_state_dict(exact=True) round-trip (17,960 bytes). Both prototype representations continue to contrast 6 from the saved 4. A collapsed old precision is rejected.

Against direct Fraction natural parameters, a plain QR root gives 3.99999998510 then 5.99999998510 on the 2^27 test. Carrying the common predictor mean as A + W [o,1][o,1]^T gives exactly 4 and 6 but fails closed when the second mean reverses or moves to zero. Sorting design rows by decreasing maximum absolute entry, applying the same permutation to response residuals, and retaining the canonical QR root gives 4 and 6.0000000000000036 on both equal and reversing means, after NPZ restart. The nonzero correlated-output case has maximum parameter error 2.8e−17 and scale error 4.5e−16 across two folds. Repeating with 20 and 64 predictors also gives 4 then 6.0000000000000036. Finite root checks use column scales and source epsilon, preserving physically distinct units. At 2^27 a float64 root resolves this test even with float32 input statistics; a float32 root fails its rank gate. With float64 root the independent inverse-precision contrast needed by #212 sampling is 1 then 0.6666666666666666. Lowering the current dense rank guard is scientifically unsound.

For three augmented coefficients the dense, root and common-origin state payloads are 112, 112 and 136 bytes. Corresponding plain NPZ sizes are 1,080, 1,090 and 1,598 bytes. Exact legacy readout and f32/f64 support, the precision_quadratic/solve interface, the strict family key/dtype schema, component stacking, rSLDS reconstruction and joint draws need an explicit compatibility choice before changing production representation. Source and numeric JSON: ~/agent-share/cffg-backlog-gates/cad-1bat/{study.py,ordered.py,check.json,ordered_check.json,legacy_check.py,legacy.json,ordered_limits.json,solve.json,scale.json}. Warm setup/update costs are queued under the shared CPU lock.

**2026-11-28T05:11:34Z**

The opt-in SquareRootMNIW posterior and declared sqrt_mniw/component_sqrt_mniw families retain a canonical upper root, including across host sampling, per-component folds and CFFGZ bundle resume. Existing dense families keep their persistent keys and identities. Public .lam rejects rounded dense precision with unresolved quadratic forms. Independent rational oracle plus real JAX x64-off Engine/Learnables: ordinary unnormalized contrast is 4; the two active mixture contrasts are 4.32033014 and 3.67966998, with dormant contrast 2. Source inputs may be float32; host centered folds store float64. Focused validation includes translated 4096 and 2^27, inverse-precision draws, rank/units, legacy conversion, original APIs and new family identity. Production timings and the short suite will use the shared CPU lock.
