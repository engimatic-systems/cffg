---
id: cff-6wpk
status: closed
deps: []
links: [cff-7t4j, cad-1bat]
created: 2026-11-24T02:37:08Z
type: bug
priority: 1
assignee: coopwilliams
tags: [learning, numerics]
---
# Preserve Gaussian spread through declared NIW and linear-regression learning

## Background in Plain English — WHY

The ordinary Gaussian and linear-Gaussian factors can infer the right variance while the registered learner receives identical evidence for two different variances. The public mixture readout fix cff-7t4j is specific to a prior mixture. The conventional NIW and MNIW families consume different statistics. These families need end-to-end coverage through Learnables.accumulate.

## Outcomes / Acceptance Criteria — WHAT

1. Run the converged public two-factor example with scalar coordinates at 4096, two independent Gaussians of variance 0.25 or 1, and a zero transition slope. Marginal variances agree with each input. With matched float64 learner priors of scale 3 and mean/intercept 4096, preserve the distinct posterior scales 3.25 and 4.0. The original float32 Engine path returns 3.0 in both families for both cases.
2. Preserve within-record input and output covariance and their cross-covariance on a correlated regression fixture. Check Gaussian prior, linear Gaussian, observed/missing linear output and switching linear components separately. Separate directly measured failures from extensions inferred by inspection.
3. Use centered statistics through the family adapters, pooling and posterior solves, including fractional component mass, zero-weight rows, stationary points, representable translations in float32/64, rank-deficient observations and invalid inputs. Test host/compiled, single/batch, supported schedules, and legacy bundle restore with one further update.
4. Compare old and new work, payload sizes and wall time before choosing a default for each family. Keep current public raw-statistic callers working.

## Methodological Suggestions — HOW

Start with the two-factor public fixture recorded in the linked notes. A correct marginal establishes the inference side of the boundary. Verify the learner separately against the analytic posterior. Reuse cff-7t4j for univariate or NIW centered pooling and derive the joint centered regression update before implementing it. Follow the architecture rows for a learnable or checkpoint feature and a registered rule. Confirm persistent semantics and migration only where they change.

## Future work

The AXIOM prior-mixture learner car-9ccf can use its own centered path. Establish the dependency before extending this ticket to an AXIOM pixel-regression implementation.

## Definitions

A posterior scale is the matrix that carries evidence about noise in the Gaussian conjugate posterior. NIW learns a mean and covariance. MNIW learns linear-regression coefficients and noise covariance. A cross-covariance measures how input and output vary together.


## Notes

**2026-11-24T02:39:18Z**

Verified through the public converged Engine and Learnables.accumulate with JAX 0.10.2 on CPU, source 8d46be4 and centered-pooling candidate b763744. Each prior and linear factor reported proper. Both scalar ground-truth marginals correctly distinguish 0.25 and 1. The two raw node records are identical and both posterior scales are 3.0. Independent callback and scalar update corroborate the mechanism. The observed-output and switching-linear callbacks use raw latent moments, with observed coordinates fixed where applicable. The adjacent AXIOM prior-mixture path has its own centered mode. Full scripts and results live under ~/agent-share/axiom-centered-checkpoint/other_families*. This follow-up is related work, not a blanket dependency on unrelated graph families.

**2026-11-24T05:17:01Z**

Implementation handoff on fd8bfd1, fix/centered-declared-learning. Architecture rows: Learnable or checkpoint feature and New factor or rule family (readout only). Check registered prior, latent and DATA linear callbacks, specialized switching-linear callback, shared host reduction and checkpoint continuation. Flood/scheduled inference feed the same readout. Messages, FE, graph encoding, parameter addresses and disk posterior layout N/A. Cover rank-deficient point observations, exact-zero mass, missing/improper rows, both float precisions and large translated coordinates. Wall-time measurements and the short gate await a coordinated slot.

**2026-11-24T05:20:14Z**

Reproduced the two scalar float32 cases on fd8bfd1: correct marginal variances 0.25 and 1, both declared scales 3.0. Ordinary NIW now returns 3.25 and 4.0, also at float64 origin 2^27. Existing raw readout remains available. 68 focused Gaussian learner, readout ownership and registered kernel tests passed (2 opt-in cases excluded).

**2026-11-24T05:25:15Z**

Registered latent and DATA regression now use joint-centered moments and QR. Exact scales at 4096 and at 2^27: 3.25/4.0. Dense, fractional, rank-one and point conditional data agree with an independent regression oracle. 85 focused conjugate, Gaussian-learning and readout-ownership cases passed (one opt-in excluded). New P2 cad-1bat reproduces an unrepresentable column precision with two translated predictors; the learner explicitly refuses that update.

**2026-11-24T05:27:25Z**

Switching-linear centered readout is tested against the independent full Gaussian joint, including its cross-covariance and responsibility. Translated sparse f32 flood and f64 auto public learning cases passed, as did the existing K=32,d=16 write path. Fused centered callbacks are candidates for coordinated timing; raw Engine and direct raw-update APIs retain their original algorithms.

**2026-11-24T05:34:10Z**

Coordinated benchmark on 8f55cfe vs fd8bfd1 (CPU float32, float64 host priors). All five cold and warm public updates improved. Warm medians in milliseconds: scalar NIW 5.88 to 2.08; latent regression d6 6.14 to 2.35; observed 12 x d3 32.21 to 1.82; switching K3/d2 50.17 to 2.51 and K32/d6 56.48 to 8.78. Readout payload summed over the selected sources, in bytes respectively 9/9, 705/481, 2604/780, 482/410 and 32770/30978. Eight paired blocks of three public replacements per shape. Raw timings and reproducible harness: ~/agent-share/cffg-6wpk/measurements.json and bench.py. Historical bundles at both prior dtypes and from both factor categories resumed another update to <6.28e-8 absolute error. On K32/d6, full promotion enlarged 9851 bytes to 13352. Checked retention instead yielded 9327 bytes, original posterior payload 16512 bytes, and exact checkpoint continuation. Across two successive bound-model updates, worst relative error in a covariance-increment direction was 3.804e-5, maximum absolute difference from a continuously float64 host stream 7.95e-8. Explicit rank and tiny update regressions force per-array promotion. The file guard never changes an existing float64 array or count representation.

**2026-11-24T05:41:08Z**

Final coordinated float32-prior timing on f8af30c: K3/d2 warm 48.24 to 2.72 ms and first replacement 3.84 to 0.85 s; K32/d6 warm 54.34 to 11.71 ms and first replacement 4.86 to 1.10 s. Timings cover the public learner operation on an already-converged result. Across both compact-precision runs the posterior payload stayed at baseline size. Saved archive sizes may vary by about one byte with archive metadata. Worst measured direction over two writes 0.003804% of its precision/scale increment; relative bound for retaining a rounded array 0.03453%. Unresolvable gains promote the affected array. Existing float64 priors use the wide path; their representative file was 12924 to 13477 compressed bytes, with no payload growth. Timing, raw paired blocks, exact continuations, sealed artifact digests and reproducer scripts are in ~/agent-share/cffg-6wpk/ (inventory.json).

**2026-11-24T05:41:08Z**

Direct old vs finalist public float32 run at coordinate 4096: NIW and ordinary MNIW both 3.0/3.0 to 3.25/4.0, observed + missing emission 2.9999999404 to 3.1250000056 (oracle 3.1249999981), switching components 4.999999903/3.999999963/3 to 3.679622763/3.480566022/3 (oracle 3.679622759/3.480565862/3). No claims for data that already lost uncertainty upstream or dense precisions that cannot be represented. Quantified rank limit remains open at P2 cad-1bat.

**2026-11-24T05:41:30Z**

Handoff: ordinary NIW, latent/observed MNIW and switching MNIW are implemented and qualified. Final focused test set covered 45 distinct Gaussian/hybrid/readout/fitter cases. The observed-output test originally used a float64 tolerance on float32 covariance and missed by 7.45e-9. It passed at one float32 covariance ULP, then all 20 Gaussian learner cases passed; the 44 sibling cases had passed. Earlier 85 conjugate/learner/ownership cases passed. Parent integrates locally and runs one cumulative short gate. No independent short suite or external publication.
