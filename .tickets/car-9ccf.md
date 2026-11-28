---
id: car-9ccf
status: in_progress
deps: [cff-lquu, cff-6tek, cff-qg1d, cff-7t4j, cff-xfnx, caoc-0dvd]
links: []
created: 2026-11-23T21:48:38Z
type: feature
priority: 2
assignee: coopwilliams
parent: car-zieb
tags: [axiom]
---
# Learn mixed Gaussian and categorical components from soft assignments

## Background in Plain English — WHY

An object may partly match two known types. Its position and an observed category then give fractional evidence to both types. CFFG can already read mixture responsibilities and Gaussian moments. The missing library operation scores mixed observations against each component and uses the same assignment weights for every observed field.

## Outcomes / Acceptance Criteria — WHAT

1. Provide a supported batched expected-log-likelihood and update API for full-covariance Gaussian feature vectors and multiple categorical fields. Expose absolute per-component scores before normalization. Callers choose whether mixing weights enter assignment and which observed fields contribute. Prediction can omit unknown reward and motion class.
2. Register weighted component Normal–inverse-Wishart updates; reuse Dirichlet counts and existing Gaussian node statistics. One atomic update applies fractional counts, first/second moments and all observed category counts with the same responsibilities. Masked fields add zero evidence. Zero mass preserves the previous parameters exactly.
3. Verify asymmetric frequencies with equal component likelihoods. For occupied counts 2 and 5 and one dormant alternative, likelihood-only assignment gives (0.5, 0.5, 0). Including the expected log mixing weight favors the second occupied component. Check absolute likelihood and every learned moment against hand calculations, including missing fields and a batch split into two updates under fixed weights.
4. Preserve exact zero support, reject improper priors/nonfinite inputs and identify an empty active set. Verify host/JIT and single/batched parity in float32 and float64; include a large bank and counts far above a unit fixture. Demonstrate save/restore before and after a fractional update.

## Methodological Suggestions — HOW

Start from cff-lquu weighted means/second moments and the cff-6tek/cff-qg1d registration path. Keep AXIOM’s threshold, component-selection and write policies in the caller. Review the “Learnable or checkpoint feature” and “New factor or rule family” rows, including current/legacy persistence. If a score is registered for inference, test flooding and scheduled owners. An offline update alone has no scheduler.

## Future work

Per-output regression noise for pixels belongs to car-kwjj. Generic joint draws of uncertain parameters are covered by cff-21vm. The release’s specific learning gates are exercised by the AXIOM model tickets.

## Definitions

A responsibility is the fraction of an observation assigned to an alternative. Sufficient statistics are the counts and sums needed for a parameter update. A Normal–inverse-Wishart distribution represents uncertainty in a Gaussian mean and covariance. A Dirichlet distribution represents uncertainty in categorical probabilities.

## Notes

**2026-11-24T01:37:58Z**

Paused at the requested accuracy checkpoint before adding the learner. New prerequisite cff-7t4j documents a public readout information-loss case and the centered-update acceptance check. The scalar absolute-score and fractional-update oracle is recorded on car-8rgl. Resume at the readout boundary after Cooper decides. No game or wall-time comparison was run.

**2026-11-24T01:38:16Z**

Additional large-count score fixture for criterion 4: with Dirichlet concentrations (2^22,1), the exact expected log probability of the first category is -2^-22 by the digamma recurrence. Independent probe of efrules.py returned zero in float32. Absolute error is 2.38e-7. Preserve as a separately quantified stress case; it was not the reason for the stop.

**2026-11-24T03:34:42Z**

The numerical prerequisite cff-7t4j is closed on the fix branch. A public readout plus centered NIW update distinguished the two previously identical variances. Paused before adding mixed-evidence learner registration because validated host pooling was 2.32x the raw time on 32 six-dimensional observations, with a 56-microsecond absolute difference. Recommend measuring a batched, validated learner update in the real data path before choosing a default. No Gameworld episodes run. Independently, cff-6wpk covers conventional Gaussian/regression learner paths and is not a hard dependency here.

**2026-11-24T03:45:30Z**

Cooper accepted the measured centered-update cost. Implementing the mixed Gaussian and categorical update on #200. Change paths: Learnable or checkpoint feature, New factor or rule family (existing factor, new learning family), Runtime numerical or status invariant. Check eager/compiled, single/vmap, fixed responsibilities vs split batches, current/legacy persistence, existing Engine readouts and schedules. The caller-driven mixed update consumes observations outside flooding/scheduled inference. The standalone additive learner and release-specific history policy have separate oracle values.

**2026-11-24T03:53:38Z**

The mixed Gaussian/categorical candidate and the declared component NIW family are implemented. The standalone float32 online bank fails repeated observations at translated coordinates. In the 1024-reading fixture, the analytic scale is 950.2 and the stored scale is 1154.7755, 21.53% too high, despite successful status at every tick. Float64 streaming, zero-origin float32 and one-batch float32 controls pass. At 2^22, four streamed quarter-count assignments add (0,4) instead of (1,3). cff-xfnx blocks qualification. The prior centered moment readout remains accurate. Timing was deferred at the requested accuracy stop.

**2026-11-24T04:26:30Z**

Final numerical source 454c7cb. After reproducing a narrow mask error and truncated snapshot restore, both affected test files passed (23 cases). After a Dirichlet precision discrepancy, the original six scale checks passed unchanged at both precisions; the independent streaming and count probes still reproduce cff-xfnx. The short dashboard suite passed 2,782 tests with 237 opt-in deselected and 4 warnings. Measured wall time 1,222.55 s. Moved only the recurrence-report generation (8.56 s including tiny input guard), reactor comparison report (3.70 s), and extra x64 variants of component-NIW writes (3.98 s), small hybrid reverse derivative (5.67 s) and discrete storage batching (3.49 s) to the opt-in suite. The nonfinite guard remains short and passed separately. Estimated short wall time ~1,197.3 s after selection, actual new time unmeasured. Full result reused under project policy; both precision stress records remain in the experiment archive. Remains in progress behind cff-xfnx.
