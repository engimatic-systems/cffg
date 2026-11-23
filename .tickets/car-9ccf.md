---
id: car-9ccf
status: open
deps: [cff-lquu, cff-6tek, cff-qg1d]
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
