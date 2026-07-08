---
id: cff-pksr
status: open
deps: [cff-4e5z]
links: []
created: 2026-07-08T16:49:34Z
type: feature
priority: 1
assignee: coopwilliams
parent: cff-ahps
---
# Hybrid discrete-continuous: conditional-Gaussian factors + switching KF

The recommended epistemic path (todo-note): discrete regime variables with Gaussian emissions. New factor: gaussian_mixture (out: Gaussian, selector: discrete, params: per-component A/b/cov or mean/cov) — the continuous transition_mixture. Messages: selector->out is a Gaussian mixture that must be COLLAPSED to one Gaussian by moment matching (mean/cov of the mixture) to stay in-family; out->selector is exact (per-component Gaussian likelihood evaluations -> categorical log-message). FE: mixture entropy needs a documented bound/approximation — this is the first place the true-BFE invariant meets a necessary approximation, so the ledger must label the term as bounded, not exact. Test tier: switching Kalman filter vs exhaustive per-regime-sequence enumeration on short horizons. This keeps the GFE composite discrete (regimes) while Gaussians carry perception — exactly what the glucose agent wants (continuous BG, discrete meal/regime states, cff-t2g3). DELEGATION: Fable derives collapse rules + the FE bound and its ledger semantics; Sonnet generates enumeration golden numbers; Opus does arena plumbing mirroring transition_mixture.

## Acceptance Criteria

Switching-KF marginals match short-horizon enumeration; regime posterior calibrated; FE term documented as bound with a test of the bound direction.

