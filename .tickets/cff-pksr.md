---
id: cff-pksr
status: closed
deps: [cff-4e5z]
links: [cff-o7hs, cff-lquu, cff-k9ws]
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


## Notes

**2026-07-09T21:12:36Z**

CLOSED (branch hybrid, chained on ef-engine PR #9). Delegation ran as planned: Fable derived D3 (DERIVATIONS.md) — exact selector message via per-regime joint log-partitions; direction-dependent collapse weights (forward = selector probs; backward needs rebuilt c~_s*Z_s scales, the 'weight trap'); node energy CLOSED FORM via entropy chain rule, so the ledger's only approximation is Gaussian edge entropy of collapsed beliefs => reported BFE provably lower-bounds mixture-Bethe (S1d, bound direction pinned). Sonnet built the numpy reference (exact enumeration + independent D3 smoother; weight-trap demo: D3 [0.0025,0.9975] vs naive [0.5,0.5], >5x posterior error; proved rank(Lam_b)=rank(Lam_o) boundary rank-deficiency for dy<d — guard's vacuous fallback is load-bearing). Opus plumbed MixGroupPlan cross-arena groups. Acceptance met: engine==independent-reference to 1e-6/1e-7; enumeration gaps MEASURED (1D: regime .027/mean .18; 2D: .068/.16) and asserted with 2x headroom; regime posterior exact at single nodes; FE documented as bound with direction test. v1 envelope: no observed mixture outs (emissions via gaussian_linear), full-rank A_s. Suite: 1031 green. gaussian_prec/precision learning: cff-7c5o (next).
