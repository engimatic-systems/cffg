---
id: cao-0pbk
status: closed
deps: []
links: []
created: 2026-11-24T05:44:59Z
type: bug
priority: 1
assignee: coopwilliams
parent: car-mps6
tags: [axiom, parity, numerics]
---
# Apply the released AXIOM Gaussian scale policy to model likelihoods

The released agent changes its Gaussian scale before computing a score. With seven context features, one origin observation and a displacement of 0.03, the mathematically unmodified Gaussian scores -11.558 nats and the released scale scores +15.697 before categorical evidence. This can reverse the -10 learning threshold. Scope the adjustment to the AXIOM experiment and preserve the original learning state.


## Reviewable outcomes

1. Convert a model snapshot to the released inverse-Wishart scale used in expected Gaussian precision and log determinant. Compare both precisions to an independent calculation from raw natural parameters and check the exact threshold, correlated features, displaced observations and a loss of positive definiteness.
2. Exercise the changed match or birth decision on a small deterministic fixture. Keep parameter learning and exact continuation on the original arrays. Connect training, context-only prediction, sampled model reduction and search to the model-facing readout as each becomes available.
3. Record which consumers are qualified before claiming released-agent parity. The general CFFG mixed learner keeps the ordinary NIW definition.

## Method and scope

The architecture path is a runtime numerical invariant. Check eager/compiled, both dtypes and batched components. Message schedules and Bethe ledgers are inapplicable to this experiment-specific readout. The public release b99f625, axiom/vi/conjugate/mvn.py lines 184–194, 247–264 and 588–618, rescales the Gaussian scale by the magnitude of its second natural parameter, bounds its normalized diagonal from below and zeros sufficiently small normalized off-diagonal entries. A view is a derived set of arrays used to evaluate scores, separate from the bank saved for the next update.

## Notes

**2026-11-24T05:57:28Z**

Release b99f625 inverts and scores the normalized clipped scale after a single bank-wide eigenvalue repair, including dormant components (axiom/vi/utils/math.py, inv_and_logdet). If the minimum eigenvalue is negative, it adds twice the difference between dtype epsilon and that minimum to every normalized diagonal. Sampling reads the unshifted inv_u to form expected_sigma first, then applies make_pd separately to each covariance (hybrid_mixture_model.py, sample). Both views retain the raw learning posterior.

release_view(full_bank) now implements scoring repair. release_minima(full_bank) returns the per-row nonpositive minima; a consumer can pass their global minimum to release_view(subset, minimum=minimum) for the same model revision. sampling_view(raw_bank) exposes the projection before the repair. An inactive component changed the score of a second active component on the regression. Before the fix, raw minimum eigenvalue +0.00025772 became -0.00018234 on projection, and the score failed. The release returned finite scores after adding 0.00036468 to both normalized diagonals in float64. Checked eager/JIT float32 and float64, physical unit scaling, unchanged raw state, independent absolute scores and cached subset equivalence. Command uv run --frozen --no-sync python -m pytest -q tests/test_axiom_evidence.py passed 6 tests in 12.80 s. Rollout and reduction owners have the integration contracts; their consumer replays are tracked in their tickets.

**2026-11-24T05:58:22Z**

Scoring implementation and regressions landed at 0e298f9. The rollout owner verified 15 bridge/planner tests, including the probability change induced by a dormant component in both dtypes. The reduction owner has the full-bank minimum, selected-row scoring and sampling contracts for the separate reduction ticket. The parent owns the final scene replay and suite.
