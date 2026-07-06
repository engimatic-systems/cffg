---
id: cff-5iuz
status: closed
deps: [cff-007q]
links: []
created: 2026-07-05T23:03:40Z
type: task
priority: 1
assignee: coopwilliams
parent: cff-mwjo
---
# Importance-sampled A-message (learned observation matrix)

Part II Fig.5 msg 3: log mu(A)=z^T xi(A), non-conjugate; importance sampling with explicit PRNG keys, Dirichlet param edges, burn-in averaging of stochastic GFE. Requires Dirichlet VMP edges (param_edges) in compiler.


## Notes

**2026-07-06T00:07:56Z**

Scope sharpened by cff-tng6 diagnosis (see its note): (1) extend GfePlan with learned-A mode taking Dirichlet counts as params; (2) messages use Abar for probabilities + closed-form E[A log A] for pseudo-entropy h-tilde (try closed form BEFORE IS — may make IS unnecessary for T-maze); (3) mu3(A) log-message / IS with PRNG keys for the composite->A direction; (4) port win-rate ramp + GFE>BFE assertions from cff-tng6; target Part II Fig.9 shape.

**2026-07-06T00:26:48Z**

DONE via closed form — IS not needed for the T-maze. Landed: GfePlan.learned mode (P-bead detail parameter_uncertainty=dirichlet), composite consumes Dirichlet counts as params; messages use Abar=E[A] for probabilities + closed-form E[A log A] pseudo-entropy h-tilde (Jensen gap = parameter-novelty channel; unit test: unexplored column preferred over known-50/50). High-count limit provably recovers fixed-A composite (marginals 1e-6, BFE 1e-5). RESULT: Part II Fig.9 reproduced — cue from trial 0, win rate ramps to ~alpha=0.9 (5/6 seeds; bimodal stuck minority faithful to paper), BFE ablation 0/30. Second key fix: MAP action selection replaced posterior sampling (soft control posteriors + sampling = random walk; AIF exploration lives in the objective). mu3/IS split to cff-iwgl.
