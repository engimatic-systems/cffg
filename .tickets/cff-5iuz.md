---
id: cff-5iuz
status: open
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
