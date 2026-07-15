---
id: cff-lquu
status: open
deps: [cff-3hgi]
links: [cff-o7hs, cff-pksr, cff-k9ws]
created: 2026-07-10T03:31:09Z
type: feature
priority: 2
assignee: coopwilliams
parent: cff-ahps
---
# Hybrid node readouts: soft sufficient statistics + posterior predictive

gaussian_mixture(_linear) nodes are skipped by Engine.gaussian_node_stats and have no predictive readout, which blocks two consumer patterns measured in library-scale evaluation: (1) exact soft EM for switching models — regime-weighted cross-moments E[x_t x_{t-1}' | s=k] are not exposed, forcing hard-assignment EM with a second conditioned gaussian_linear pass per window (soft responsibilities discarded; recovery still works but is statistically inefficient); (2) one-step posterior predictives (collapse of per-regime forward predictions) must be reassembled in consumer code from endpoint marginals + raw params. Add: per-regime joint stats from the mixture node belief (the rho_s-weighted joint Gaussians D3 already constructs for energies) and a mixture_predictive readout.

## Acceptance Criteria

gaussian_node_stats covers hybrid groups (per-regime weighted Syy/Syx~/Sx~x~ + responsibilities); soft-EM parameter recovery test beats hard-EM on a low-separation fixture; predictive readout matches quadrature on one node.

