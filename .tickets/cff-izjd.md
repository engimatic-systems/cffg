---
id: cff-izjd
status: open
deps: [cff-dfpi]
links: []
created: 2026-07-13T02:48:31Z
type: feature
priority: 2
assignee: coopwilliams
parent: cff-ahps
---
# Hybrid GFE composite: engine integration of the D5 regime-epistemics node

Implement the P-substituted hybrid mixture composite specified by DERIVATIONS.md D5 (cff-dfpi decision): compiler detection of a gaussian_mixture(_linear) node facing a Gaussian goal with a P_SUBSTITUTION bead; K-logit Newton solve of rho=softmax(r(rho)+mu_e) reusing the D1 globalized solver + status contract; selector message r_s(rho); z-edges keep D3 mixlin collapse messages (v1 scope); exterior substituted x-edge with goal absorbed (interior cavity = collapse(q_w) oslash mu_down deferred); ledger term = direct Ghat (never the Lagrangian value — cff-hp3y). Oracle: cffg/experiments/hybrid_epistemics.py + tests/test_hybrid_epistemics.py are the executable spec; engine values must match them. Include bound-direction diagnostics (sandwich + H(rho) cap) and the hostile-inputs checklist (zero-weight regimes, improper covariances, float32, K up to ~32).

## Design

D5 Prop 4 has the fixed point, messages, and ledger; Prop 3 the surrogate + sandwich; Prop 6 the closed-form NIW/known-V parameter-novelty channel (v2 candidate). Coupled stationarity over (rho, m_s, S_s) is explicitly out of scope until derived (S1h item 4).

