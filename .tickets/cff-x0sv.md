---
id: cff-x0sv
status: open
deps: [cff-cd1i]
links: [cff-gjjk]
created: 2026-07-08T22:00:30Z
type: feature
priority: 2
assignee: coopwilliams
---
# Generic conjugate-update API: declared learnables replace learn_a special-casing

GPT-5.5 feedback item 4 (agreed, but sequenced AFTER B-learning so the API generalizes from two concrete clients, not zero — repo culture is concrete-first). Each factor type declares its conjugate family and sufficient-statistic map: categorical/matrix_cat/transition_mixture -> Dirichlet counts from node beliefs; gaussian/gaussian_linear -> NIW stats (that half lands with cff-7c5o, design the contract to fit it). Template marks learnable parameter roles (constraint bead or FactorSpec flag — decide; serialization additive). RollingAgent.learn_a/learn_b collapse into a learnables dict {factor_or_group: counts}; checkpoint/restore generic. Includes d-learning (accumulate q(z0) across episodes) and e-learning (habits: accumulate chosen controls) as trivial instances — GPT item 2 folds in here EXCEPT learned goals c which stays cff-gjjk (hyperprior mu1(c)=Dir(c|Az+1) has its own message derivation). DELEGATION: Fable specifies the contract + which updates are exact conjugacy vs mean-field approximations (document per type); Opus refactors A/B onto it behind existing tests (zero regressions) and adds d/e instances.

## Acceptance Criteria

A- and B-learning re-expressed on the API with zero test regressions; d and e learnable with recovery tests; contract documented in ARCHITECTURE (rule-library section).

