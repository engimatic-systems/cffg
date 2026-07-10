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


## Notes

**2026-07-09T21:23:55Z**

cff-7c5o landed the Gaussian half of what this API generalizes: Engine.gaussian_node_stats (NIW/MNIW sufficient stats per node belief, obs-masked) + cffg.conj (NIW/MNIW updates with rollover() returning VMP-correct E-parameters — mean and E[W]=nu*Psi^-1, per D4.1.3). The generic declared-learnables API should unify: discrete counts (agent.py learn_a/_elog), Gaussian NIW/MNIW (conj.py + stats readout), and EF-edge conjugates (in-loop, free — cff-8s6a). The trial-loop pattern to generalize is tests/test_conj.py::test_trial_loop_recovers_lgssm_parameters (compile once, swap params through state, accumulate stats, update, roll).

**2026-07-10T03:31:28Z**

Evaluation finding for this API: member_of covers discrete groups only — consumers needing per-member writes into Gaussian/EF/hybrid/plate-expanded groups currently write state.params[group] arrays by member order by hand. The generic learnables API should extend member_of across all group families and offer a bulk set_params(factor_name -> arrays) helper.
