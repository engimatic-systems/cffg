---
id: cff-6tek
status: open
deps: [cff-x0sv]
links: [cff-l6i5]
created: 2026-07-27T20:52:13Z
type: feature
priority: 2
assignee: coopwilliams
---
# Learnables across non-discrete families: member_of beyond Cat/Dir, NIW/MNIW registration

The declared-learnables API (cff-x0sv) resolves write targets through Arena.member_of, which covers the discrete families only — the Gaussian, EF, hybrid and plate-expanded group builders all skip it deliberately. A learnable over a gaussian or gaussian_linear factor therefore raises a compile error naming that limitation instead of learning.

Two pieces, in order:

1. Extend member_of across the remaining group families, so every factor with parameters has an addressable per-member slot. Consumers today write state.params[group] arrays by member order by hand (evaluation finding, 2026-07-10); a bulk set_params(factor_name -> arrays) helper falls out of the same map.

2. Register NIW and MNIW as learnable families. The machinery exists: cff-7c5o landed Engine.gaussian_node_stats and cffg.conj's NIW/MNIW updates with rollover() returning VMP-correct E-parameters. What is missing is the adapter — the Family protocol assumes node_beliefs as the statistic source, and the Gaussian half reads gaussian_node_stats and owns several blobs at different shapes. LearnPlan already carries per-blob shapes for exactly this.

The trial-loop pattern to bring onto the API is tests/test_conj.py::test_trial_loop_recovers_lgssm_parameters, which currently accumulates by hand.

## Acceptance Criteria

A gaussian_linear parameter learns through Template.learnables with no hand-rolled accumulation, recovering the LGSSM parameters the existing conj trial loop recovers. member_of addresses every parameterized factor family, and set_params writes members by name.

