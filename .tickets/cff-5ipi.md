---
id: cff-5ipi
status: closed
deps: [cff-b47g]
links: []
created: 2026-07-28T04:19:49Z
type: task
priority: 3
assignee: coopwilliams
tags: [experiment, preregistration, gfe]
---
# Preregister the factored-corridor annealed-novelty experiment

## Plain English

This ticket writes down the complete test of fading transition curiosity
before the test is run. It fixes the model, sensor conditions, metrics,
timing method, thresholds, and interpretation of each possible outcome so
the evidence can be judged against a stable protocol.

Write the replacement protocol for cff-2gev only after cff-b47g's factored builder and two-block corridor fixture exist and before any replacement experiment is run. The previous flat Hallway comparison remains useful negative evidence but cannot choose the new protocol: its state aliasing, independent flat transition columns and count-mass scale are exactly what the replacement changes. This ticket fixes the environment, observability interventions, weight ownership, metrics and decision thresholds without implementing or executing the experiment.

## Design

State exactly which declared transition learnable each novelty weight follows and how tied-instance statistics become per-parameter count mass. Define reachable/active column masks for cost drop-back; never use a global maximum over structurally unreachable or irrelevant columns. Fix prior mass, n_half rationale, observability levels, arms, seeds, episode budget, evaluation windows, known-policy reference occupancy, task-success metric, model-error metric, timing method and outcome language. Separate mechanism identities and unit tests from empirical claims. Record how non-convergence, divergent FE or insufficient effective data invalidate a run. The protocol is appended to cff-2gev before execution; changes after a run begins void it visibly.

## Acceptance Criteria

Before any replacement result exists, cff-2gev contains one clearly marked live protocol naming exact builder configuration and learned blocks; lambda/count equations and drop-back mask; prior and tuning rationale; observability sweep; arms, seeds, episodes and windows; independent KL and task-return metrics; timing comparison; convergence/status requirements; and pass/fail/ambiguous outcome language. Every choice is fixed or explicitly derived from builder structure. The ticket makes no implementation change and records no experiment result.

## Notes

**2026-07-29T18:45:11Z**

Installed the LIVE, unrun cff-5ipi/v1 protocol in cff-2gev. It fixes the 8x2 factored corridor, position_transition-only uncertainty, count/weight equations, goal-policy active mask, observability interventions, paired arms/seeds/windows, independent task/KL metrics, timing isolation, hostile-input checks, invalidation rules, and locked outcome labels before any replacement result. No implementation or experiment was run. Next: cff-2gev implements this protocol without changing its locked choices. Validation: git diff --check and tk dep cycle pass; full pytest is not required for ticket-only work.
