---
id: cff-j4mu
status: open
deps: []
links: [cff-fxmj, cff-5edo, cff-i45j]
created: 2026-07-10T18:04:08Z
type: bug
priority: 0
assignee: coopwilliams
tags: [adversarial-reviewer, convergence, numerics]
---
# Inference runtime cannot distinguish converged, exhausted, diverged, or infeasible states

ADVERSARIAL REVIEWER FINDING (2026-07-10). Engine.run returns only (State, delta_trace), stops on message delta rather than the required FE signal, never returns a termination status, never evaluates FE, and lets non-finite deltas run to the budget and return normally. tests/test_gfe.py::test_run_ignores_nonfinite_deltas explicitly codifies silent exhaustion, despite closed ticket cff-i45j saying divergence should surface loudly. Budget exhaustion is likewise indistinguishable from convergence (the H=25 evaluation already trusted an under-converged policy until a later experiment). Exact-zero contradictions also look numerically successful: an exactly_k=0 factor with both binary DATA ports clamped to 1 stops after 2 sweeps with delta 0, point marginals, and finite BFE 1.0e30 even though Z=0 and true FE=+inf. _gfe_solve additionally returns after 25 Newton steps with no residual/status. Practical impact: callers and agents can act on poisoned, infeasible, or under-converged beliefs; dashboards/tests often infer status from deltas[-1] ad hoc; FE cannot serve as the first-class convergence and policy-scoring signal promised by AGENTS.md.

## Acceptance Criteria

1. Public RunResult reports status in {converged,budget_exhausted,diverged,infeasible}, iterations, message-delta trace, FE trace (where defined), and final residual; compatibility migration is explicit. 2. FE change/stationarity is the primary convergence criterion in sum/GFE mode, with message delta retained as a coherence diagnostic; max mode uses a documented objective/residual criterion. 3. Any non-finite message, delta, FE, or inner-solver residual terminates as divergence and cannot return a normal-looking State without an explicit opt-in. 4. All-zero support/contradictory hard constraints report infeasible with FE=+inf, never a finite NEG-sentinel score. 5. GFE Newton solves expose convergence/budget/divergence and use dtype-scaled tolerances. 6. RollingAgent, Colony, decimation, estimator, tests, and dashboards consume status rather than guessing from trace length. 7. Hostile tests cover sparse support, float32, a large chain, poisoned messages, an unsatisfiable CSP, and deliberate budget exhaustion.
