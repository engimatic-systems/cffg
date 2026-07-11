---
id: cff-j4mu
status: closed
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

## Notes

**2026-07-11T15:45:55Z**

Planning hostile-input contract (2026-07-11):

- Sparse/exact-zero support: retain log-zero semantics; any edge/factor belief with empty support classifies infeasible and its FE is +inf.
- Improper/vacuous incoming: transient FE is recorded only once the ledger is defined; a non-finite evaluated FE or inner residual is divergence, while proven empty discrete support takes infeasible precedence.
- Float32: outer/inner tolerances and Newton safeguards derive from `jnp.finfo(dtype)`; no absolute floors/ridges.
- Scale: tests cover a 100-variable chain plus a larger GFE simplex; iteration status must remain truthful at budget.
- Poisoned messages: terminate immediately as diverged.
- Budget: deliberate too-small budgets report `budget_exhausted`, never converged.

Design: `Engine.run` returns a non-tuple `RunResult`; callers must explicitly inspect status / require convergence. Sum/GFE convergence requires FE stationarity plus message coherence. Max mode uses message fixed-point residual and validates the final decoded objective.

**2026-07-11T16:10:12Z**

Implemented on branch `cff-j4mu-inference-status`.

Commits:

- `956269c`: dtype-scaled, gauge-projected GFE Newton solver with converged/budget/diverged diagnostics and K=80 float32 test.
- `4c81798`: public non-tuple `RunResult`; FE-stationarity + coherence convergence; max residual/objective validation; divergence and exact-support infeasibility; GFE diagnostics threaded outward; RollingAgent/Colony/decimation/estimator/test/dashboard migrations.

Hostile coverage: sparse/exact-zero support, float32, 100-variable chain, poisoned message, contradictory `exactly_k` CSP, deliberate budget exhaustion, large GFE simplex.

Compatibility: old `state, deltas` unpack is intentionally loud; use `result.state` for deliberate fixed budgets or `result.require_converged()`.

Verification: dashboard full suite 1102 passed, 0 failed on 2026-07-11. Dashboard refreshed at `~/agent-share/cffg-tests.html`.

Gotcha: one curiosity fixture's old 50-sweep budget was genuinely under-converged; increased to 100 after status exposed it.
