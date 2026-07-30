---
id: cff-weff
status: open
deps: [cff-wafy]
links: [cff-2gev, cff-wbhu]
created: 2026-07-30T14:04:30Z
type: bug
priority: 2
assignee: coopwilliams
tags: [gfe, solver, numerical-robustness]
---
# Globalize mixed-GFE outer fixed-point inference

## Plain English

Mixed-GFE plans will execute the proof-backed outer solver selected by
cff-wafy and will either certify the original constrained-GFE stationarity
conditions or report why they were not reached. The implementation will retain
the papers' local message equations and exact node/edge free-energy ledger
while replacing caller-tuned outer damping. Inner Newton success will remain
separate from coupled outer convergence.

## Motivation and evidence

The retained run on closed PR #43 exposed the defect in its first learn-only
cell, before annealing could activate: after one learned transition count
changed, the inner Newton solve converged while the outer mixed-GFE message and
free-energy sequence entered a finite limit cycle. Fixed damping values from
0.5 through 0.99 and budgets up to 15,000 sweeps did not recover convergence.
The failure is therefore in the coupled outer Picard iteration, not the local
Corollary-1 Newton solve.

Closed PRs #42 and #43 retain the rejected annealed-novelty design and its
invalid run for historical comparison. Annealing scaled a scientific novelty
term; it neither caused nor repaired the outer-solver defect, and the run
stopped before that scaling activated. This ticket implements no annealing,
objective reweighting, temperature schedule, or experiment-derived threshold.
The redesigned nonstationarity research remains cff-2gev and is independent of
the numerical method used here.

## Design

Implement exactly the route selected and proved by cff-wafy:

- If cff-wafy proves a tight double-loop majorizer for the supported
  P-substituted objective, implement its factor-local convex inner solve and
  outer bound update, including the proof's original-GFE descent certificate.
- Otherwise implement its gauge-reduced trust-region
  Gauss--Newton/Levenberg--Marquardt solve of the coupled KKT/fixed-point
  residual. The residual merit governs step acceptance but never replaces
  `Engine.bfe` or a policy score.

No generic "adaptive damping or backtracking" implementation satisfies this
ticket. The production method must instantiate the selected derivation,
including its assumptions, active-support behavior, dtype scaling, acceptance
ratio or majorization certificate, and failure conditions.

Put the JAX-native numerical core in a focused module with a one-way dependency
from `engine.py`. Use factor-local operations or matrix-free JVP/VJP products;
do not allocate a dense whole-graph Jacobian or Hessian and do not add a solver
dependency. Preserve the existing local `_gfe_solve_result` equations and
diagnostics.

`Engine.run` should select the proof-backed method explicitly for mixed-GFE
plans. The existing `damp` argument remains meaningful only for non-GFE
iterative schedules or an explicitly requested legacy Picard diagnostic; it
must not be silently reinterpreted as a trust radius, regularizer, or
majorization parameter. Migrate project-owned callers away from fixed damping
for mixed GFE.

`RunResult` must expose the accepted outer residual/merit or majorization
certificate and a termination reason that distinguishes convergence, inner
failure, non-finite evaluation, infeasibility, detected cycle/stagnation, and
iteration/linear-solve budget exhaustion. Public `RunStatus` may coarsen those
reasons only if the detailed reason remains machine-readable. `CONVERGED`
requires all local solves, the coupled outer certificate, support checks, and
the unchanged FE ledger to be finite and stationary.

## PR chain and downstream contract

The intended chain is:

    cff-wafy formalism/reproducer PR
        -> cff-weff implementation PR
            -> cff-wbhu compiled colony loop
            -> cff-izjd hybrid GFE composite

The formalism draft starts from `main` on
`agent/cff-weff-formalism`; the proof author continues on that same branch.
The implementation branch is then cut from the accepted formalism head and its
PR targets the formalism branch until the first PR merges, after which it is
retargeted to `main`.

cff-wbhu is linked sequencing work rather than a mathematical dependency, but
the selected solver core must be compatible with `jit` and `vmap` so that
ticket does not reimplement convergence semantics. cff-izjd is a dependency
consumer: a new mixed P-substituted composite must not land on the known
un-globalized outer loop.

## Change-path handoff

Applicable row: **Runtime numerical or status invariant**.

- Authority: `AGENTS.md` numerical contract, cff-wafy's
  `docs/DERIVATIONS.md` result, and `src/cffg/engine.py` (`RunStatus`,
  `RunResult`, `Engine.run`).
- Execution surfaces: flooding and mixed schedules; `_make_step`,
  `_iter_step`, `_make_bfe`, host support checks and their JIT counterparts;
  inner-solver status aggregation; `RollingAgent`, `Colony`, estimator,
  benchmark, and dashboard consumers.
- Duplicated pairs: host/JIT and flooding/mixed are affected;
  single-instance/`vmap` must share one solver contract; exact two-pass
  schedules and current/legacy checkpoint decoding are N/A.
- Focused tests: `tests/test_gfe.py`, `tests/test_b_epistemics.py`,
  `tests/test_schedule.py`, `tests/test_runtime_status.py`,
  `tests/test_agent.py`, and `tests/test_colony.py`.

## Hostile Inputs

- **Exact log-zero support:** use the fixed-support/active-set construction
  proved by cff-wafy. Feasible zeros stay exact; support changes restart the
  local chart; incompatible support is infeasible, not regularized.
- **Vacuous or improper input:** a defined feasible iterate may globalize;
  an undefined residual, improper belief, or non-finite ledger maps to the
  proof's truthful failure reason.
- **Float32:** all residual scaling, stopping rules, rank tests, trust-radius
  or bound guards, and linear-solve tolerances derive from the active dtype and
  problem scale.
- **Uncomfortable scale:** cover the horizon-8 factored model with 24-logit
  GFE-TM members and retain the 32-position by 4-selector rule case without a
  dense whole-graph derivative matrix.
- **Finite cycle:** reproduce the one-update mechanism independently of the
  abandoned annealing experiment and show fixed Picard damping still fails
  while the selected method reaches or truthfully rejects the same equations.

## Acceptance Criteria

1. The cff-wafy cycle fixture reaches a certified outer solution under the
   selected method without changing its model, local rules, objective, or
   tolerance; explicitly selected fixed Picard mode still demonstrates the
   historical failure mechanism.
2. Existing convergent observation-GFE, D1 transition-GFE, T-maze, and
   two-composite constrained-objective fixtures retain their marginals,
   messages, and ledger values to their existing analytic tolerances.
3. `CONVERGED` requires the proof's outer certificate plus converged inner
   solves and finite support/FE checks. A deliberately unsolved nonzero-merit
   case returns budget exhaustion or stagnation; NaN input diverges;
   impossible exact support is infeasible.
4. `RunResult` exposes machine-readable outer diagnostics and preserves the
   existing FE and message traces at accepted iterates. Solver merit remains
   absent from the free-energy ledger and policy-scoring API.
5. Project-owned mixed-GFE callers use the proof-backed method without
   caller-tuned damping. The core is JIT- and `vmap`-compatible, with parity
   across flooding/mixed and single/batched execution.
6. Sparse support, vacuous/improper input, float32, and uncomfortable-scale
   cases satisfy the numerical robustness contract with no new dependency.
7. `docs/ARCHITECTURE.md`, `docs/DERIVATIONS.md`, `docs/LIMITATIONS.md`, and
   `README.md` state the solver contract and its honest boundary: numerical
   globalization does not guarantee the global minimum of a non-convex CFFG.
   The full suite, `tk dep cycle`, and `git diff --check` pass.
