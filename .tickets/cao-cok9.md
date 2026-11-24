---
id: cao-cok9
status: closed
deps: []
links: []
created: 2026-11-24T06:38:10Z
type: chore
priority: 2
assignee: coopwilliams
tags: [tests, runtime]
---
# Restore time margin in the default numerical test suite

## Background in Plain English — WHY

The November 23 CPU run took 1,400.17 seconds and selected 2,924 tests. Its four failures remain active regressions. The project budgets 1,200 seconds for routine validation. Some end-to-end examples and combinations of an algorithm, schedule, or precision repeat claims already exercised in the short run.

## Outcomes / Acceptance Criteria — WHAT

1. The default collection retains all four observed failures, representative numerical oracles, float32 stress, and coverage for new failure and restart paths.
2. The moved cases remain runnable with the slow marker and retain their assertions.
3. Subtracting measured durations of moved cases from the recorded wall time gives an estimate below 1,175 seconds. Record the baseline, savings and uncertainty.
4. The before and after collections, including all opt-in cases, account for the same test cases.

## Methodological Suggestions — HOW

Use the runtime numerical or status invariant row of the architecture change-path index as the coverage checklist. Check the short and slow pytest selections, dashboard file ownership and representative host/JIT, exact/iterative and float32/float64 paths. Runtime dispatch, serialization and model definitions are N/A for these selection edits. Prefer local markers on duplicate parameter choices.

## Future work

The integration owner will run the corrected short suite at the final head and record the actual duration.

## Definitions

A parameter choice is one set of inputs to a test function. An opt-in test uses pytest’s slow marker and runs when explicitly selected.


## Notes

**2026-11-24T06:47:55Z**

The baseline CPU run selected 2,924 cases. It took 1,400.17 s with 2,920 passes and four failures. The revised default selects 2,798 cases across the same 159 test files. The other 379 cases run with `-m slow`. Before and after, the combined inventory contains exactly the same 3,177 IDs. All 126 cases moved out of the default had passed in the original run. Of those, 103 are parameter variants whose functions still run by default. The other 23 are complete scenarios covered below.

The estimated short-suite time is 1,168.95 s, giving 31.05 s of headroom against the 1,200 s budget. Subtract 231.223 s of observed case durations from 1,400.17 s. Match the original dashboard's case timings to collected IDs and use the original pytest durations log for names shared across files. Credit zero savings for the one ambiguous `test_recurrence_link::test_vmap_step_matches_sequential` timing. The estimate retains the original process overhead. JIT costs can shift to a later test, and hardware load can vary. The integration owner's next run will measure the revised wall time.

Retained short-suite coverage for the complete scenarios now selected with slow:

- The full Sudoku decimation scenario exercises the same finite constraints covered by exact all-different marginals and MAP, Latin-square table constraints, and two seeded bipartite matching oracles.
- Goal seeking, ambiguity ablation, the high-count learned composite, and the behavioral hybrid probe retain the discrete GFE local-objective minimum, direct message/energy formula, learned novelty direction, and the separate hybrid explore-and-win and ablation tests.
- Switching-system fit at tenfold horizon, joint parameter rollover, plate pooling, and the scheduled/flooded Gaussian and Pólya–Gamma update scenarios retain explicit fit budget and residual checks, independent tied-source Pólya–Gamma and matrix normal–inverse Wishart oracles, 64-owner bundle resume, dense quadrature, and float32 K32/d16 state and parameter updates.
- The three sequential-versus-batched integrations and nonlinear Colony certificate scenario retain independent Gaussian engines, mixed members, failure and budget cases, adaptive startup under vmap, and compiled per-rule vmap tests for recurrence, mixture and hybrid GFE.
- The two-pass Kalman scenario retains the analytical RTS smoother **and the direct Kalman BFE = negative log likelihood test**. The genuine two-pass float32 T256 test and the two-pass HMM forward/backward oracle retain scheduled-route checks.
- Mixed/flooded recurrence and hybrid-root comparisons, group reordering under damped shared goals, and two hybrid readout scenarios retain fixed-drive pair-region energy, dense hybrid oracle, shared-goal pooling and coordinate alias checks, 32-composite sparse support, independent soft-weight algebra, and improper-input/float32 readouts. Their full schedule, damping and multi-member combinations remain available with slow.
- The neural large-parameter memory/work scenario retains both source precisions of the network Jacobian oracle and the generic cost estimator. The complete neural hybrid gradient retains public float32 fixed-point gradient checks, sparse K32 forward/reverse gradient agreement, and the zero-predictive-row derivative regression.

Verified both collections with `uv run --frozen --no-sync pytest --collect-only -q` and `uv run --frozen --no-sync pytest --collect-only -q -m slow`. Compared their disjoint union to the original full collection and checked the four failed IDs from /tmp/cffg-axiom-overnight-final-suite.log. The compiler-value import, core import, registration ownership and README export regressions all remain short. The coordinator owns the next short-suite run after fixing those regressions.
