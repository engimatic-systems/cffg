---
id: cao-ciro
status: closed
deps: []
links: []
created: 2026-11-24T16:22:57Z
type: chore
priority: 2
assignee: coopwilliams
parent: cao-srin
tags: [tests, runtime]
---
# Bring measured short-suite time back under twenty minutes

## Background in Plain English — WHY

All 2,808 selected tests passed during the 10,000-step development pilot. Pytest took 1,317.22 seconds. The dashboard measured 1,333.136 seconds for the whole command, including setup and reporting. The project budget is 1,200 seconds. Concurrent work makes that measurement unsuitable for estimating the pilot's own speed.

## Outcomes / Acceptance Criteria — WHAT

1. Moving repeated combinations and longer integrations to the slow suite gives a conservative estimated dashboard wall time under 1,200 seconds.
2. The default retains direct mathematical oracles, hostile-input cases, public failure checks, and the campaign changes exercised by the passing run.
3. Both selections together contain exactly the same test IDs and assertions. The ticket records the measured wall, projected time, retained witnesses, and limitations.

## Methodological Suggestions — HOW

Compare the log and dashboard case durations for this run. Check each moved case against a retained witness. Use the runtime numerical/status row of the architecture change-path index as the coverage checklist. Changes to runtime, numerical rules, source seals, scoring, I/O and dashboard code are N/A.

## Future work

cao-srin owns the development campaign. Reuse this passing run for selection-only changes. A future independently required suite will measure the new wall time.

## Definitions

A test ID identifies one input choice for one test. Slow tests stay available by explicitly selecting the slow marker. A witness is a retained test that directly checks a stated behavior.


## Notes

**2026-11-24T16:29:39Z**

The integration run passed all 2,808 default tests, with 380 slow tests deselected and four warnings. Pytest measured 1,317.22 s. The dashboard measured 1,333.136 s wall time. The 10,000-action development pilot was running on the same host during that suite. The earlier 1,163.637 s run had a different selection and workload. No isolated timing here attributes a number of seconds to the pilot.

The new default selects 2,714 tests in all 161 previous default test files. The slow suite selects 474. Exactly 94 passing cases moved. Of these, 78 are individual variants whose test function still runs by default, and 16 are entire extended scenarios. The disjoint union of both selections is exactly the original 3,188 test IDs. All ten default tests added since f74543f remain selected, including Hunt multi-point reward handling, exact checkpoint corruption and atomic refusal, reduction bounds/ties, and the rounded precision opt-out, restart, and invalid-weight cases.

Estimated dashboard wall time is 1,170.313 s. The 94 test IDs account for 162.823 s in the completed run. Subtracting only their measured case times from 1,333.136 s leaves the recorded 15.916 s of dashboard overhead in place. Durations came from the run's dashboard and /tmp/cffg-axiom-10000-final-suite.log. A conservative sensitivity check used the smaller of this run's time and the November 23 overnight run's time for each of 93 IDs. It credited zero savings for the HMM vmap test, whose older dashboard name also occurs in other files. This gives 135.078 s of savings and an estimated 1,198.058 s wall. Both values are estimates. Compilation can move to a later test and future host load can differ.

Coverage retained for the longer scenarios and parameter matrices:

- Solver comparisons. The T-maze five-run cold/warm/Picard/changed-prior comparison and the two full cubic-sensor solves with forced trust fallback move to slow. The default retains matrix-free trust single/vmap parity, cache keys, the T-maze LAIF golden, float32 invalid-proposal fallback with budget accounting, the host-energy veto, uncertified proposal rejections and independent curved-sensor fixed-point checks.
- Batched execution. Full HMM sequential/vmap, custom Gaussian kernel Colony, and nonlinear two-instance host/vmap integrations move to slow. Independent HMM forward/backward and BFE, exact multi-member Gaussian Colony versus individual engines, observed custom Gaussian scan/trace, adaptive mixed-ready vmap and each local rule's numerical tests remain. The broad execution matrix keeps hybrid GFE and recurrence vmap witnesses. Additional plain-mixture and transition combinations are available with slow. Rank-deficient and failed mixture members still run in genuine batches.
- Gaussian and mixture results. The Kalman improper-edge smoke repeats the public single/bulk NaN-and-recovery test, which still runs in both precisions. Kalman BFE = negative log likelihood, RTS smoothing, missing evidence and the observation cavity all remain. The switching Kalman filter's loose BFE-distance canary moves to slow while exact enumeration, collapsed-reference comparison and the hand-assembled hybrid ledger remain. Single-regime engine/linear parity, identical-regime forward collapse, the ordinary float32 finite/close smoke and scalar predictive quadrature move to slow. The default retains the direct K=1 forward/backward/energy identities, independent quadrature for mixture messages and energy, multivariate weighted predictive moments, equal likelihoods with nonuniform selector priors, and explicit float32 singular/vacuous and uncomfortably large cases.
- Recurrence and planning. Repeated schedule-to-schedule recurrence root comparisons and an extra dtype replay of diagnose_case move to slow. The default keeps enumeration versus forward/backward, the public pair/learning oracle, dense quadrature/energy bounds, multistick learning, and flood/mixed equality at exact-zero support. The three-block rollout compile/run/round-trip and the goal-hyperprior next-policy behavioral demonstration move to slow. The independent two-block joint and BFE oracle, factored sparse float32 workload, tied transition updates, goal posterior and accumulated free-energy algebra and two-window rollover remain.
- The simulated basal-only glucose benchmark joins the other opt-in patient episodes; the cohort definition remains short. Run the slow selection when evaluating the controller's patient outcomes.
- For the remaining matrix reductions, the stricter or directly relevant case stays short, for example float32 warm updates, 200-member outer geometry, the 5,000-slot bank lifecycle and the three-regime public pair oracle. Core double-precision mathematical oracles remain. The short collection still exercises all injected failure types, both schedules and both precisions across their owning tests.

Validated before/after default and opt-in collections with `uv run --frozen --no-sync pytest --collect-only -q` and the same command with `-m slow`. An AST comparison across the 43 edited modules verified that outside pytest markers and their import, the assertions, fixtures, inputs to test bodies and test helpers stayed unchanged. The selection-only budget rule reuses the green suite. The dashboard history and study evidence remain with the campaign owner.
