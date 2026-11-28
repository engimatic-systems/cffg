---
id: ctcp-qodi
status: closed
deps: [ctcp-w305]
links: []
created: 2026-11-28T04:50:36Z
type: task
priority: 2
assignee: coopwilliams
parent: cff-2gev
---
# Preregister and run the changing-corridor transition-curiosity comparison

Find out whether a changing transition prior restores purposeful information seeking and prediction after a real change in the factored corridor. Task success, posterior uncertainty and compute are different outcomes.

## Design

Use the causal model, applied action and true solver status in every arm. Do not grant any arm the hidden change label or reset signal. Keep held-out runs sealed until all three serialized models exist and the protocol passes a development-only audit.

## Acceptance Criteria

Meet cff-2gev sections 3 and 7. Freeze every model and threshold, per-arm budget and seed list before scoring. Retain all stationary, no-epistemics and changed-law outcomes, including nonconvergence. Report unchanged and changed blocks, predictive quality, visitation, task success, runtime and memory separately.


## Notes

**2026-11-28T13:54:06Z**

Preregister a 2-position, 2-mode corridor. The changed world reverses one position column at a known scorer tick while mode dynamics stay fixed. All arms share the scripted exposure prefix, goal, sensor, action prior and random-number streams. The changing BP control removes the full transition P-substitution. The observation goal stays in every planner. Architecture change path is isolated experiment use of existing public model, solver, learning and bundle surfaces. No core equation or scheduler change. Runtime status and result interpretation require a fresh short suite. Development evidence is disjoint and cannot establish a comparative outcome.

**2026-11-28T13:57:12Z**

Version 1 stopped on disjoint development 19501. All arms received 33 probe visits. Four of six cells failed the requirement D1<=0.08 on every one of the final eight warm pre-action forecasts (largest changing forecast 0.08897). Both stationary cells passed. Full cells and failure receipt preserved in ~/agent-share/changing-corridor-v1-gUpybO. No held-out outcomes were opened. Version 2 retains the world, hazard, timing, thresholds and formal seeds. Its shared physical acquisition uses only the last jointly observed position and mode to target the probe and otherwise return to position zero. It must qualify on both fresh development seeds 19511 and 19512 before formal access.

**2026-11-28T14:01:45Z**

Both preregistrations stopped before opening the six formal seeds. Recomputed from physical actions and pre-action forecast arrays: v1 seed 19501 had 33 visits, max D1 .014758 stationary vs .088966 both changing variants; the latter exceeded .08 on 4/8 late ticks, yielding 2/6 complete development cells. V2 with observation-only targeted acquisition, seed 19511: 45 visits, maxima .011142 vs .147090, 2/6 complete. On tick 126 a nominal 8%-probability self-transition raised target D1 from .0609 to .1471. Seed 19512: 53 visits, maxima .009475 vs .051885, all 6/6 complete. V2 total 8/12. No solver failure or formal-estimand measurement underlies either readiness verdict. Status open pending Cooper’s choice on sustained pre-change saturation as a prerequisite versus nominal false alarms as a distinct measured outcome. No threshold changes or further seeds.

**2026-11-28T14:01:45Z**

Replay the original generation in clean checkouts with CPU/x64: v1 298943d, v2 50f177f. In each revision run uv run python tools/changing_corridor.py prepare ROOT in a fresh directory, then uv run python tools/changing_corridor.py development ROOT --seed SEED --arm ARM --world WORLD for every development seed (v1 19501, v2 19511/19512), each arm changing/stationary/no_epistemics, each world change/sham. Archived version 1 and 2 raw tapes, serialized model bundles, solver statuses and hash-checked readiness summaries are in ~/agent-share/changing-corridor-v1-gUpybO and ~/agent-share/changing-corridor-v2-hZmyRk. From the review branch run uv run python tools/changing_corridor_study.py readiness ARCHIVE to verify a preserved stop. V2 qualify independently checks model identity, pre-change histories, statuses and final eight forecasts before any freeze can be written. Formal campaign and scorer are exercised by synthetic checks only.

**2026-11-28T14:10:13Z**

Terminal disposition — stopped, readiness failed. Cooper decided: “Keep the requirement that curiosity settle first. Publish the failed readiness checks and stop this comparison.” V1 passed 2/6 and V2 passed 8/12 development cells. Zero of 36 formal cells opened, with all six held-out seeds sealed. The changing learner implementation is qualified. The required precondition failed on frozen development evidence, the formal demonstration was not established, and no post-change effect was measured. No follow-on protocol or tuning is authorized. Exact development archives, original qualification receipts and independently recomputed readiness reports are published under notes/changing-corridor-readiness/v1 and v2. To replay, copy one version directory, unzip development.zip into it, and run tools/changing_corridor_study.py readiness on that directory with uv run python from this review branch.
