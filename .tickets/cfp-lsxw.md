---
id: cfp-lsxw
status: in_progress
deps: [cfp-9806]
links: []
created: 2026-11-22T01:00:18Z
type: task
priority: 1
assignee: coopwilliams
parent: cff-xa6z
tags: [pcgym, experiment]
---
# Measure paired closed-loop reactor control and first-action latency

## Background in Plain English — WHY

A fast plan can still yield poor control. Use the recorded pilot protocol to measure the full controller on matched episodes. A negative result is a valid outcome.

## Outcomes / Acceptance Criteria — WHAT

1. Run the agreed paired seeds and setpoint sequences for the directed controller and the comparison controller. Record revisions, hardware, noise, physics knowledge, preview, horizon, clipping policy and compute access.
2. Report per-seed and paired differences in return, tracking error, bound violations, deadlines and fallbacks. Report uncertainty and the number of episodes; retain every failed episode.
3. Report process launch to first checked action, cold solve and repeated-step latency separately. Label disk cache, resident code and any reused belief state.
4. Separate solver equivalence on identical replayed windows from reward on trajectories that can differ after the first action. Archive commands, raw measurements and the conclusion in the ticket.

## Methodological Suggestions — HOW

Prefer the existing host-linearized controller if the pilot establishes a matched comparison. A short deterministic smoke test can stay local; use the external PCGym checkout for longer trials. Runtime numerical/status row is consumer-only. Engine/Colony equivalence, learner and checkpoint implementation are N/A.

## Future work

Feed the fixed-model result to cff-wt1c; its parameter-learning, action-bound and facade comparisons remain separate.

## Definitions

A paired seed exposes both controllers to the same initial conditions and source of environment noise. A fallback is the action applied after a failed or late plan.

## Notes

**2026-11-22T03:24:16Z**

Implementing on experiment/paired-reactor-and-bounds above #187. Architecture row Runtime numerical or status invariant as an external consumer only. Reuse identical pilot simulation, clipping, deadlines, statuses and observation history. Compare the directed reference to the real host controller; replay saved windows through Engine separately. No production Engine, Colony, learning, checkpoint or scheduling semantics change. The held-out protocol and the seed-pair bootstrap are fixed before any comparisons. Keep all failed or late trials and archive raw diagnostic windows.

**2026-11-22T03:29:11Z**

Before held-out measurement: odd seeds run directed first and even seeds run host first. Descriptive near-switch bin uses scored steps 18-22 and 38-42 (decision ticks 17-21 and 37-41), within two steps of the switches. Record both the widest raw proposal gap and widest applied action gap if they occur on different inputs. Replay failures/misses and selected windows from immutable recorded history. Pilot/control facts and raw source hashes are in the manifests.
