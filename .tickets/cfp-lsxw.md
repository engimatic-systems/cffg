---
id: cfp-lsxw
status: closed
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

**2026-11-22T03:38:43Z**

Closed at the prespecified ten held-out pairs. Timed applications ran in 20 separate fresh CPU processes at 6331e24. Both arms completed 590 decisions, with all 1180 observed noise innovations independently checked against their seed. Directed mean return −1.780842 vs host −2.134100; paired difference +0.353259 [0.335428, 0.374467] across seeds. Directed had 0 missed budgets or holds and 77 clipped proposals. Host missed the first budget on every seed, held 10 times, and clipped 68 proposals. The host was faster once warm. No failed controller checks or applied-limit violations. 50/50 directed windows passed stock parity, and both controller self-replays passed. Replacing only the first host hold with its checked and clipped proposal in a post-hoc one-step plant replay improved first reward from −1.09327 to −0.73870 (directed −0.87094). This does not measure an entire deadline-free episode. The fixed-model result supplies cff-wt1c; parameter uncertainty, bounds and facade comparisons remain separate.

**2026-11-22T03:38:43Z**

Reproduction from the tracked tools: cstr_paired_study.py run --output STUDY/run, cstr_paired_replay.py STUDY --output STUDY/replay, cstr_paired_first_step.py STUDY/run STUDY/first-step-intervention.json, and cstr_paired_report.py STUDY. Use uv run --frozen python. HTML, chart, raw windows, integrity checks, runtime sources and per-seed intervals are at ~/agent-share/cffg-reactor-paired/. The report and 171 local links were checked over HTTP. The seed-0 comparator adapter matched the original 59/59 unclipped proposals and applied commands with the clock disabled; a forced miss checked hold propagation. The standalone experiment and reporting code changes no package, engine, production numerical rule, pytest selection or pass/fail interpretation. Focused simulations, negative controls, replay and independent raw-data validation are the ticket gate.
