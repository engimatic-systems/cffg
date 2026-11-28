---
id: cao-8t1j
status: closed
deps: []
links: []
created: 2026-11-24T17:20:36Z
type: task
priority: 2
assignee: coopwilliams
parent: cao-srin
tags: [axiom, report]
---
# Report the sealed 10,000-action Explode development pilot

## Background in Plain English — WHY

Cooper wants the Explode pilot reported before deciding whether to run the later stages. The pilot has one development model seed and one simulator seed. A reader should be able to understand what the agent did, what it earned, and how much time it took.

## Outcomes / Acceptance Criteria — WHAT

1. The renderer requires the frozen study and signed-off pilot receipt, verifies the result and step-file hashes, and reads only development evidence.
2. The report shows the actual 10,000-action total, catches and misses, and the ten successive 1,000-action totals. It gives the published untuned mean and standard deviation across ten seeds as descriptive context.
3. The report identifies the seeds, timing clocks and denominators, scheduled reductions, and the scope of the single development run.
4. Synthetic data cover successful, incomplete and tampered evidence and check the score, chart and interpretation. The campaign owner renders the real archive after closure.

## Methodological Suggestions — HOW

Reuse the sealed campaign scorer and compare the new displays with hand-constructed rewards. For the architecture change-path index this is a standalone experiment report. Host/device inference, solver, serialization and campaign execution surfaces are N/A.

## Future work

cao-srin owns publication using real evidence. The two later stages remain deferred at this reporting milestone.

## Definitions

A catch pays one point. A miss costs one point. A run's return sums its rewards over all real actions. A standard deviation describes variation across the ten published seed returns.


## Notes

**2026-11-24T17:25:28Z**

Ready for the campaign owner. After pilot closure run: uv run --frozen --no-sync python tools/axiom/explode_report.py ~/agent-share/axiom-10000. Writes explode.html and explode-block-rewards.png. Seals the exact study bytes, pilot receipt, result and 10,000 native action records. The report states the same-host suite overlap in UTC. This is a post-processing tool outside the frozen source inventory and changes no campaign runner, scorer, selection or numerical code. Four focused checks passed with uv run --frozen --no-sync python tools/axiom/explode_report_check.py. The synthetic oracle had 165 catches, 110 misses, net +55, ten known block scores and known per-action clocks. Missing and altered receipts/records, restarted timing scope, stage isolation, local links, unchanged index and 1440 x 624 PNG verified. Chart inspected visually, py_compile passed. All outcome reads and report renders by the author used temporary synthetic data. The campaign owner handles the actual render.
