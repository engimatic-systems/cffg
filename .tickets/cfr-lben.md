---
id: cfr-lben
status: open
deps: []
links: []
created: 2026-11-23T03:03:36Z
type: bug
priority: 2
assignee: coopwilliams
parent: cff-xa6z
tags: [premerge, pcgym, recovery]
---
# Make interrupted four-library reactor runs resumable through final scoring

## Background in Plain English — WHY
In #195 the runner keeps an interrupted trace under its normal episode directory with an extra .jsonl filename. The scorer rejects any extra .jsonl file there. A resumed campaign can therefore reach the final analysis and still fail. The runner can also record only the final retry process duration as the time for an entire eight-episode block. The measured archive has no interrupted traces.

## Outcomes / Acceptance Criteria — WHAT
1. An interruption midway through an episode can be resumed and scored with all canonical episodes present once. The incomplete evidence is preserved outside the scored inventory.
2. Report retry timing with an accurate boundary. Do not represent time for the remaining episodes as the original full eight-episode process time.
3. Unknown canonical-directory traces, missing completed traces and mismatched sidecars still fail validation.

## Methodological Suggestions — HOW
Exercise continual_comparison.py:167–170 and :237–242 with continual_comparison_analysis.py:353–355. A temporary mirror of the complete evidence with one extra stationary-candidate.interrupted-12345.0.jsonl file reproduces the present scoring failure. Use a small interrupted-worker fixture. This is isolated benchmark orchestration.

## Future work
cpcc-2dc3 records the completed comparison. No changes to scoring, the numerical models or the archived results are required.

## Definitions
A trace records the decisions taken in one episode. A sidecar records its identity, duration and content digest.
