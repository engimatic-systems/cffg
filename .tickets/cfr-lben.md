---
id: cfr-lben
status: closed
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

## Notes

**2026-11-23T03:09:54Z**

Isolated benchmark orchestration. The architecture change-path index has no campaign persistence row. The runner, canonical episode inventory, worker metadata audit, and report timing boundary are affected. Engine runtime status, host/JIT, batch scheduling, model serialization and checkpoint decoding are N/A. Use injected failures in a temporary campaign and read the sealed original archive in place.

**2026-11-23T03:19:00Z**

The runner stages each measurement outside the scored inventory and publishes the validated trace before its sidecar. On an unsealed, source-matching campaign it preserves known old timestamped partials and orphaned traces in unique attempts directories. Unknown JSON artifacts, corrupt sidecars and lost completed traces fail validation. Children retain the study lock if the parent exits. Finished studies and qualification seals are protected. Process time is available only for a first measured launch publishing all eight episodes. A retry or a missing parent clock gets null, the campaign total becomes unavailable, and the report shows the full-clock denominator. Older frozen evidence keeps its recorded timing interpretation. The scorer has a read-only --check option.

**2026-11-23T03:19:01Z**

Focused gate: uv run --frozen --no-sync pytest -q tools/pcgym_timing/continual_comparison_recovery_check.py passed 10 cases. Covered worker failure before setup, midway through trace, between trace and sidecar publication, after all eight episodes, and after worker record publication. Also checked legacy and unknown files, missing or tampered completed artifacts, missing seals and survival of a child process lock. Four modules compiled and tk dep cycle passed. Actual --check on ~/agent-share/pcgym-continual-comparison audited 320 episodes and 191680 decisions read-only. In-memory analysis matched the archived JSON exactly, including all 692 hashes and 40 full clocks. A temporary replay copy with a legacy partial and a one-episode retry produced identical paired results and audit, retained both incomplete artifacts and reported 39/40 full clocks. Validation was confined to benchmark tooling. Parent owns the integrated suite.
