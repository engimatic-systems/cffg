---
id: cff-sten
status: open
deps: [cff-yy73]
links: []
created: 2026-09-13T16:54:37Z
type: task
priority: 3
assignee: coopwilliams
tags: [testing, performance]
---
# Reuse prepared constraint checkpoints across migration checks

## Background in Plain English — WHY

The cff-yy73 timing audit found repeated preparation in constraint checkpoint tests. Six migration cases took about 31 seconds and three rejection cases about 14 seconds in the shared dashboard at commit 48dbc8b. All call prepared_checkpoint, which solves an agent before writing an archive. Several cases use the same constraint family. The distinct migration formats and rejection mechanisms remain valuable; repeated setup may be avoidable.

## Outcomes / Acceptance Criteria — WHAT

1. The migration and rejection cases retain their existing constraint-family and archive-format coverage.
2. Reused preparation cannot share mutable arrays, agents or target state between cases.
3. Each rejection still tests the actual corrupt archive and verifies the intended target-state invariant.
4. Focused measurements demonstrate whether reuse reduces execution time, with the compared selection and source revision recorded.
5. Final validation follows the cff-yy73 test protocol and records one complete PR gate after the final change.

## Methodological Suggestions — HOW

The implementer may choose another sound method. Inspect prepared_checkpoint and its callers in tests/test_constraint_reconstruction.py. Consider one prepared immutable archive per constraint kind, loading independent array copies and fresh restore targets per case. Follow the fixture pattern in tests/test_agent_mu3.py without narrowing the three constraint families or the current/agent_v1 formats. Use durations from necessary validation; do not run an extra broad benchmark suite. Before implementation record the applicable architecture change path and affected execution pairs.

## Future work

No further work is identified. This follows cff-yy73 to continue reducing repeated setup after the testing protocol is established.

## Definitions

- Prepared checkpoint: an archive written after solving the agent's planning state.
- Constraint family: exactly_k, at_most_k or table, each with a separate implementation.
- Immutable: cannot be modified after creation; bytes are suitable for shared archive contents.
- PR gate: the complete required test selection at the proposed pull-request head.


