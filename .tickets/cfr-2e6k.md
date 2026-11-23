---
id: cfr-2e6k
status: in_progress
deps: [cfr-yuhl, cfr-zc4v, cfr-dliy, cpr-tnuq, cpr-pch9, cpr-rjpb, cfr-ex1c, cfr-lben]
links: []
created: 2026-11-23T03:19:11Z
type: task
priority: 1
assignee: coopwilliams
tags: [premerge, validation]
---
# Qualify the cumulative reactor and inference changes for a merge into main

## Background in Plain English — WHY
Forty-one pull requests share a line of development that ends at PR 195. A final review found numerical, public API, and benchmark-tool defects. Cooper authorized merging the reviewed changes after the corrected engine passes its tests. The reactor tools check for a historical source revision in the Git history.

## Outcomes / Acceptance Criteria — WHAT
1. Each review finding has a verified fix and an independent review.
2. A fresh qualification reports accuracy and cold and warm latency for the corrected directed controller against the same archived reactor decisions.
3. The default short suite passes on the integration code. Its measured wall time and any test-budget adjustment are recorded.
4. The review identifies all ancestor pull requests, records outstanding review or build checks, and confirms the historical revision is reachable from the proposed merge head.

## Methodological Suggestions — HOW
Use the architecture change paths for runtime status, readouts, training and persistence. Run the archived directed controller qualification with a fresh cache. Review the production fixes and the two archive-tool fixes separately. Use an ordinary GitHub merge commit and verify the remote main history after the merge.

## Future work
More complex control benchmarks can build on the merged engine. They use a separate experimental design.

## Definitions
Cold latency is the time from launching a fresh process to its first applied action. Warm latency measures a decision after compilation. An ancestor is a source revision reachable by following the parent links of a later Git commit.


## Notes

**2026-11-23T03:21:48Z**

Reviewed all 41 ancestor PRs and the changes through the premerge fixes. The main branch requires a PR and allows a merge commit. No unresolved review threads or pending hosted checks were found. The unrelated PR 140 falls outside this history. Independent review reproduced the numerical fixes and an old optimizer checkpoint export. Review also found that a duplicate seed could conceal omitted archive manifests; the revised auditor enforces the 30 ordered seeds and all 394 distinct worker paths.
