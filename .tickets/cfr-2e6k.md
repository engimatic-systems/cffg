---
id: cfr-2e6k
status: closed
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

**2026-11-23T03:51:14Z**

Both clean-code qualifications passed on CPU float64 with an empty disk cache. Pre-fix 5c4c43b versus corrected 2967b07: 599 matched decisions, 59 archived coupled comparisons, all checks green. Maximum change in raw coolant command was 5.69e-14 K. Fresh process to first action 1.993 s versus 4.078 s (one launch each), warmed median over 590 held-out decisions 7.788 versus 8.345 ms. Raw artifacts are ~/agent-share/cffg-premerge-195/directed-baseline.json and directed-final.json. Production verification adds real cold latency. Archived #195 was independently audited read-only, 320 episodes, 191680 decisions, 692 hashes and 40 worker clocks, with identical numerical analysis. For #191 development only, three qualification snapshots preceded their adapter commits and the GTSAM qualification snapshot lacks a native binary hash. All twelve scored #191 workers match their source and all later #192 manifest checks passed.

**2026-11-23T03:51:14Z**

The default suite on corrected code 2967b07 passed 2766, with 196 slow cases deselected and four gym deprecation warnings. Dashboard wall 1236.962 s. Moved 41 overlapping precision/shape combinations in four existing parameter matrices to opt-in slow. The default retains every specialized family at float32, all constrained order, binding and schedule variants at float32, both mixture forms and both recurrence cardinalities across the two precisions. All eight fixes retain their full regression coverage. Checked the default selection is a subset of the passing run, with 2725 selected and 237 opt-in. The moved cases took 39.579 s, for an estimated default wall of 1197.383 s. Raw selection ledger and console output are under ~/agent-share/cffg-premerge-195/. The remaining merge step uses a normal merge commit to retain historical revisions.
