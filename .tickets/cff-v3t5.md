---
id: cff-v3t5
status: closed
deps: []
links: []
created: 2026-09-09T15:58:40Z
type: chore
priority: 2
assignee: coopwilliams
tags: [housekeeping, process]
---
# Remove chat archives and narrow routine context and tooling gates

## Background in Plain English — WHY

Cooper approved removing the repository chat archives and using focused validation for isolated tooling changes. He also asked for an explicit instruction change to avoid repeatedly loading unchanged text. Existing ticket audit narration must remain intact.

## Outcomes / Acceptance Criteria — WHAT

1. The six tracked files under chats/ are removed from the working tree without rewriting Git history; their removal is recoverable. Future local chat archives are ignored by default.
2. Existing ticket audit narration remains unchanged.
3. AGENTS.md tells agents to reuse current repository instructions and source text already in context, refresh missing or changed content, and read relevant sections using bounded searches.
4. Isolated tooling changes can use focused tests and changed-artifact checks, with their isolation rationale and results recorded. Changes affecting engine behavior, shared dependencies, test discovery/selection/execution, or pass/fail handling retain the full short-suite gate.
5. README and AGENTS agree on validation. No engine, test-runner, dependency or numerical code changes.
6. Archive scope, instruction consistency, ticket dependencies and patch whitespace are validated; the housekeeping PR is updated with separate logical commits.

## Methodological Suggestions — HOW

Optional: verify the tracked archive paths without loading transcript contents, remove only those paths, and retain recovery through Git. Replace the unconditional startup read with explicit refresh criteria and consolidate duplicate delegation instructions. Keep read tracking in working context, not new repository state files.

## Future work

cff-umag still owns optimizing future wording-audit output. Cooper explicitly deferred editing existing audit narration. Compiler and engine extraction remain outside this housekeeping work.

## Definitions

- Focused validation: tests and artifact checks for the affected component rather than the whole inference suite.
- Context: text currently available to the agent in the conversation.
- Isolation rationale: an explanation of why the tooling change cannot affect engine behavior or the test gate.


## Notes

**2026-09-09T15:59:19Z**

Pre-implementation handoff: continue PR112 at b014d99 on agent/housekeeping-runtime-history, refreshed from origin. Change-path index: Runtime numerical or status invariant, contributor validation policy only; no runtime authority changes. Host/JIT, flooding/scheduled, direct/batched inference and model checkpoint pairs are N/A. Review pairs: AGENTS/README testing instructions; supplied/changed/missing repository context; tracked/untracked archive scope. Verified exactly six clean tracked regular files under chats/ and no untracked or ignored contents. Validation is documentation/configuration and deletion-scope checking, not an inference run. Existing audit notes will not be edited.

**2026-09-09T16:03:00Z**

Completed on PR112 in separate logical commits. Removed only the six verified repository chat archives and added /chats/ to .gitignore; the directory is gone, while Git history and original agent logs outside the repository are untouched. Recovery is available from b014d99 before the archive-removal commit 379e037.

AGENTS now reuses current supplied instructions, refreshes changed/missing content, scopes reads to sections or functions, checks PR paths before patches, bounds tool/research output, and keeps read tracking out of the repository. Isolated tooling now defaults to focused tests plus artifact validation with a recorded isolation rationale; runtime behavior, shared dependencies and test-gate changes retain the full suite. README matches.

Validation: exact deletion scope and ignore behavior verified; existing ticket contents unchanged; AGENTS/README reviewed together; git diff --check and tk dep cycle pass. No src/tests/tools/dependency/runtime-history change since b014d99, so the preceding 1,578-test code gate is retained without another inference run. Existing audit narration and scientific reference content remain intact. No further cleanup or architecture work is started.
