---
id: cdc-b65n
status: closed
deps: []
links: []
created: 2026-09-14T02:37:14Z
type: chore
priority: 2
assignee: coopwilliams
tags: [process]
---
# Record the pull request writing rule

## Background in Plain English — WHY

Cooper asks that pull request descriptions explain this implementation stack at an undergraduate level. The rule is already present as an uncommitted edit in the original worktree and must be included in the supervised work.

## Outcomes / Acceptance Criteria — WHAT

1. AGENTS.md requires pull request bodies to use simpler explanations than the technical documentation.
2. The rule is committed without changing its meaning.

## Methodological Suggestions — HOW

Copy the authorized two-line rule into the isolated implementation worktree. Documentation-only change; runtime execution pairs are N/A. Check the diff and ticket dependency graph.

## Future work

Apply the rule to pull requests for cff-ffsg and its D12 implementation and qualification descendants.

## Definitions

- Pull request: a proposed set of repository changes submitted for review.
- D12: the documented hybrid inference approximation being implemented.


## Notes

**2026-09-14T02:37:31Z**

Copied Cooper’s authorized wording exactly. The original and isolated worktree AGENTS.md files compare equal. Documentation-only change: architecture and all numerical/execution surfaces are N/A. Validation: cmp, git diff --check, and tk dep cycle; no test-suite run is required. No remaining work.
