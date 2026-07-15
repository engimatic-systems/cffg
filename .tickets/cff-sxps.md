---
id: cff-sxps
status: closed
deps: []
links: []
created: 2026-07-15T04:06:42Z
type: chore
priority: 1
assignee: coopwilliams
tags: [workflow, git]
---
# Session bootstrap: refresh base before branching

Prevent ticket branches from starting on stale local base history. Put the safe base-update commands near the top of AGENTS.md and explain that upstream changes/conflicts should surface before implementation.

## Acceptance Criteria

AGENTS.md tells agents to switch to the intended base and run git pull --ff-only origin <base-branch> before creating a work branch; the rationale names stale-base and late-rebase surprises.


## Notes

**2026-07-15T04:07:25Z**

Implemented the top-level Session bootstrap: choose BASE_BRANCH, switch to it, then git pull --ff-only origin before creating the work branch. The start-of-session checklist links to the bootstrap. This makes stale upstream history/conflicts visible before implementation. Docs-only change; git diff --check passes.

**2026-07-15T04:07:49Z**

PR: https://github.com/engimatic-systems/cffg/pull/31
