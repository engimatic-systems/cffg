---
id: cff-xnxj
status: closed
deps: []
links: []
created: 2026-07-13T01:06:13Z
type: feature
priority: 2
assignee: coopwilliams
tags: [dashboard, ui]
---
# Live ticket dashboard: priority sorting, labels, and tag filtering

Extend ~/agent-share/ticket_dashboard.py so the generated live ticket dashboard exposes priority in each card, sorts tickets by priority, and filters tickets by tag.

## Acceptance Criteria

Each ticket card shows a visible accessible P0-P4 priority label. A control sorts top-level tickets and children by priority in either direction while retaining the hierarchy. A tag control lists all tags plus untagged and filters matching cards, including matching child tickets when the parent does not match. Existing status and text search filters continue to work. The generated dashboard renders successfully.


## Notes

**2026-07-13T01:18:24Z**

Landed. Live dashboard script committed in agent-share as d9c6ad7: accessible P0-P4 labels, status/priority sort selector with P0-first and P4-first modes, tag/untagged filters, and child-aware filtering. Repo regression test + EPIC_FILES registration + ticket committed as 005d55e. Verification: uv run pytest -q = 1116 passed; embedded JS syntax and generated HTML parse checks pass; uv run python tools/dashboard.py refreshed ~/agent-share/cffg-tests.html with 1116 tests, 0 failing. No follow-up required.
