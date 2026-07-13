---
id: cff-xnxj
status: in_progress
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

