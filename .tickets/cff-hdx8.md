---
id: cff-hdx8
status: closed
deps: []
links: []
created: 2026-07-13T01:47:39Z
type: chore
priority: 1
assignee: coopwilliams
---
# Audit Ready queue and encode missing prerequisite dependencies

Audit open tickets against their descriptions, acceptance criteria, architecture roadmap, and adversarial-review priority handoff. Add only concrete prerequisite dependency edges so tk ready exposes a focused executable queue.

## Acceptance Criteria

Missing prerequisite edges are added without cycles; epic tickets with open acceptance work are not Ready; tk ready/blocked reflect a coherent order; audit rationale is recorded.


## Notes

**2026-07-13T01:48:55Z**

Dependency audit complete. Evidence: ticket acceptance criteria, docs/ARCHITECTURE.md roadmap, and notes/adversarial-review-2026-07-10/06-ticket-priorities.txt. Added 44 edges: engine gate (3hgi -> ri8m -> d56d; 44we parallel; d56d+44we -> wbhu -> 9xmq), downstream API/storage/schedule consumers, M3 param-edge prerequisites, viz/helper prerequisites, glucose runtime/learning prerequisites, and open-child blockers on ahps/mwjo/21ta epics. Added gul8 -> cj5h so EIG builds on the MI diagnostic interface. Validation: tk dep cycle reports none; actionable Ready queue falls from 30 to 8 after this audit closes. Intentionally left 4oq8, dfpi, ey4p, 9bex, and cj5h independent because their work can proceed without the hardening chain; 44we remains parallel per its acceptance criterion.

**2026-07-13T01:49:49Z**

Correction to the preceding count: after closing this audit ticket and adding the final gul8 -> cj5h edge, tk ready contains 7 tickets (not 8).
