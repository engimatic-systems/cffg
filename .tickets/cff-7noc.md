---
id: cff-7noc
status: open
deps: [cff-x6c0]
links: []
created: 2026-11-26T19:44:48Z
type: feature
priority: 2
assignee: coopwilliams
parent: cff-xikb
tags: [studio]
---
# Exchange anchored critiques and review agent edits in Studio

## Background in Plain English — WHY

A coding agent needs to open the same revision as Cooper, explain a concern about a particular wire, and offer an edit that Cooper can inspect or accept.

## Outcomes / Acceptance Criteria — WHAT

1. A documented agent command-line client reads snapshots and diffs, previews and applies changes, follows revisions, and posts a structured critique or a proposed fix.
2. A reviewer can reply, resolve, accept or decline a revision-anchored proposal. A stale fix displays the conflicting entities and requires renewed validation at the current revision.
3. Accepted changes and resolutions appear for both clients after reconnection. Concurrent edits, retries, renamed anchors, and deleted targets have explicit outcomes.

## Methodological Suggestions — HOW

Build on the library API, capability schema, stable element identifiers and durable event cursors. Exercise two independent clients. Engine execution and checkpoint pairs are N/A.

## Future work

The keyboard editor consumes the same review protocol and the simulator can link a failing run to a critique.

## Definitions

An anchor identifies an element and the revision under discussion. A cursor is the last event a client received.
