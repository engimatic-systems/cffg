---
id: cff-w551
status: open
deps: [cff-x6c0]
links: []
created: 2026-11-26T19:44:49Z
type: feature
priority: 2
assignee: coopwilliams
parent: cff-xikb
tags: [studio]
---
# Find host models and open Studio through agent-share

## Background in Plain English — WHY

A remote browser cannot index the Mac's projects by itself. Cooper wants one share entry and search results that identify which local files are safe to open or import.

## Outcomes / Acceptance Criteria — WHAT

1. The share entry opens the authenticated, same-origin Studio service over Tailscale after a host restart. Agent tokens can be revoked independently and cannot read arbitrary paths.
2. A host worker discovers supported bundles and JSON files with a recognized CFFG template structure in user-selected roots, supports add/remove root and rescan, and handles moves, stale indexes and duplicates.
3. Search returns model name, path, format and diagnostics. Import snapshots external sources. Explicit save-back detects files changed since import and replaces files atomically where supported.

## Methodological Suggestions — HOW

Use the macOS launch service, a private Tailscale route, SQLite full-text search, watchfiles and a reconciliation walk. Confirm bundle and loose-template imports with the compiler service. Current/legacy input decoding applies. Numerical execution pairs are N/A.

## Future work

Use the same catalog search from the editor and agent CLI.

## Definitions

A search root is a directory the owner enables. Tailscale is the private network used by agent-share. The index is a rebuildable search database.
