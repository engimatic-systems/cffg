---
id: cff-w551
status: closed
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

## Notes

**2026-11-26T21:13:28Z**

Shipped a loopback launchd service behind private Tailscale HTTPS, plus the agent-share/cffg-studio/index.html launcher. Reinstalled, restarted and reopened the final release through the shared HTTP launcher and real TLS proxy. Unauthorized request returned 401. Owner remote browser identity is supplied by Tailscale; this tagged host was tested with its bearer. Host search indexes chosen directories, handles moves/deletes, coalesces rescan during root changes and indexes embedded project names. Stale roots are filtered before pagination; unchanged mtime with rewritten content is reindexed. On 50,000 synthetic catalogue entries, 500 warmed queries (417 distinct) had 12.9ms p95 and 38.6ms maximum. Source: discovery.py, test_studio.py and /tmp/cffg-studio-qa/search-verification.json. High confidence for the measured workload.
