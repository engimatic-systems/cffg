---
id: cff-c6rh
status: closed
deps: []
links: []
created: 2026-07-06T02:45:04Z
type: task
priority: 1
assignee: coopwilliams
parent: cff-xjlh
---
# Edit-action hardening: JSON codec + dry-run check

GUI contract: edit bundles serialize to/from JSON (action registry + spec codecs), ReplaceFactor action, check_edits(t, actions) -> problem list without raising (dry-run for live validation in an editor).


## Notes

**2026-07-06T02:49:57Z**

Landed: JSON codec for all 11 action types (bundle_to/from_json, value-equal round-trip), ReplaceFactor (one-action rewiring), check_edits dry-run returning problem list without raising (GUI live-validation hook).
