---
id: cff-x6c0
status: closed
deps: [cff-0266]
links: [cff-1xig, cff-gnpn]
created: 2026-11-26T19:44:48Z
type: feature
priority: 2
assignee: coopwilliams
parent: cff-xikb
tags: [studio]
---
# Store and publish compile-checked model revisions through a shared API

## Background in Plain English — WHY

A GUI and coding agents must be able to finish an edit in several gestures while other readers continue to see a complete working graph. The existing edit actions validate structure. Compilation also resolves arrays, learning declarations and supported rules.

## Outcomes / Acceptance Criteria — WHAT

1. A private model library stores a named blank starter and a current rSLDS with complete learning state. A fresh checkout can reconstruct each current-format model.
2. A request can preview or apply an atomic batch against a base revision. Publishing validates the exact staged contents, round-trips a portable bundle where supported and compiles against the declared engine environment.
3. Invalid graphs, missing arrays, stale revisions, a crash before publication and a retry each preserve one coherent accepted history.
4. A versioned HTTP interface exposes the model, change, diagnostic, capability and revision schemas to both clients. Saving or exporting the current graph preserves learning state.

## Methodological Suggestions — HOW

Start at the architecture index row New IR field for persistence and action consumers. Reuse apply_edits, compile_template, save_bundle and load_bundle. Use Git conditional ref updates and an idempotent request identifier. Review current/legacy decode for imports. Numerical host/JIT, flooding/scheduled and single/vmap are N/A for the storage service.

## Future work

The Studio editor, collaboration, search and experiment tickets consume this API.

## Definitions

An API is the programmatic interface. A bundle is a portable .cffgz archive. A base revision identifies the exact model an edit started from.

## Notes

**2026-11-26T21:13:28Z**

Landed the private bare Git library with compiler-checked atomic publication, conditional revisions, retry receipts and portable native/Studio exports. HTTP clients independently exercised preview, invalid shape, stale revision, retry and export/import of all four rSLDS learner priors and posteriors. Source: tests/test_studio.py, src/cffg/studio/library.py, and the real two-client acceptance run. Concurrent writers observed one 200 and one 409. Large-array Git LFS checkout, hydration and export were exercised separately. The central repository is on the Studio host at ~/projects/cffg-models. New writes go through /v1/models. High confidence.
