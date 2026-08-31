---
id: cff-0qfy
status: closed
deps: []
links: []
created: 2026-08-30T16:00:47Z
type: epic
priority: 1
assignee: coopwilliams
tags: [consolidation, rslds]
---
# Rebuild the rSLDS production stack from main at d805135

## Background in Plain English — WHY

The open rSLDS pull-request lineage contains production improvements that are
still needed, but that lineage also contains experiment setup, review records,
old ticket transitions, and generated runtime history. Merging the lineage
would import all of that ancestry and make the retained implementation hard to
review against `main` commit `d805135`.

This epic rebuilds the useful implementation as a new pull-request stack. Each
layer starts from the preceding clean layer. Existing pull requests and
branches remain unchanged.

## Outcomes / Acceptance Criteria — WHAT

1. Five linked implementation tickets port the requested sources in order:
   PR #64, PR #65, commit `30aa87e6`, the five named production commits from
   PR #92, then commits `9dbcc017` and `7405d609`.
2. Every new branch descends from `main` commit `d805135` or the immediately preceding
   new stack branch. No new branch merges the old stack ancestry.
3. The new stack retains only production code, tests, derivations, canonical
   documentation, and test-dashboard registration required by retained tests.
4. The new stack excludes old ticket-state changes, experiment
   preregistration, audit scaffolding, review notes, and old generated runtime
   metadata.
5. Each layer uses atomic logical commits, passes its focused tests, and has a
   pull request whose base is the preceding new stack branch.
6. The stack head passes one final default short-suite run after its last code
   change. That run refreshes the current dashboard and runtime metadata.
7. PR #64, PR #65, PR #92, and every pre-existing local and remote branch
   remain open or present and are not modified, closed, or force-pushed.

## Methodological Suggestions — HOW

These suggestions are optional. The implementer may choose another sound
method.

- Apply source commits without committing, discard excluded paths, resolve
  documentation against `main` commit `d805135`, and then make new atomic commits.
- Compare every completed layer with its requested source delta by path and by
  behavior rather than requiring identical commit hashes.
- Push each branch normally and open each pull request before building the next
  layer.

## Future work

No future work is identified. The five child tickets own the complete requested
consolidation.

## Definitions

- **Canonical documentation:** `README.md`, `docs/ARCHITECTURE.md`,
  `docs/LIMITATIONS.md`, and `docs/DERIVATIONS.md` when formal equations change.
- **PR stack:** Pull requests whose base branches form an ordered chain.
- **Semantic port:** A new change that preserves the source behavior while
  adapting it to a different base instead of preserving source ancestry.
- **Source delta:** The changes introduced by a pull request relative to its
  declared base, or by a commit relative to its first parent.

## Notes

**2026-08-30T16:42:42Z**

Completed the clean five-PR stack from main d805135: PR #95 (cff-qg1d), #96 (cff-0qry), #97 (cff-rl7r), #98 (cff-s6yy), and #99 (cff-4jk1). Each PR targets the preceding clean branch. Existing PRs #64, #65, #92 and all old branches were left unchanged. The final default short suite passed 1,463 tests with 39 deselected and no failures.
