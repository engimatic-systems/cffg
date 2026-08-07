---
id: cff-twzc
status: in_progress
deps: []
links: []
created: 2026-08-07T15:20:31Z
type: chore
priority: 1
assignee: coopwilliams
tags: [process, tickets, ambiguity]
---
# Ticket semantics: audit wiggle words and require explicit capability ownership

## Plain English

Make ticket capability claims mechanically reviewable instead of relying on overloaded prose. Preserve a reproducible two-pass vocabulary audit, then require future tickets to state capability ownership, implementation layer, and evidence boundaries explicitly. Apply the resulting structure to the remaining roadmap tickets before implementation resumes.

## Scope

The audit is pinned to the cff-y4nx branch corpus at c855af6 so v1 and v2 remain comparable. The refinement covers every open or in-progress ticket; closed tickets remain historical evidence except where an active ticket depends on interpreting one. This ticket owns ticket-language policy and ticket rewrites, not rSLDS implementation.

## Design

Applicable architecture change path: documentation/ticket process; runtime, compiler, serialization, host/JIT, scheduling, and checkpoint surfaces are N/A because no executable model behavior changes.

Each remaining ticket gets a capability-ownership ledger. Every referenced capability is classified as delivered prerequisite, owned here, delegated future, external, or unowned; non-current owners name a ticket or authority, and ticket relations agree with tk dependency direction. Reserved claim words such as end-to-end, generic, learning, implemented, validation, performance, and N/A receive the structured boundary required by their measured ambiguity.

Numerical hostile-input cases are structurally N/A. Process-hostile cases are missing owners, list-level N/A dispositions, unnamed comparison universes, oracle-assisted validation, mixed cold/warm performance claims, and a title whose scope exceeds its acceptance evidence.

## Acceptance Criteria

1. V1 and v2 candidate lexicons and occurrence audits parse as YAML, retain exact corpus/lexicon hashes, and reconcile every raw match.
2. AGENTS.md defines required capability ownership, claim-layer, N/A-disposition, and validation-boundary fields without relying on prose keywords alone.
3. Every open or in-progress ticket is reviewed against the structure; applicable tickets are updated in place, and every deferred capability has a named accountable ticket or is explicitly unowned.
4. cff-y4nx is narrowed or expanded unambiguously, depends on this process ticket, and distinguishes fixed-parameter oracle inference from observed-only learned fitting.
5. tk dependency records agree with the ownership ledgers; tk dep cycle and git diff --check pass. This ticket-only change does not run pytest.

## Notes

**2026-08-07T15:22:31Z**

Handoff 2026-08-07. Draft PR #57 on agent/cff-twzc-ticket-semantics, based on agent/cff-4apw-pg-learning. V1 and v2 lexical/context audits are committed at ba672d7 and compare the identical cff-y4nx corpus snapshot c855af6: v1 65 candidates/3,651 occurrences/28 ambiguous/238 unlisted/14 false positives; v2 87 candidates/4,300 occurrences/7 ambiguous/5 unlisted/10 false positives. The seven survivors are structural rather than missing vocabulary. Next: add the mandatory ownership/claim-layer/validation/N/A schema to AGENTS.md, review all open/in-progress tickets, and rewrite cff-y4nx before rebasing PR #56. Preserve PR #56 implementation and experiment evidence; do not restart from scratch. Gotcha: the audit corpus is intentionally pinned to the child commit c855af6 even though this process PR is upstream, so do not regenerate reports from the upstream working tree. Existing unrelated local AGENTS.md edits remain unstaged.
