---
id: cff-5d1z
status: closed
deps: []
links: []
created: 2026-08-13T04:09:35Z
type: chore
priority: 2
assignee: coopwilliams
tags: [workflow, tickets]
---
# Keep wiggle-word audits out of ticket handoffs

## Background in Plain English

`AGENTS.md` requires authors to scan new or materially rewritten tickets for
words that can hide an unclear owner, time, layer, scope, comparison, or
evidence claim. That review is useful. A second instruction requires the
handoff to list every retained candidate and its intended meaning. Agents have
responded by appending long glossaries that repeat clear ticket prose and make
the implementation handoff harder to find.

This ticket removes only the glossary-reporting requirement. Authors still run
the scan, review every match in context, clarify unclear claims, rerun the scan,
and check capability ownership separately.

## Outcomes / Acceptance Criteria

1. `AGENTS.md` no longer instructs an author to report every retained scan
   candidate and intended meaning in a ticket handoff or note.
2. `AGENTS.md` continues to require the ticket scan after a ticket is created or
   materially rewritten.
3. `AGENTS.md` continues to require contextual review and clarification of
   claims with more than one reading.
4. `AGENTS.md` continues to require a scan after refinement and a separate
   capability-ownership check.
5. The historical `slice` command remains available for investigating an older
   classification when it is relevant.
6. The three exhaustive classification paragraphs added while authoring
   `cff-cfqg`, `cff-dqk2`, and `cff-11jj` are absent. Their useful dependency,
   change-path, ownership, and duplicated-surface handoffs remain.
7. `git diff --check` passes.

## Methodological Suggestions

The requested change is one deletion from the ticket-authoring rule. An
implementer may choose another wording only if it does not recreate a reporting
requirement for retained scan matches.

## Future work

No future work is identified. This ticket does not change the scanner,
lexicon, audit schema, or ticket-writing requirements outside retained-match
reporting.

## Definitions

- **Capability owner:** The ticket or component responsible for delivering a
  named behavior.
- **Candidate:** A word or phrase flagged by the mechanical scan because it can
  carry more than one meaning.
- **Handoff:** The concise record that tells the next contributor what changed,
  what remains, and which boundary matters.
- **Wiggle-word audit:** The repository's mechanical scan for wording that may
  hide an ambiguous claim.

## Notes

**2026-08-13T04:11:00Z**

Completed 2026-08-12: removed only the retained-match reporting clause from AGENTS.md; the scan, contextual review, refinement, ownership check, and historical slice guidance remain. Removed the three uncommitted exhaustive audit glossaries while preserving their dependency and change-path handoffs. Ticket/document-only change; no pytest run required.
