---
id: cff-umag
status: open
deps: []
links: []
created: 2026-09-09T03:57:33Z
type: task
priority: 2
assignee: coopwilliams
tags: [tooling, context]
---
# Reduce wiggle-audit output and token use

## Background in Plain English — WHY

The ticket wording audit repeats candidate definitions and occurrence details,
producing far more text than an agent needs to review a ticket. Cooper asked
us to stop requiring it and optimize its token use before reconsidering the
workflow. The tool remains available for this improvement work.

## Outcomes / Acceptance Criteria — WHAT

1. The default review output groups repeated matches and omits repeated
   definitions while retaining ticket IDs, line references, and enough context
   to assess each potentially ambiguous claim.
2. Full occurrence details are available through an explicit detail option.
3. A comparison on a short ticket and a long ticket records the command,
   tokenizer/version, and output-token count before and after the change;
   the default output uses at least 75% fewer tokens on both fixtures.
4. Compact output states when it omits or truncates results and explains how
   to retrieve those details. Existing machine-readable modes retain their
   documented contract or receive an explicitly documented version change.
5. Repository instructions no longer require running the audit. Optimizing
   the tool does not reinstate that requirement.

## Methodological Suggestions — HOW

These suggestions are optional; another method may satisfy the outcomes.

- Start with the scan renderer in
  tools/wiggle_words/query_ticket_wiggle_audit.py.
- Use line-grouped matches, one summary, and opt-in candidate definitions.
- Select representative fixtures before measuring output reduction.

## Future work

No further work is identified. Reintroducing a mandatory audit would require
a separate decision by Cooper.

## Definitions

- **Token:** A unit of text consumed by a language model.
- **Tokenizer:** The algorithm used to count those units.
- **Candidate:** A word or phrase the tool flags for contextual review.

## Notes

**2026-09-09T03:57:48Z**

Removed the mandatory scan/slice/rerun instructions from AGENTS.md at Cooper’s request. Historical review evidence and the tool remain intact. Output optimization is still open; no audit commands ran for this change. Validation: tk show, tk dep cycle, and git diff --check.
