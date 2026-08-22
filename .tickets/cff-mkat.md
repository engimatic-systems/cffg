---
id: cff-mkat
status: closed
deps: []
links: []
created: 2026-08-22T16:13:50Z
type: chore
priority: 1
assignee: coopwilliams
tags: [hygiene, rslds, documentation]
---
# Sanction the user-requested rSLDS review-note directory

## Background in Plain English — WHY

Cooper explicitly requested parallel Codex and independent-reviewer notes plus
a synthesis under `notes/rslds-debug/`. Those six tracked Markdown files now
serve as the preserved audit record for the rSLDS repair decision. The generic
canonical-document hygiene test rejects them because the policy and test were
not amended when the requested directory was added.

Deleting or relocating the audit would contradict the requested review
workflow. Broadly exempting arbitrary Markdown would defeat the document-sprawl
guard. The repository needs one path-specific, documented exception.

## Outcomes / Acceptance Criteria — WHAT

1. `AGENTS.md` names `notes/rslds-debug/` as the user-sanctioned, closed-scope
   multi-review audit directory and distinguishes it from project/session state.
2. The hygiene test exempts Markdown only below that exact repository-relative
   root; similarly named directories elsewhere remain prohibited.
3. The six existing audit files pass the hygiene gate while a synthetic
   Markdown file outside canonical documents and sanctioned roots still fails.
4. No review content, canonical document, ticket record, or other hygiene rule
   is removed or weakened.

## Methodological Suggestions — HOW

- Represent the exception as an exact relative root rather than adding the
  basename to the broad ignored-directory set.
- Keep the test helper pure so hostile paths can be checked without writing a
  temporary Markdown file into the repository.

## Future work

New review projects still belong in tickets unless Cooper explicitly sanctions
another durable audit directory. This ticket does not authorize general notes
or session-state Markdown.

## Notes

Architecture change paths are structurally irrelevant. Applicable surfaces are
the collaboration policy in `AGENTS.md` and its executable hygiene test.

Hostile-input decisions: exact path-prefix matching is required; a sibling such
as `notes/rslds-debug-copy/`, a nested directory elsewhere named `rslds-debug`,
and ordinary root/docs Markdown remain prohibited.

**2026-08-22T16:14:38Z**

Implemented the exact notes/rslds-debug path exception in AGENTS.md and tests/test_hygiene.py. The helper exempts the user-requested audit tree while rejecting a sibling path, the same basename elsewhere, and unregistered docs Markdown. All four hygiene tests pass; existing review files are unchanged.
