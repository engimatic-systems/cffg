---
id: cff-j72j
status: open
deps: []
links: []
created: 2026-08-18T16:09:13Z
type: bug
priority: 2
assignee: coopwilliams
tags: [tooling, tickets]
---
# Restrict tk status updates to YAML frontmatter

## Background in Plain English

Closing cff-x5z2 correctly changed its YAML `status` field to `closed`, but it
also replaced a prose line in the Markdown body that began with `status:`. The
body originally explained an experimental runtime status and had to be restored
manually. Ticket commands must not rewrite ticket history or technical notes.

## Outcomes / Acceptance Criteria

1. `tk start`, `tk status`, `tk close`, and `tk reopen` change only the
   frontmatter `status` value of the selected ticket.
2. A regression fixture places `status:` at the beginning of multiple body
   lines and proves every byte outside the intended frontmatter value remains
   unchanged after each status command.
3. The commands reject missing, duplicate, malformed, or unsupported
   frontmatter status fields instead of searching farther into the Markdown
   body.
4. Existing ticket dependency and listing behavior remains unchanged.

## Methodological Suggestions

These suggestions are optional. The implementer may choose another sound method
that satisfies the outcomes.

- Parse the bounded YAML frontmatter before updating it rather than applying a
  file-wide text replacement.
- Compare the complete before/after ticket in the regression, masking only the
  one expected scalar value.

## Future work

This ticket covers status mutation only. Other `tk` commands should receive
separate tickets if they are shown to rewrite body text.

## Definitions

- **Frontmatter:** The YAML metadata block between the opening and closing
  `---` lines at the top of a ticket.

## Notes

Observed while closing cff-x5z2 on 2026-08-18. The YAML field and a later prose
line beginning with `status:` both became `status: closed`; the prose was
restored before the cff-x5z2 closure commit. No CFFG runtime surface is affected.
