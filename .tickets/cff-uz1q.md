---
id: cff-uz1q
status: closed
deps: []
links: []
created: 2026-08-08T14:35:13Z
type: chore
priority: 2
assignee: coopwilliams
tags: [tooling, tickets]
---
# Group wiggle-word tooling in one directory

## Background in Plain English

The repository keeps the wiggle-word query program, schema, lexicons, and audits as six related files at the top of `tools/`. This makes the general tools directory harder to scan and separates one tool family from its supporting data only by filename convention. This ticket places that complete tool family in `tools/wiggle_words/` and updates every repository reference so agents and validation commands continue to use it from one clear location.

This ticket owns only the file organization and the path changes required by that move. The historical ticket corpus, audit classifications, and candidate meanings do not change.

## Outcomes / Acceptance Criteria

1. `tools/wiggle_words/` contains the query program, schema, both candidate lexicons, and both contextual audit files.
2. No wiggle-word tool-family file remains directly under `tools/`.
3. The query program locates its default audit, linked lexicon, and repository ticket files from its new directory.
4. Every tracked repository reference names the new file locations.
5. Provenance hashes affected by path-reference changes match the referenced file contents.
6. The internal audit validator succeeds.
7. The version 2 audit satisfies its JSON Schema at the new paths.
8. Bounded slice and current-ticket scan commands succeed from the repository root.
9. `tk dep cycle` and `git diff --check` succeed.

## Methodological Suggestions

- Move the six related files with Git-aware renames so history remains easy to follow.
- Define the repository root separately from the tool-family directory in the query program.
- Recompute only provenance hashes whose referenced file content changes because an embedded path changed.
- The implementer may choose another sound method that satisfies every outcome.

## Future work

No future work is identified. The dependency-aware ticket rewrite scheduler remains a separate repository tool because it schedules ticket work rather than querying or defining the wiggle-word audit.

## Definitions

- **Candidate lexicon:** A YAML file that defines words or patterns whose intended meaning an agent reviews in ticket context.
- **Contextual audit:** A YAML report that records how candidate occurrences were interpreted in a fixed historical ticket corpus.
- **Provenance hash:** A SHA-256 digest that identifies the exact contents of an input file used to produce another artifact.
- **Tool family:** A program and the data and schema files that directly support that program.

## Notes

**2026-08-08T14:46:17Z**

Handoff 2026-08-08. Moved the query program, version 2 schema, version 1 and version 2 lexicons, and version 1 and version 2 audits into tools/wiggle_words/. Updated AGENTS.md, schema navigation commands, embedded provenance paths, repository-root resolution, and the affected SHA-256 provenance values. The internal audit validator reconciled 87 candidates, 4,300 occurrences, and 1,907 candidate-ticket pairs. The version 2 audit satisfied its JSON Schema through uvx check-jsonschema. Python compilation, a bounded historical slice, a bounded current-ticket scan, uv lock --check, tk dep cycle, and git diff --check succeeded. The final short suite recorded 1,289 successful tests, 38 deselections, and zero failures in 358.03 seconds, then refreshed the dashboard. Retained scan candidates and intended meanings before this note: fixed = pinned historical corpus; support = direct relationship between a program and its data; complete = the named set of six artifacts; remaining = location or state after the move; check = a named validation command; exact = byte contents selected by SHA-256; match = digest equality; sound = a method that satisfies the numbered outcomes; validate = automated artifact validation; intended = the contextual meaning selected by a sentence; may = implementation permission; explicit_owner = this ticket owns the file move and path updates; every = each member of a named finite set; only = a ticket-scope or selective-recomputation boundary; current = the current-ticket scan command. Capability ownership was checked separately. No work is delegated. Intelligible-ticket review: complete.

**2026-08-08T14:46:46Z**

Post-note scan additions: implement = permission for the method author; test = automated execution in the named short suite. Both uses are bounded by the surrounding sentence.

**2026-08-13T05:00:13Z**

PR review correction (2026-08-12): removed both generated contextual audit snapshots from version control and ignored their canonical local paths. The checked-in query tool now scans current tickets directly from the compact version 2 lexicon, so the documented authoring check works in a clean checkout; historical slice, summary, and validation modes continue to accept a locally retained audit through --audit. The focused live-dashboard regression passed after moving its already-existing parser-derived fixture commit onto this prerequisite branch. Final default suite at code commit 1f808d5: 1,289 passed, 38 deselected, zero failures in 360.06 seconds. The current-ticket scan and dependency-cycle check passed.
