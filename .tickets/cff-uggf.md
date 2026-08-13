---
id: cff-uggf
status: closed
deps: []
links: []
created: 2026-08-08T19:28:14Z
type: bug
priority: 2
assignee: coopwilliams
tags: [dashboard, tests]
---
# Keep ticket-dashboard fixtures aligned with parser output

## Plain English

The repository test renders synthetic tickets through the live dashboard in
`~/agent-share`. The dashboard parser now supplies a `body_sections` field,
but the test duplicated an older ticket dictionary without that field. This
repair builds each synthetic ticket body through the dashboard parser so the
renderer and its fixture share one schema.
`tests/test_ticket_dashboard.py` owns the synthetic ticket construction used
by this regression; the live script continues to own parsing and rendering.

## Outcomes / Acceptance Criteria

1. `tests/test_ticket_dashboard.py` obtains rendered-body fields from
   `ticket_dashboard.parse_body`.
2. The regression test continues to assert priority labels, priority sorting,
   tag filtering, and child-aware filtering.
3. `uv run pytest -q tests/test_ticket_dashboard.py` passes against the live
   dashboard script.
4. The repair does not modify the live dashboard or its generated HTML.

## Methodological Suggestions

- Give the loaded dashboard module to the fixture helper and parse a minimal
  Markdown title through its module-level parser.
- Keep frontmatter fields in the fixture; `parse_body` returns the Markdown
  body fields.

## Future work

No work is deferred by this fixture repair.

## Definitions

- **Live dashboard:** The ticket renderer at
  `~/agent-share/ticket_dashboard.py` exercised by this repository's
  regression test.
- **Parser output:** The dictionary returned by the dashboard's `parse_body`
  function and consumed by its renderer.

## Notes

**2026-08-08T19:36:40Z**

The live dashboard added richer body sections after the previous green suite, while this regression still hand-built the older renderer dictionary. Commit 614bf18 now derives the Markdown-body fixture fields from parse_body and keeps frontmatter fields in the fixture. Focused dashboard plus rSLDS checks: 14 passed. Final repository gate: 1,291 passed, 38 slow tests deselected, zero failures. Retained scan meanings are close for administrative ticket status, check for automated execution, pass for successful automated execution, test for an automated check or test artifact, explicit owner for the two components named in Plain English, and deferred for the explicit statement that this repair leaves no future work.
