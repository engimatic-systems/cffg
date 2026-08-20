---
id: ccn-i59e
status: closed
deps: []
links: []
created: 2026-08-20T16:50:00Z
type: chore
priority: 2
assignee: coopwilliams
parent: cff-nkbj
tags: [rslds, diagnostics, documentation]
---
# Set up independent rSLDS diagnostic review workspace

## Background in Plain English — WHY

The final rSLDS experiment exposed two unresolved areas: the external reference adapter did not mix, and the heterogeneous CFFG fits did not reliably converge or recover source-specific transition functions. Cooper wants two models to investigate these areas independently and compare their conclusions only after both reviews finish. The repository needs a shared brief and separate writing areas so the reviews use the same questions without influencing each other.

## Outcomes / Acceptance Criteria — WHAT

1. The notes directory contains one shared rSLDS diagnostic brief that names the existing tickets, evidence boundaries, and comparison procedure.
2. Codex and an independent reviewer have separate folders with matching report templates.
3. The shared brief keeps adapter diagnosis separate from heterogeneous coverage, convergence, and recovery diagnosis.
4. The completed final artifact and its final seeds remain immutable and are not used to tune a repair.

## Methodological Suggestions — HOW

- Give both reviewers the same diagnostic questions and reporting headings.
- Keep draft conclusions isolated until both reviewers mark their reports ready for comparison.
- Record evidence, competing explanations, recommended next tests, and confidence.

## Future work

The reviews will inform cff-l5ag, cff-q1au, cff-qulu, and cff-0oqo. A later comparison will reconcile the independent findings. This setup ticket does not diagnose or fix those defects.

## Definitions

- **Independent review:** A diagnosis written without reading the other reviewer’s conclusions.
- **Reference adapter:** The external paper-equation rSLDS implementation used as the comparison arm.
- **Source-specific recovery:** Learning different transition-probability functions for different source regimes.
