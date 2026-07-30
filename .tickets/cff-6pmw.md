---
id: cff-6pmw
status: closed
deps: []
links: [cff-d56d, cff-j4mu, cff-x0sv]
created: 2026-07-28T01:35:55Z
type: chore
priority: 3
assignee: coopwilliams
tags: [maintainability, agent-accessibility]
---
# Map cross-cutting change paths for contributors

## Plain English

Future contributors can follow a compact map from each common kind of change to
every code, test, and documentation surface that must stay aligned. Large engine
and compiler modules gain source-adjacent section guides, and ticket planning will
name duplicated execution paths before implementation begins. This reduces
repeated repository scans, context use, and review effort for both agents and
people.

cff-x0sv needed repeated review not because the local algorithms were obscure, but because one medium feature crossed the entire IR lifecycle and duplicated runtime surfaces. A model had to discover Template declaration, validation, serialization/hash, scoping, plates, compiler resolution, arena addressing, host and JIT execution, agent cadence, checkpoint compatibility, and several test tiers. The repository has good durable detail, but it is organized by architecture layer rather than by change path; engine.py (2235 lines) and compiler.py (1411 lines) then make missing one derived surface easy for a lower-context model. Repeated broad searches also consume context that could be spent on the change itself. This ticket improves navigation for human and agent contributors without creating another Markdown document or weakening tests.

## Design

Update docs/ARCHITECTURE.md in place with a compact change-path index for four recurring ticket shapes: new IR field, new factor/rule family, runtime numerical/status invariant, and learnable/checkpoint feature. Each row names the authoritative definition, every derived surface that must agree, the canonical focused tests, and the documentation destination. Add short module maps to compiler.py and engine.py naming their major sections and paired host/JIT implementations. Add one AGENTS.md planning sentence under Session protocol requiring a ticket handoff to name its change-path row and any duplicated execution surfaces before implementation. Do not split files merely by size; only extract a module when it creates a one-way dependency and a named contract.

## Acceptance Criteria

ARCHITECTURE has a contributor-oriented change-path index covering the four feature shapes and naming concrete code/test anchors. compiler.py and engine.py have concise navigational maps that remain source-adjacent. AGENTS.md tells future tickets to identify the applicable change path and duplicated runtime surfaces. No new Markdown file, no behavior change, full suite green.

## Notes

**2026-07-30T14:58:35Z**

Final handoff (2026-07-30): added the contributor change-path index for IR fields, factor/rule families, numerical/status invariants, and learnable/checkpoint features; each row names authorities, derived surfaces, focused tests, and canonical documentation. Added compact compiler and engine module maps, including host/JIT and flooding/scheduled pairs, and made future ticket handoffs name checked duplicate surfaces. Corrected the stale cffg.compile/cffg.infer module names. This meta-documentation ticket has no applicable implementation row; all four rows are its deliverable, and executable paths are documented but unchanged. Validation: py_compile passed for compiler.py and engine.py; hygiene/ticket dashboard tests 4 passed; full dashboard suite 1281 passed, 0 failed, with 4 third-party deprecation warnings; tk dep cycle and git diff --check passed. Next ready roadmap work remains cff-l6i5. No implementation gotchas.
