---
id: cff-6pmw
status: open
deps: []
links: [cff-d56d, cff-j4mu, cff-x0sv]
created: 2026-07-28T01:35:55Z
type: chore
priority: 3
assignee: coopwilliams
tags: [maintainability, agent-accessibility]
---
# Map cross-cutting change paths for lower-context agents

cff-x0sv needed repeated review not because the local algorithms were obscure, but because one medium feature crossed the entire IR lifecycle and duplicated runtime surfaces. A model had to discover Template declaration, validation, serialization/hash, scoping, plates, compiler resolution, arena addressing, host and JIT execution, agent cadence, checkpoint compatibility, and several test tiers. The repository has good durable detail, but it is organized by architecture layer rather than by change path; engine.py (2235 lines) and compiler.py (1411 lines) then make missing one derived surface easy for a lower-context model. This ticket improves navigation without creating another Markdown document or weakening tests.

## Design

Update docs/ARCHITECTURE.md in place with a compact change-path index for four recurring ticket shapes: new IR field, new factor/rule family, runtime numerical/status invariant, and learnable/checkpoint feature. Each row names the authoritative definition, every derived surface that must agree, the canonical focused tests, and the documentation destination. Add short module maps to compiler.py and engine.py naming their major sections and paired host/JIT implementations. Add one AGENTS.md planning sentence under Session protocol requiring a ticket handoff to name its change-path row and any duplicated execution surfaces before implementation. Do not split files merely by size; only extract a module when it creates a one-way dependency and a named contract.

## Acceptance Criteria

ARCHITECTURE has an agent-oriented change-path index covering the four feature shapes and naming concrete code/test anchors. compiler.py and engine.py have concise navigational maps that remain source-adjacent. AGENTS.md tells future tickets to identify the applicable change path and duplicated runtime surfaces. No new Markdown file, no behavior change, full suite green.
