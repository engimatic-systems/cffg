---
id: cff-hb8k
status: closed
deps: []
links: []
created: 2026-07-05T21:59:30Z
type: chore
priority: 1
assignee: coopwilliams
---
# Scaffold uv project + markdown-hygiene test

uv init src layout, pytest as dev dep, tests/test_hygiene.py enforcing the four-canonical-docs rule from AGENTS.md (exempting .tickets/ and .venv). Also record JAX ecosystem dependency policy in ARCHITECTURE.md.


## Notes

**2026-07-05T22:01:46Z**

Scaffolded uv src-layout project (python>=3.12, pytest dev-dep). tests/test_hygiene.py enforces the 4-canonical-docs rule (exempts .tickets/, .venv/ etc) and verified to fail on a planted rogue file. Dependency policy + M7 staging recorded in ARCHITECTURE.md. Next real work: M0 epics + M1 core IR.
