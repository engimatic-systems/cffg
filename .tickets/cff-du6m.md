---
id: cff-du6m
status: open
deps: []
links: []
created: 2026-07-13T03:51:58Z
type: task
priority: 2
assignee: coopwilliams
parent: cff-cynx
---
# Structure-learning literature pass: AXIOM/RGM growth-merge criteria + BMR closed forms

Sub-agent research per AGENTS.md protocol, with SPECIFIC questions: (1) AXIOM (arXiv:2505.24784): what is the exact criterion for expanding a mixture component online (threshold on what quantity?), and what is the exact BMR-based merge/prune test and its schedule? (2) Supervised structure learning (arXiv:2311.10300): what is the model-selection prior over expansions and how does data ordering enter the evidence accounting? (3) BMR (arXiv:1805.07092): write out the reduced-evidence formulas specialized to our conjugate families — Dirichlet columns (prune a state / merge two states) and NIW/MNIW (prune an edge/regime) — ready to implement against cffg.conj. Distil answers into this ticket as question -> answer -> source -> confidence; durable formulas go to docstrings when implemented.

## Acceptance Criteria

Ticket note contains: the AXIOM expansion criterion and merge test in equations; the 2311.10300 selection prior; BMR reduced-evidence formulas for Dirichlet and NIW/MNIW specialized to cffg's parameterizations, each with a hand-checkable toy example.

