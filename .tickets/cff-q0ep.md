---
id: cff-q0ep
status: closed
deps: []
links: []
created: 2026-07-05T22:12:45Z
type: task
priority: 0
assignee: coopwilliams
parent: cff-br53
---
# Core IR dataclasses + factor-type registry + validation

VariableSpec/BlobRef/FactorSpec/ConstraintSpec/PortSpec/Template, extensible FACTOR_TYPES registry (categorical, matrix_cat, transition_mixture, dirichlet, equality), validate() collecting all problems: FFG degree<=2, dangling-edge-must-be-port invariant, constraint target checks.


## Notes

**2026-07-05T22:14:51Z**

Landed: ir.py (specs, registry with 5 discrete factor types, param edge-XOR-blob binding), validate.py (all-problems-at-once ModelError; port invariant, degree<=2, symmetric domains, p-sub role checks). HMM-slice fixture in conftest shared by later tiers. 16 tests.
