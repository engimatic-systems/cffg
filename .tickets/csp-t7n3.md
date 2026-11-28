---
id: csp-t7n3
status: in_progress
deps: []
links: []
created: 2026-11-28T06:34:04Z
type: task
priority: 2
assignee: coopwilliams
parent: cff-g7rc
---
# Compile identical mixed-schedule propagation passes only once

A mixed model can use the same ordered message updates twice, once to prepare its initial state and once to reconstruct its forest of ordinary factors. The T-maze and D1 cold traces show that JAX compiles a fresh copy for each use. A model containing an ordinary cycle can require different passes.

## Design

Use the ScheduleProgram owner in the runtime change-path row. Both host and compiled trust-region controllers consume this owner. Compare the entire immutable stage sequence before sharing the compiled function. Flood, colored, exact two-pass, numeric kernels, batch transport and serialization require no changes.

## Acceptance Criteria

The equal-stage case uses one compiled callable and the unequal-stage case retains both behaviors. Compare inputs, native outcomes, cold compile counts and repeated timings on the two retained cases. Report warm costs separately. The broader solver target stays with cff-g7rc.

