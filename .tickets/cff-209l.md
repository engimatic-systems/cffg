---
id: cff-209l
status: open
deps: [cff-ri8m, cff-l6i5]
links: [cff-1o72, cff-o7hs, cff-l6i5]
created: 2026-07-30T16:09:57Z
type: chore
priority: 2
assignee: coopwilliams
parent: cff-ahps
---
# Bind discrete and Gaussian arenas to the FactorKernel execution contract

## Plain English

Discrete and Gaussian factors will dispatch through the executable FactorKernel registry while retaining their dedicated arena layouts. This removes the remaining central type branches so future same-arena factors can register without compiler or engine surgery.

## Technical scope

Generalize FactorKernel beyond the scalar-EF arena, then migrate the built-in discrete and Gaussian factor plans incrementally. Preserve serialized factor identities, arena layouts, rule signatures, schedules, readouts, ledgers, and numerical results. Hybrid multi-arena factors remain a separately checked boundary unless the generalized contract covers them without widening this ticket.

## Change-path handoff

Applicable row: **New factor or rule family**. Check compiler grouping/packing, Arena plans, flooding and scheduled dispatch, BFE, readouts, serialization manifests, estimator, and public exports. Explicitly audit host/JIT, flooding/scheduled, single-instance/vmap, and current/legacy checkpoint pairs.

## Hostile-input plan

Preserve exact categorical zero support, vacuous and improper Gaussian messages, float32 dtype-scaled guards, and uncomfortable cardinality/dimension behavior. Registration declarations must name each outcome and existing hostile tests must remain unchanged.

## Acceptance Criteria

A representative test-only discrete factor and Gaussian factor register and run without central compiler/engine type branches; built-ins retain bit-comparable HMM/Kalman results and unchanged structure hashes; the full suite passes without new dependencies.
