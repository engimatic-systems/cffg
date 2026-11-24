---
id: cao-24tn
status: open
deps: [cao-0cla, car-9ccf]
links: []
created: 2026-11-24T05:21:34Z
type: feature
priority: 2
assignee: coopwilliams
parent: car-k6oi
tags: [axiom]
---
# Run component birth and merge in compiled and batched learning loops

## Background in Plain English — WHY

A model that learns inside a compiled time loop must be able to change a slot without returning to Python. The host lifecycle in cao-0cla supplies fixed-capacity transactions and a numerical reference. Its selected-row merge uses host arithmetic.

## Outcomes / Acceptance Criteria — WHAT

1. A compiled birth and proposed merge return fixed-layout state and per-bank status, preserve all carried NIW and categorical fields, and reject invalid or stale references atomically.
2. Applying the same operation independently to many banks produces the same statuses, identities, active masks and posteriors as individual calls, including when some banks reject the proposal.
3. Qualify host versus compiled results using independent fraction-weighted examples and validated 32-, 500- and 5,000-slot cases in both precisions, including empty/full banks, near-singular covariance and sub-ULP updates. Establish and report any supported precision limit.
4. Restoring the exact standalone snapshot allows the same next compiled operations in single and batched execution. The parent ticket records the compiled performance and the remaining model-integration status.

## Methodological Suggestions — HOW

Use the Learnable or checkpoint feature and Runtime numerical or status invariant change paths. Reuse the host lifecycle metadata contract and mixed high/low arithmetic. Check eager, JIT, vmap, and exact snapshot parity; flooding/scheduled graph sweeps, IR and bundle version changes are N/A for this standalone bank.

## Future work

Parent car-k6oi owns qualification of the full fixed-capacity lifecycle before the AXIOM model tickets use it.

## Definitions

JIT compiles an operation for repeated execution. vmap applies it independently to a batch of banks. An ULP is the spacing between adjacent floating-point numbers at a given magnitude.


## Notes

**2026-11-24T05:25:21Z**

Prerequisite --> dependent (from tk dep tree --full, tk ready and tk blocked).

cff-jilz (closed) --> cao-0cla (host transactions) --+
                                                     +--> cao-24tn (compiled transactions)
cff-xfnx (working) --> car-9ccf (qualification) -----+
                               |
                               +--> car-k6oi (lifecycle parent) --> car-kwjj, car-mps6 --> car-zieb

cao-24tn is a child of the lifecycle parent. Its dependency records are on this ticket; closing the host child does not close car-k6oi or satisfy the numerical qualification.
