---
id: cao-0cla
status: closed
deps: [cff-jilz]
links: []
created: 2026-11-24T05:18:13Z
type: feature
priority: 2
assignee: coopwilliams
parent: car-k6oi
tags: [axiom]
---
# Activate, merge and restore component slots in host transactions

## Background in Plain English — WHY

The mixed learner can accumulate evidence in a fixed set of rows. A model also needs to activate an unused row, identify it after a restart and retire it after a merge. A row number alone can point to the wrong component after reuse.

## Outcomes / Acceptance Criteria — WHAT

1. A host transaction creates or restores an empty or populated bank, gives each active row a durable logical identifier and generation, and chooses the lowest free slot. A full bank or stale reference leaves all state intact.
2. A caller can inspect a proposed merge and keep the original or the proposal. Transfer the donor's learned Gaussian, mixing and categorical evidence once, preserve every supported high and low field, and reset the donor to its immutable prior. Reject an improper candidate atomically.
3. An exact snapshot binds ordering, identifiers, generations, reset priors and posteriors. After restoration, the next merge and birth produce the same state. Reject missing or inconsistent metadata and aliased inactive identities.
4. Check unequal fractional masses, translated means, separate within-component spread, missing categorical evidence and donor reuse in float32 and float64. Exercise bank capacities 32, 500 and 5,000 with a fixed layout.

## Methodological Suggestions — HOW

The Learnable or checkpoint feature change path covers the standalone exact codec. Reuse the mixed learner's schema and validation for its payload. An independent fractional arithmetic calculation can audit the production merge. Eager host transactions are in scope. Engine schedules, template IR fields, bundle migration and batch-of-banks execution are N/A for this initial host operation.

## Future work

Parent car-k6oi owns integration after qualification of cff-xfnx and car-9ccf, and the remaining compiled, batched and model-specific acceptance checks. The caller owns merge selection and any randomness used in making that choice.

## Definitions

A generation counts how many successive occupants a slot has had. A logical identifier names one occupant. A reset prior is the initial belief restored when the occupant leaves. A carry stores the part of a numerical value that does not fit in its primary floating-point number.


## Notes

**2026-11-24T05:26:39Z**

Implemented cffg.component_bank.ComponentBank for host-side transactions on one bank. Birth chooses the lowest unoccupied slot with a new logical ID and incremented generation. update calls the public compensated mixed learner. propose_merge(receiver, donor) returns a candidate without changing its source. Exact selected-row arithmetic on centered NIW statistics transfers the donor's prior-subtracted evidence across all six high/low fields; the source prior and donor reset remain independent of proposals. Both populated and empty banks use a versioned standalone codec with nested exact v3 mixed records. Keys and candidate choice remain with the caller.

uv run --frozen --no-sync pytest -q tests/test_component_bank.py passed 11 tests in 9.85 s after the final edit. Includes real fractional/partially observed updates, the independent uncentered rational natural-parameter oracle, drift below a primary ULP, dormant support, differing priors, invalid covariance, representability rejection, damaged exact snapshots, duplicate/stale identities, actual NPZ restore and 32/500/5000-slot layouts. Staging a host constructor reports its host-only contract. tk dep cycle passed.

One synchronized warm proposal per capacity on CPU float32, six features, categorical widths 32/32/4/3/500 and four observations per merged row: 13.53 ms at 32 slots and 23.83 ms at 5,000 slots. These are smoke observations, not a timing distribution; the first calls shared compilation caches. Arithmetic reads only three parameter rows. Producing a new immutable bank copies affected field arrays. No game or full suite here. The parent worktree owner coordinates the final suite and dashboard mapping. cao-24tn owns compiled and batched lifecycle acceptance. car-k6oi remains open behind car-9ccf.

**2026-11-24T05:34:54Z**

Final dependency a72cabc. Added replace_posterior(proposal, previous=source.posterior) for callers with their own per-field write gates. It refuses stale sources, layout/occupancy changes, dormant writes or invalid parameters. Restore also handles alternate encodings of a half-ULP midpoint. After those changes all 12 focused tests passed in 10.55 s. The separate compiled/batched child cao-24tn is in progress; the parent owns the final short suite.
