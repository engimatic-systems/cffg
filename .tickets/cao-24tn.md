---
id: cao-24tn
status: in_progress
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

**2026-11-24T05:31:43Z**

Starting independent implementation on Cooper’s request while prerequisites qualify. Device identities and generations use two uint32 words per host signed-64-bit ID. Exact selected-row host merge supplies a separate oracle; device status and any numerical range refusals will be reported before qualification.

**2026-11-24T05:44:05Z**

Implemented cffg.component_bank_device with from_host/to_host, reference, birth and propose_merge. Each returns device-compatible identities or a fixed-layout Result. Status OK supplies an immutable candidate; FULL/ID_EXHAUSTED, STALE_REFERENCE/SAME_COMPONENT, INVALID_POSTERIOR and NUMERICAL_RANGE preserve the input. IDs use two uint32 words, including under x64-disabled JIT. The module's usage example retries a model-requested NUMERICAL_RANGE using to_host(result.bank) and the same two references in ComponentBank.propose_merge, then evaluates the candidate under the ordinary model acceptance rule. A host-only reduction calls the exact merger directly.

Final validation: uv run --frozen --no-sync pytest -q -m 'slow or not slow' tests/test_component_bank_device.py passed all 17 tests in 34.01 s. Includes eager/JIT/vmap/scan, independent exact selected-row comparison, literal zero support via donor reset, invalid-covariance rollback, stale and full status, uint32 rollover and int64 exhaustion, exact replay, and all 32/500/5000 capacities x both float dtypes. A separate JAX_ENABLE_X64=0 probe produced next ID 4294967296 across a live merge and rebirth. All 24 ordinary seeded fractional-update proposals compared against the exact host accepted: max absolute effective mean error zero; max scale error 5.69e-14 f32 and 9.87e-32 f64. No device timing campaign.

The device can decline very small products and cancellation at ordinary scales; cao-ar2d tracks the concrete two-thirds/five-thirds example. Its exact host retry passed in both precisions. Implementation and focused qualification are complete. Leave this ticket in progress until car-9ccf and the parent worktree's aggregate short suite are ready. The release-specific rMM reduction will use the exact host proposal directly.

**2026-11-24T07:18:58Z**

The aggregate short suite passed at f74543f, including the device lifecycle cases in its default selection. The separate 17 focused and scale cases cover 32, 500 and 5000 slots. The remaining prerequisite for this ticket is adoption of car-9ccf. cao-ar2d tracks device cancellation refusals and the qualified host retry.
