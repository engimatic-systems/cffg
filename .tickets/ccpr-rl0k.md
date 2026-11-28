---
id: ccpr-rl0k
status: closed
deps: [cff-cpxj]
links: []
created: 2026-11-28T05:57:47Z
type: feature
priority: 3
assignee: coopwilliams
parent: cff-6s31
tags: [plates, compiler]
---
# Address plate copies and neighbors without constructing an index list

## Background in Plain English

A plate repeats one model fragment over a grid. The current compiler first builds a list containing every coordinate. Large grids need a way to address a copy or its neighbor directly. This also gives parameter binding and the compiler one definition of copy order.

## Outcomes / Acceptance Criteria

1. Convert between a copy number and its complete coordinate in row-major order. The last axis varies fastest.
2. Resolve chain neighbors for each declared axis and positive offset. Preserve the exposed ports at both ends.
3. The standard one-level expansion and at least one other existing consumer use the same addressing rules. Looking up one address allocates space proportional to the number of axes.
4. Reject nonpositive or nonintegral dimensions, an invalid axis or offset, and a shape whose product exceeds signed 64-bit range before enumerating or allocating copies. Include the bad dimensions in an overflow error.
5. Compare every address and boundary on small one-, two-, and three-dimensional grids against direct enumeration. Test single lookups on a grid too large to materialize. Existing plate declarations, paths, and bindings round-trip.

## Methodological Suggestions

Use exact integer arithmetic. Check the existing integer types admitted by the source format and cover NumPy integers where that contract allows them.

## Future work

The compact compiler and resource estimator will consume these addresses. The parent cff-6s31 owns their integration.

## Definitions

A copy number is its zero-based position in the ordered grid. A chain connects one copy's exit port to another copy's entry port.


## Notes

**2026-11-28T05:58:13Z**

Use the New IR field change-path index for structural validation and plate grounding; the serialized field set is unchanged. Check the materializing expansion path, parameter-only Colony binding, public observation fan-out, source validation, and compact edit replay. Grid addressing uses exact host integers; schedules, energy evaluation, JIT, message support and numerical dtype branches are unchanged. Keep the old one- and two-dimensional model admission boundary in this slice; the address calculations themselves handle additional axes for the later nesting ticket.

**2026-11-28T06:02:57Z**

The normal expansion and Colony per-copy parameter resolver now stream row-major coordinates. Direct lookups use one stride per axis. Independent small-grid enumeration and near-limit signed-64-bit lookups passed. Invalid grids and chain fields fail before a copied record is allocated; an invalid axis and offset are reported together. Across 64×64 and 256×256 grids, tracemalloc measured the coordinate iteration alone at 264,608 and 4,237,856 bytes for the previous list, versus 1,216 and 1,344 for the stream. These figures cover Python coordinate allocation, not total compiler memory or time. Final focused run of indexing, plate, composition, Colony spawn, edit, bundle and README tests: 99 passed, 4 opt-in deselected. The single complete short suite remains the review gate.

**2026-11-28T08:06:50Z**

Canonical short suite passed at 15c720f1471c: 2,869 cases in 428.032 seconds. The source allocation and addressing ticket is complete. ccpr-ojza owns inference-array compilation, ccpr-m0bu owns nested models and ccpr-u190 owns the estimator. cff-6s31 remains open.
