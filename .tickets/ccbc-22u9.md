---
id: ccbc-22u9
status: in_progress
deps: []
links: []
created: 2026-11-24T05:54:25Z
type: task
priority: 2
assignee: coopwilliams
parent: car-k6oi
tags: [checkpoint, axiom]
---
# Make rounding corrections optional in component lifecycle checkpoints

## Background in Plain English — WHY

The mixed learner already gives callers a choice about saving rounding corrections. Its component lifecycle wrapper always writes them twice, once for the reset prior and once for the posterior. A user who wants rounded parameters currently pays the larger file cost without requesting exact continuation.

## Outcomes / Acceptance Criteria — WHAT

1. ComponentBank.state_dict() saves both mixed banks in compact form. Passing exact=True saves every correction using the existing exact lifecycle schema. Both retain the occupancy, identifiers, generations and next identifier. The device-to-host bridge preserves every correction.
2. Restore every historical exact snapshot and the new compact schema, including empty banks. Reject mixed nested formats, damaged fields, invalid identities, counts below the reset prior and dormant rows with learned evidence. Compact reload starts both banks with zero corrections.
3. Compact rounding preserves zero excess for an equal prior/posterior count even when the exact source words use opposite midpoint endpoints. Saving leaves the source bank untouched. Exact reload reproduces the next learned and lifecycle operations bit for bit.
4. Record observed uncompressed NPZ file sizes for learned component banks in both modes. Include float32/float64, proper dormant priors, sparse assignments, empty support and malformed snapshots in focused validation.

## Methodological Suggestions — HOW

Reuse the mixed learner's v2 compact and v3 exact codecs. Follow the Learnable or checkpoint feature row in docs/ARCHITECTURE.md. Check save/reload, device transport, host-only empty banks and both precisions. The graph schema, scheduled/synchronous messages and the scoring formulas are N/A. Big-bank timing and the full suite fall outside this change.

## Future work

car-k6oi owns device and host lifecycle integration. The learned scene can request exact checkpoints for deterministic training continuation.

## Definitions

A primary word holds a rounded numeric value. A low word keeps the remaining correction. A dormant slot is reusable and holds its reset prior. A midpoint has two equally near primary words.


## Notes

**2026-11-24T05:54:58Z**

Gauss confirmed the device-to-host conversion and stale-source guard must remain exact. The compact format uses outer v2 with two nested v2 mixed records. Exact serialization keeps the historical v1 envelope with two nested v3 records. A posterior count with the same exact midpoint as its reset prior must project to the same compact value even when their stored high words differ.
