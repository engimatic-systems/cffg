---
id: ccbc-22u9
status: closed
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

**2026-11-24T06:00:16Z**

Implemented at 0b6ac43394c55765d3c55b2fd26e9af1c0f45b6c on fix/component-bank-compact from 81805fd. ComponentBank.state_dict() uses lifecycle v2 with two compact v2 mixed records. Both restored carries start at zero. Pass exact=True for the historical lifecycle v1 with two v3 records and all corrections. Both modes retain the same logical IDs, generations, next ID and per-bank occupancy. Each envelope requires its matching pair of nested versions and complete field sets. Equal exact counts with opposite midpoint encodings take the reset prior's stored primary on compact save, preserving zero excess in either orientation. A compact save modifies only the returned host copies. Device-to-host transport explicitly calls the exact codec. Gauss confirmed the lifecycle boundary and received the implementation SHA for the learned reduction replay.

Focused validation from this worktree:

```sh
PYTHONPATH="$PWD/src" uv run --project /Users/ravd/projects/cffg-axiom-online --frozen --no-sync python -m pytest -q tests/test_component_bank.py tests/test_component_bank_device.py tests/test_axiom_mixture.py tests/test_mixed_learning.py
```

49 passed and six opt-in slow cases were deselected in 51.21 seconds. Float32/float64 tests cover low words in both banks, exact learned continuation and merging, empty/dormant slots and rebirth, both directions of the count-midpoint tie, original format restore, malformed/mixed versions, illegal count/ID changes, extra or missing fields, and atomic state after a save. The unmodified reader/writer from 81805fd restored all three real learned exact files and reproduced all 38 fields byte for byte. The old reader explicitly rejects the newer compact envelope. The full suite was outside the requested scope.

Actual files and reproduction script are under ~/agent-share/axiom-online-precision/component-bank-compact. measure.py uses the archived development n1-k32, n1-k500 and n1-k5000 learned exact posteriors and their original seeded reset priors. Each has six Gaussian dimensions, categorical widths 3 and 5, float32, and 32 preceding training batches. sizes.json includes digests, array payload sizes, physical file lengths and learned nonzero correction counts. legacy-reader.json records the cross-version check.

| Slots | Stored compact | Stored exact | Deflated compact | Deflated exact |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 20,466 B | 37,822 B | 15,735 B | 23,156 B |
| 500 | 227,322 B | 443,110 B | 165,013 B | 236,555 B |
| 5,000 | 2,216,322 B | 4,340,110 B | 1,600,293 B | 2,305,932 B |

All six files in each compression mode passed public round-trip checks. Default files have 24 fields, exact files 38. These sizes describe the documented synthetic learned shapes; categorical widths and learned values affect other workloads. Reproduce the measurements using the same uv environment as the tests, JAX_ENABLE_X64=0, JAX_PLATFORMS=cpu, PYTHONPATH="$PWD/src:$PWD/tools/axiom" and the archived measure.py.
