---
id: caci-0rj9
status: in_progress
deps: []
links: []
created: 2026-11-24T05:38:52Z
type: task
priority: 2
assignee: coopwilliams
parent: cff-xfnx
tags: [axiom, checkpoint, performance]
---
# Reduce save and restore time for mixed-bank checkpoints

## Background in Plain English — WHY

After cff-xfnx, compact archives kept the same byte count for the same learned parameters. Saving one took 8–97% longer and restoring one took 49–119% longer on the six CPU fixtures. The checkpoint entry points dispatch many numeric checks separately. They must still reject corrupted files and invalid live state.

## Outcomes / Acceptance Criteria — WHAT

1. Compact v2 and exact v3 records retain every field, dtype, shape and bit. Restoring v1, v2 and v3 preserves the existing continuation and copy semantics.
2. Saving a live bank and loading an archive keep the existing structural and numeric rejection rules. Include dormant rows, empty support, mixed or disabled dtypes, invalid scale matrices, and invalid residuals in compact and exact paths.
3. Report paired save and restore costs on the same learned arrays at K=32, 500 and 5000 with one or 32 observations per tick. Compare identical NPZ files and verify exact continuation. Report cold separately from warmed times, including the 10% practical threshold and uncertainty.
4. Run the focused checkpoint and numerical tests, then validate the final integrated code with the short suite.

## Methodological Suggestions — HOW

The architecture index row is Learnable or checkpoint feature. Review both MixedBank.state_dict and MixedBank.from_state_dict, compact and exact, host and JIT numeric validation, legacy and current snapshot layouts. The score/update kernels share the underlying validator. Graph message updating, scheduler dispatch, model structure migration and the dashboard are N/A for this change. Preserve source float32 and float64 checks, scaled matrix guards and exact half-ULP ties. Use the same learned files and run order across implementations.

## Future work

The separate recurrent score-and-update optimization for cff-xfnx addresses compute costs between snapshots.

## Definitions

A compact snapshot stores rounded parameter values. An exact snapshot also saves the smaller corrections used at the next update. One ULP is the gap between neighboring representable numbers. NPZ is a ZIP container for NumPy arrays.

