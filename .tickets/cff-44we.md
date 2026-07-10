---
id: cff-44we
status: open
deps: []
links: [cff-fxmj, cff-j7mm]
created: 2026-07-10T18:05:00Z
type: feature
priority: 1
assignee: coopwilliams
parent: cff-ahps
tags: [adversarial-reviewer, performance, arena]
---
# Replace global k_max/d_max padding with shape-segmented arenas

ADVERSARIAL REVIEWER FINDING (2026-07-10). Every discrete slot is allocated [k_max] and every Gaussian precision slot [d_max,d_max], where maxima are global across the model (compiler.Arena; Engine.init_state). One outlier domain therefore multiplies every other edge's state, bandwidth, damping, and convergence scan. Example formulas in float32: 1,000 binary variables plus one K=10,000 variable allocate 2,002*10,000=20.02M message values (~80.1MB) versus 24,000 values (~96KB) in a flat segmented arena, ~834x overhead. For 1,000 scalar Gaussian edges plus one d=128 edge, g_lam alone is ~131MB versus ~139KB block-ragged, ~942x. This conflicts with the PGMax-inspired scalable-arena claim and will become worse as modular factor families mix domain sizes.

## Acceptance Criteria

1. Discrete messages use flat value storage plus per-edge starts/lengths, or equivalently group-segmented buffers, so memory is O(sum K_e), not O(E*max K). 2. Gaussian eta/Lambda storage is dimension-grouped/block-ragged, O(sum(d_e+d_e^2)), not O(E*max d^2). 3. Gather/scatter, damping, delta constraints, readouts, Colony batching, BFE edge terms, and serialization-derived reconstruction retain parity. 4. Resource estimator reports exact bytes for every arena. 5. Heterogeneous hostile tests demonstrate <2x representational overhead and no compile/runtime regression on homogeneous models. 6. Design coordinates with, but does not wait for, the executable rule-registry refactor.
