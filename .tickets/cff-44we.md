---
id: cff-44we
status: closed
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

## Notes

**2026-07-14T03:26:48Z**

Hostile-input plan (2026-07-13): sparse/exact-zero support remains represented only by in-domain NEG values; segmentation removes padding cells and support checks slice exact segments. Vacuous discrete messages initialize uniform per segment; vacuous Gaussian messages remain exact zero eta/Lambda per segment; empty family arenas are zero-length. Float32 offsets are integer metadata and all constructed/scattered values preserve the state dtype; no new numeric guard is needed, and a heterogeneous float32 step is required. Scale acceptance: exercise a 1,000x binary/scalar population plus K=10,000/d=128 outliers; persistent allocation must equal the theoretical sum (1.0x, below 2x), while existing homogeneous parity/runtime suites remain green.

**2026-07-14T03:48:48Z**

Implemented on branch cff-44we-shape-segmented-arenas: compiler prefix offsets pack discrete messages as sum K, Gaussian eta as sum d, Lambda as sum d^2, and conditioned observations as sum d. Runtime gather/scatter, local rule padding, damping, delta hardening, all readouts, Colony/vmap, BFE edge ledgers, and CSP decimation now use exact segments. Estimate.arena_bytes reports every persistent State arena by name; cff-j7mm remains responsible for transient/batch accounting, XLA diagnostics, and calibration timing. Hostile 1,000-small + K=10,000/d=128 test allocates exactly 24,000/2,256/34,768 discrete/eta/Lambda values (1.0x representation) and passes a finite float32 sweep. Full verification: 1,132 passed, 4 dependency deprecation warnings, 516.44s. Gotcha for internal consumers: State.messages/g_eta/g_lam/g_obs are now 1-D; use Arena.message_indices/g_eta_indices/g_lam_indices/g_obs_indices.
