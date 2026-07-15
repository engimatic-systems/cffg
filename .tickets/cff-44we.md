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

**2026-07-15T01:47:07Z**

RESTORED NOTE (originally written 2026-07-13, lost to a git reset --hard before it was committed): ACCEPTED (2026-07-13 review, commit adef83a). Judged against the six acceptance criteria: (1) discrete messages are one flat sum-K array addressed by message_starts prefix offsets — met; (2) Gaussian eta/Lambda are flat sum-d / sum-d^2 arrays with per-slot offsets, O(sum(d+d^2)) — met; (3) parity retained across gather/scatter (_disc_rows/_disc_both/_gauss_rows plus exact-segment scatters), damping, delta hardening (per-cardinality groups), all readouts, Colony (shape-agnostic pytree vmap, unchanged), BFE edge terms (interior_slot_groups uses the identical filter as interior_slot_pairs), and serialization (templates recompile deterministically) — met; (4) Estimate.arena_bytes reports exact bytes for all eight persistent State arenas, with message_bytes kept as a compatibility property — met; (5) hostile 1000-small + K=10000/d=128 test asserts exactly 1.0x representation (>800x/>900x vs the removed padding) plus a finite float32 step; full suite re-run independently during review: 1132 passed, 4 dependency deprecation warnings, 528s — met (runtime regression is covered only by suite wall time, no dedicated benchmark; timing/XLA accounting is deferred to cff-j7mm, which is acceptable); (6) rule signatures unchanged — k_max/d_max arguments now mean group-local widths with docstrings updated, so the rule-registry refactor is not blocked — met. Transient rule blocks still pad to the per-group maximum, which is by design (homogeneous groups) and documented. Docs (ARCHITECTURE.md) updated; no stale global-padding references remain. PR opened: https://github.com/engimatic-systems/cffg/pull/27 (base: main).

