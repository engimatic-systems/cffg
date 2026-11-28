---
id: cei-k93f
status: closed
deps: [cei-6lc5]
links: []
created: 2026-11-28T06:17:51Z
type: feature
priority: 2
assignee: coopwilliams
parent: cenc-luez
tags: [serialization, neural, inference]
---
# Bind the exact interval domain to serialized neural factors and graph inference

The exact interval algebra needs an ordinary model entry point with an explicit domain. Gaussian dimensions currently imply Gaussian messages; the new template must bind a separate interval domain without changing positive-noise neural semantics.

## Design

New IR field, new factor or rule family, and runtime numerical or status invariant. Carry independent coordinate dimension and static interval capacity through model declarations, compile, sources and bundles. Bind parameter vectors for input/output knots and positive slopes. Preserve complete cells and propagate typed failures through scalar/batch and flooding/exact routes.

## Acceptance Criteria

Round-trip a template and bundle and run the reconstructed model against the independent D13a tree oracle in f32/f64 at 1D/32D. Attribute -H(input) at the graph factor and both positive separator entropy corrections once. Check capacity failure, impossible/improper/numerical diagnostics, stable model/shape identity, and no aliasing to Gaussian inference. Document the supported graph boundaries and unchanged positive-noise behavior.


## Notes

**2026-11-28T06:23:48Z**

Parent approved an interval-only sum-product forest specialization through normal compile_template, Engine and Colony. Admit declared interval_potential unary, interval_activation, equality, DATA-bound interval likelihoods or missing unit. Build exact directional dependency order from the actual Forney incidence, run flood from simultaneous snapshots, and retain evidence scales. Use existing source data and bundle loading where appropriate and retain explicit working-precision geometry checks. No per-state storage on legacy models. No Gaussian bridge, cyclic/constraint/learning admission or parameter inference. Verify serialized/hash identity, 5 signed contributions, multi-instance batch failure isolation and proper/impossible/improper/numerical/capacity statuses against D13a.

**2026-11-28T06:39:16Z**

Implemented interval-only forest admission through Template/compile_template/Engine/Colony. New variables persist interval_dim and interval_capacity; interval_activation binds knot/offset/left/right; interval_potential binds finite lower/upper/log-height/count with explicit outer tails flags. Normal JSON, shape hashes, .npy arrays and CFFGZ reconstruct without weakening existing finite-blob checks. Per-instance raw chart metadata distinguishes merged boundaries on narrowing; the legacy runtime State has no additional storage. Topological directed messages implement exact two-pass; synchronous flood solves from simultaneous snapshots, with exact stationarity support for repeated small budgets. The JAX path and independent D13a oracle agree on the two-chart Jacobian, all five signed contributions, 3/8 event, separate component normalizers and the 32D product. Capacity and invalid geometry roll back per member. Proper finite, infeasible +inf, improper and numerical outcomes have separate diagnostics. Focused command on interval messages/forest, D13a, IR, serialization, plates, bundle, compiler contracts, README, Colony and readout ownership: 183 passed, 7 deselected. Final JSON/IR/bundle focused rerun after source-format cleanup: 38 passed. Canonical closure follows.

**2026-11-28T14:13:40Z**

Implemented the interval-only forest, normal template/compiler/Engine and Colony admission, model reconstruction, two boolean outer-tail flags for finite serialized source arrays, per-member failure isolation and both schedules. ff95806 contains the graph integration. Final diagram registration 94df674 and its focused regression passed. Integrated short suite passed 2,858 tests in 421.89 s.
