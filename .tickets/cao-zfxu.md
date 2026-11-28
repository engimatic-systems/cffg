---
id: cao-zfxu
status: closed
deps: []
links: [cff-6wpk]
created: 2026-11-24T06:37:06Z
type: bug
priority: 1
assignee: coopwilliams
---
# Keep centered Gaussian registration usable without JAX

## Background in Plain English

Opening and validating a model or inspecting compiled model descriptions can run without the numerical runtime. Registering the new centered Gaussian statistics immediately imports JAX, so both operations fail in an environment that blocks it.

## Outcomes / Acceptance Criteria

1. The public model core validates and serializes templates without importing JAX or Equinox.
2. The compiled value types load with the runtime and compiler blocked.
3. The centered Gaussian prior and regression callbacks retain their compiled execution and numeric results on first and subsequent reads.

## Methodological Suggestions

Use the existing Gaussian registration owner to defer compilation until a statistics callback runs. Change path is New factor or rule family. Check host registration and JIT execution. Solver scheduling, batching semantics and checkpoint formats are unchanged.

## Future work

The integration owner runs the cumulative short suite.

## Definitions

JAX supplies compiled array operations. Centered statistics retain covariance directly for conjugate learning. Registration associates a factor with its numerical operations.


## Notes

**2026-11-24T06:38:39Z**

The built-in Gaussian registry now creates its centered prior and regression JIT wrappers on first use. Both callbacks retain their original static arguments for dimensions and observed role. An instrumented six-read probe counted one JIT constructor per callback. The compiled-value and core-validation import guards failed before the fix and passed afterward (2 tests in 0.40 s). The four focused files test_compiled.py, test_core_imports.py, test_gaussian_kernels.py and test_gaussian_learnables.py passed 86 tests, with 2 slow cases deselected (41.35 s). These include actual compiled and eager numeric comparisons. No changes to the guards or numerical formulas. The integration owner has the final suite gate.
