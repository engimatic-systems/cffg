---
id: cff-1et5
status: closed
deps: []
links: [cff-6kwo]
created: 2026-09-15T19:14:45Z
type: bug
priority: 3
assignee: coopwilliams
---
# Differentiate hybrid entropy-bound diagnostics on sparse support

## Background in Plain English — WHY

Hybrid GFE training evaluates the reported energy without the optional entropy-bound diagnostics. A separate lower-bound diagnostic still evaluates log of an exactly zero regime probability before masking it, so reverse differentiation of that diagnostic can produce NaN. Current neural training does not use this path.

## Outcomes / Acceptance Criteria — WHAT

1. The optional pairwise entropy lower bound has finite forward and reverse derivatives on fixed sparse support without adding mass.
2. Bound values retain their existing meaning and agree before and after the derivative repair.
3. Sparse float32 and float64 cases, including a larger regime count, verify the derivative and value contracts.

## Methodological Suggestions — HOW

Use Runtime numerical or status invariant. Inspect hybrid_gfe._pairwise_entropy_lower; use finite inactive logarithm operands, as in cff-6kwo. Keep this distinct from the trained energy, whose include_bounds=False path already avoids the diagnostic. Improper Gaussian inputs retain existing failure semantics.

## Future work

No further work is identified. This ticket does not block cff-1o72.

## Definitions

Entropy bound: a computable limit on mixture uncertainty. Fixed support: exactly zero probabilities remain excluded. Reverse differentiation: gradients propagated backward from a scalar output.

## Notes

**2026-11-28T04:20:34Z**

Architecture handoff: runtime numerical or status invariant. The prior and affine diagnostics share hybrid_gfe._pairwise_entropy_lower through _terms(include_bounds=True). The trained energy uses include_bounds=False. Exercise exact zeros on a nonempty fixed simplex face, independent overlap values and derivatives, true float32/float64 and up to 32 regimes. Keep the public diagnostics and prior improper-Gaussian statuses intact. No IR, persistence, solver, flooding/scheduled, or Colony change. Base is the freshly fetched origin/main a774a54.

**2026-11-28T04:24:53Z**

Implemented the finite inactive log operand in the optional pairwise bound. At a774a54 the reverse weight derivative was NaN at all 29 excluded labels of the 32-regime float32 case and the single excluded label of the 4-regime float64 case. Independent scalar Gaussian overlap formulas check the bound, JVP and every VJP entry for weights, means and variances on the fixed support. Inactive derivatives are exactly zero. A solved mixture-prior diagnostic also matches the derivative for two symmetric normals plus a third excluded component. Sixteen old/new eager/JIT primal comparisons with 3D SPD covariances, four support sizes and both dtypes were bit-identical. Commands from this worktree with uv run --frozen python -m pytest -q -o addopts= tests/test_hybrid_gfe.py tests/test_neural_gfe.py::test_sparse_hybrid_energy_reverse_derivative_matches_forward_mode passed all 16 in 87.21s, including improper-input statuses. A final dtype assertion rerun (-k sparse_pairwise) passed both cases in 3.85s. The true-float32 test switches off x64 after module import, so JAX warns about the existing log(2 pi) constant; the bound, JVP and all three VJPs are verified float32. Parent coordinates the final combined short-suite gate and review.

**2026-11-28T05:02:38Z**

Combined reviewed source 9f3bc99 based on #208. Canonical uv run python tools/dashboard.py --run-suite passed 2,860 selected tests on four workers, 431.726 seconds elapsed (431.561 seconds in the dashboard runtime history). Final focused cross-ticket checks passed 22 tests in 8.33 seconds. Log and history: ~/agent-share/cffg-hybrid-math-9f3bc99/. The stacked PR contains the tree derivation and oracle, sparse entropy diagnostic fix, and destination-softmax derivation. Production softmax remains deferred in cds-1t47.
