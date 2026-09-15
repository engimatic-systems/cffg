---
id: cff-1et5
status: open
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


