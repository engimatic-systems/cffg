---
id: cff-6kwo
status: closed
deps: []
links: [cff-1et5]
created: 2026-09-15T19:10:07Z
type: bug
priority: 2
assignee: coopwilliams
parent: cff-1o72
---
# Keep sparse GFE derivatives finite on exact support

## Background in Plain English — WHY

A hybrid Gaussian/discrete energy can be finite when one regime has exactly zero probability, while reverse differentiation produces NaN. The entropy expression evaluates log(0) before masking it. A transition-mixture message has the same mechanism when an output row has exactly zero predictive mass. Neural training needs finite derivatives on that fixed support.

## Outcomes / Acceptance Criteria — WHAT

1. Entropy and transition-mixture messages preserve exact zero mass and finite values without adding probability floors.
2. Forward and reverse derivatives of the complete hybrid energy and zero-row transition-mixture messages agree on sparse fixed support in float64 and float32.
3. A neural sensor coupled to an admitted hybrid GFE graph has a certified finite total gradient and preserves exact zeros.

## Methodological Suggestions — HOW

Use Runtime numerical or status invariant. Replace inactive logarithm operands with one before differentiating, leaving their entropy terms exactly zero. Audit local forward/reverse mode and complete residual/ledger; flooding and mixed schedules retain the same owner. Sparse support is the failure mechanism; vacuous and improper cavities keep existing refusal semantics. Float32 and 32-regime scale use dtype-aware tolerances; no covariance or probability floors.

## Future work

This unblocks cff-ll6y and the cff-1o72 neural training PR. No separate future work is identified.

## Definitions

Fixed support: probabilities that are exactly zero stay excluded. Reverse differentiation: computing parameter gradients by propagating output derivatives backward. GFE: Generalised free energy.

## Notes

**2026-09-15T19:15:01Z**

Research question: do existing deterministic GFE callbacks differentiate the complete equations? Answer: fixed-A and fixed-goal hybrid custom JVPs restore full cavity/parameter dependence after stopping solver history. Two unsafe inactive logarithms poisoned reverse mode: hybrid entropy and transition-mixture zero predictive rows. Both now evaluate finite inactive log operands without changing exact support or adding mass. Source: rules.py _implicit_gfe_solution_jvp and gfe_tm_update; hybrid_gfe.py _implicit_solution_jvp and _discrete_entropy; confidence high from forward/reverse probes and independent complete-energy finite differences. Optional pairwise entropy-bound diagnostic differentiation is outside training and tracked separately as cff-1et5. Focused regression passed5 tests in32.21s, including local float32 K32, sparse TM JVP/VJP, complete neural-to-hybrid directional finite differences and stale-state refusal. A final stronger three-regime fixture retains two active regimes plus one exact zero; final parent suite follows.

**2026-09-15T19:15:19Z**

Final strengthened focused check: uv run pytest -q tests/test_neural_gfe.py tests/test_amortization.py —6 passed,1 slow deselection in47.60s. The complete gradient fixture uses three regimes with two positive probabilities and one exact zero. Recognition tests also carry a separate discrete exact-zero island and preserve its messages bit-for-bit through proposal publication.
