---
id: cff-7t4j
status: open
deps: []
links: []
created: 2026-11-24T01:37:29Z
type: bug
priority: 1
assignee: coopwilliams
parent: car-zieb
tags: [axiom, learning, numerics]
---
# Preserve within-component Gaussian uncertainty for mixed-evidence learning

## Background in Plain English — WHY

A Gaussian component has a location and a spread. The existing mixture readout returns the weighted mean and raw second moment. Rounding can make two different spreads produce identical values for both fields. The planned component learner cannot recover that lost information from those fields, even with a double-precision accumulator.

## Outcomes / Acceptance Criteria — WHAT

1. Expose accurate conditional or weighted within-component covariance alongside a numerically stable representation of the location for the Gaussian mixture readout. Preserve the contract for existing callers of the raw-moment fields. Handle dormant components without dividing by zero.
2. Use the public fixture in tools/axiom/moment_probe.py. With equal weights and mean 10000, distinguish the true variances 0.125 and 0.5. With two displaced components, recover 1/6 and 1/5 to working precision. Check weights independently, the translated and near-zero controls, eager/compiled kernels, single/batched members and float32/float64. Include nonfinite/improper inputs and exact zero support.
3. Show that pooling the centered evidence with fractional responsibilities yields the same posterior spread after translation and after splitting a batch. Preserve preexisting outputs and current/legacy learner state. Establish whether any other public learning readout has the same information loss and record its scope.
4. Measure time and device/host transfer sizes on representative banks before adopting a default. Surface material performance tradeoffs to Cooper.

## Methodological Suggestions — HOW

The mixture rule already computes the conditional covariance before forming the raw second moment. Return it directly or provide an equivalent centered statistic. Accumulate within-component spread and between-mean scatter separately. Compare to the scalar product-of-Gaussians formula. Architecture change paths are New factor or rule family and Learnable or checkpoint feature. Check host/compiled and public readout pairs. The readout itself does not schedule messages. For integration, exercise supported schedules and check serialization only if persistent state changes.

## Future work

car-9ccf uses these statistics in the mixed Gaussian/categorical learner. AXIOM pixel and game performance require the downstream model.

## Definitions

Within-component variance measures uncertainty conditional on one object type. A raw second moment is the average of the squared coordinate. Translation adds the same constant to every location in a problem.


## Notes

**2026-11-24T01:37:58Z**

At 125f0a4 the public readout on two equally weighted components at coordinate 10000 returned identical float32 weights 0.5, weighted means 4999.99951171875 and weighted second moments 49999992 for two models whose true posterior variances were 0.125 and 0.5. Both reported proper. No downstream function of those three fields alone can distinguish these cases. On the displaced fixture the two reconstructed variances were 3.068345 and -2.423770. Centered and float64 controls passed. Independent kernel-level reproduction also loses unit variance at 4096 and a separate direct conj.NIW.update check gives scale 1 instead of 3 with float32 prior mean 10001, kappa 2, psi 3, nu 4 and an exactly matching point supplied in float64. Preserve within-component spread at emission and centered arithmetic during pooling. Run uv run --frozen python tools/axiom/moment_probe.py float32 and again with float64.

**2026-11-24T01:39:38Z**

Reproducer handoff: this isolated diagnostic only reads existing public Engine outputs and writes JSON to stdout. No library, test-selection or pass/fail logic changed. Ran both dtype commands and independently checked the 20 rows, exact equality of the ambiguous raw statistics, analytic weights, translated controls and negative reconstructed variance. tk dep cycle and git diff --cached --check passed. The blocker stays open.
