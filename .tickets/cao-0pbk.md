---
id: cao-0pbk
status: in_progress
deps: []
links: []
created: 2026-11-24T05:44:59Z
type: bug
priority: 1
assignee: coopwilliams
parent: car-mps6
tags: [axiom, parity, numerics]
---
# Apply the released AXIOM Gaussian scale policy to model likelihoods

The released agent changes its Gaussian scale before computing a score. With seven context features, one origin observation and a displacement of 0.03, the mathematically unmodified Gaussian scores -11.558 nats and the released scale scores +15.697 before categorical evidence. This can reverse the -10 learning threshold. Scope the adjustment to the AXIOM experiment and preserve the original learning state.


## Reviewable outcomes

1. Convert a model snapshot to the released inverse-Wishart scale used in expected Gaussian precision and log determinant. Compare both precisions to an independent calculation from raw natural parameters and check the exact threshold, correlated features, displaced observations and a loss of positive definiteness.
2. Exercise the changed match or birth decision on a small deterministic fixture. Keep parameter learning and exact continuation on the original arrays. Connect training, context-only prediction, sampled model reduction and search to the model-facing readout as each becomes available.
3. Record which consumers are qualified before claiming released-agent parity. The general CFFG mixed learner keeps the ordinary NIW definition.

## Method and scope

The architecture path is a runtime numerical invariant. Check eager/compiled, both dtypes and batched components. Message schedules and Bethe ledgers are inapplicable to this experiment-specific readout. The public release b99f625, axiom/vi/conjugate/mvn.py lines 184–194, 247–264 and 588–618, rescales the Gaussian scale by the magnitude of its second natural parameter, bounds its normalized diagonal from below and zeros sufficiently small normalized off-diagonal entries. A view is a derived set of arrays used to evaluate scores, separate from the bank saved for the next update.
