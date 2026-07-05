---
id: cff-iovq
status: closed
deps: []
links: []
created: 2026-07-05T23:23:05Z
type: task
priority: 0
assignee: coopwilliams
parent: cff-1d6t
---
# Delta form-constraint on edges (MAP readout)

DELTA bead on an edge -> after each sweep project the edge to a pointmass at the argmax of its marginal (both slots one-hot), per Senoz-style form constraints. Needed for Part I Fig.29/30 variant.


## Notes

**2026-07-05T23:32:49Z**

Landed: DELTA bead -> delta_pairs projection at argmax (both slots one-hot); harden flag + two-phase run() (premature pointmass locks its own argmax — soft to coherence first, default harden_last=n/2); damp param (log-space blend, ~0.5 for multi-composite models whose outer coordination limit-cycles at ~0.03 amplitude undamped); NEG floor per sweep (exact log-zeros compound additively under hard deltas -> -inf -> NaN; exp identical either way).
