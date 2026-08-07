---
id: cff-y4nx
status: open
deps: [cff-k9ws, cff-4apw, cff-twzc]
links: []
created: 2026-07-15T01:46:10Z
type: feature
priority: 2
assignee: coopwilliams
parent: cff-ahps
tags: [hybrid, validation, rslds]
---
# rSLDS tier: recurrent switching state-space with continual learning, end to end

Integration tier once the recurrence factor (cff-k9ws) and its weight learning (cff-4apw) land: a full recurrent switching linear dynamical system — matrix_cat/recurrence-link regime chain, gaussian_mixture_linear per-regime dynamics, gaussian_linear emissions — with all three learning loops running between trials (Dirichlet counts for discrete transitions, MNIW for per-regime dynamics, PG posterior for recurrence weights). Mirrors the switching-KF tier's honesty pattern: engine vs independent numpy reference (tight) and vs exact regime-sequence enumeration on small T (measured gap with headroom, never claiming assumed-density equals exact).

## Acceptance Criteria

1. Small-T posterior gap vs exact enumeration measured and pinned with headroom. 2. Joint continual learning recovers generating dynamics + recurrence weights on synthetic data within documented tolerances. 3. Streaming filter holds flat memory over long horizons (compile-once/swap-params). 4. BFE ledger finite and bound-direction-consistent throughout; float32 sweep survives at 10-100x fixture scale.

