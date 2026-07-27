---
id: cff-ey4p
status: in_progress
deps: []
links: [cff-hp3y]
created: 2026-07-11T04:23:12Z
type: bug
priority: 1
assignee: coopwilliams
tags: [numerical, agent]
---
# RollingAgent belief log-floor underflows in float32

DISCOVERED during cff-hp3y Hallway rerun (2026-07-10). agent.py converts the rolling belief with np.log(np.maximum(self.belief, 1e-300)). When the consumer belief is float32, the literal floor casts to zero, producing divide-by-zero warnings and -inf rather than the engine NEG/log-space convention. This violates the dtype-scaled guard contract and can poison parameter swapping on exact-zero beliefs. Keep separate from GFE-TM cavity consistency.

## Acceptance Criteria

Use a dtype-derived floor or explicit exact-zero log representation with its invariant documented; sparse/zero belief remains semantically zero without warnings; float32 and float64 agent act/observe paths stay finite where the model is feasible; uncomfortable state cardinality regression; no behavior change for full-support float64 beliefs.

