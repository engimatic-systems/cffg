---
id: cff-j7mm
status: open
deps: [cff-44we]
links: [cff-fxmj, cff-44we]
created: 2026-07-10T18:05:50Z
type: bug
priority: 2
assignee: coopwilliams
tags: [adversarial-reviewer, performance, estimator]
---
# Resource estimator ignores non-discrete arenas and its XLA analysis no longer calls step correctly

ADVERSARIAL REVIEWER FINDING (2026-07-10). estimate() defines message_bytes as state.messages.nbytes and params only, omitting g_eta, g_lam, g_obs/g_mask, ef_eta/obs/mask and any future arena state. On a 20-step d=2 LGSSM it reports 1,104 bytes while the State leaves occupy 5,088 bytes; message_bytes is literally zero although Gaussian messages use 3,744 bytes. _cost_analysis still calls engine._step(m,p,0.0,True) using the pre-multi-arena signature; the exception is swallowed, so flops_per_sweep and bytes_accessed_per_sweep are always None. README/ARCHITECTURE call analytic memory exact, making this a misleading capacity-planning API.

## Acceptance Criteria

1. Exact persistent and transient byte accounting covers every State leaf by named arena category, observations/masks, params, and batch dimension. 2. XLA lowering calls the current public/internal step signature and a test requires non-None cost data on a supported backend; unsupported-backend errors are exposed as diagnostics rather than silently swallowed. 3. Estimates cover discrete, Gaussian, EF, hybrid, plate, and Colony batch fixtures and equal actual pytree leaf bytes. 4. Calibration synchronizes every relevant output and reports warm vs compile separately. 5. Integrate shape-segmented accounting when cff-44we lands.
