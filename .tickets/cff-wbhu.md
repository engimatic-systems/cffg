---
id: cff-wbhu
status: open
deps: [cff-j4mu, cff-d56d, cff-44we]
links: [cff-fxmj, cff-5edo, cff-d56d, cff-yl0a]
created: 2026-07-10T03:34:54Z
type: feature
priority: 2
assignee: coopwilliams
parent: cff-ahps
---
# Colony throughput: jitted sweep loop (lax.scan) — dispatch-bound batching measured at 1.5x

Colony/vmap batching of 32 instances measured only 1.5x over sequential on CPU: the sweep loop re-enters filter_vmap(step) from Python each sweep, so per-call dispatch overhead dominates while the vmapped kernel itself vectorizes fine. Wrap the sweep loop in a jitted lax.scan (one dispatch per run/tick instead of per sweep), with the delta-trace/tol early-exit expressed as a scan carry (or while_loop). Applies to Engine.run single-instance too (per-sweep Python overhead).

## Acceptance Criteria

32-instance colony >=8x sequential on the same fixture; single-engine run() no slower; delta trace and tol semantics preserved (incl. the non-finite-delta divergence guard); vmap parity tests unchanged.

