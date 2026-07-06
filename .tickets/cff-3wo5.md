---
id: cff-3wo5
status: closed
deps: []
links: []
created: 2026-07-06T02:45:04Z
type: task
priority: 1
assignee: coopwilliams
parent: cff-xjlh
---
# Resource estimator: analytic memory + XLA cost analysis + calibration

cffg/estimate.py: message/param bytes from arena shapes (exact, static); flops + bytes-accessed per sweep from jit lower/compile cost_analysis (defensive: may be None per backend); measured seconds/sweep via calibration run; predict(n_iters).


## Notes

**2026-07-06T02:49:57Z**

Landed estimate.py: exact message/param bytes from state nbytes, XLA cost_analysis flops+bytes (works on CPU: tmaze planner = 9.1KiB state, 27.6k flops, 0.2ms/sweep), calibration + predict(n). Defensive None when backend refuses.
