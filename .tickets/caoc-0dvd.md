---
id: caoc-0dvd
status: in_progress
deps: []
links: []
created: 2026-11-24T05:39:58Z
type: task
priority: 1
assignee: coopwilliams
parent: car-zieb
tags: [axiom, performance, learning]
---
# Reduce recurring CPU cost of compensated mixed scoring and updates

## Background in Plain English — WHY

Remembering small updates fixes a real loss of evidence in online component learning. On the six measured float32 workloads (1 or 32 observations and 32, 500 or 5,000 components), a72cabc increased median CPU time over fd8bfd1 by 79–176% within a compiled stream and 83–148% through the host API. Both workload types matter to the pixel learner.

## Outcomes / Acceptance Criteria — WHAT

1. Qualify the public score and update on the frozen analytical, translated, sparse, long-stream, frame and scale fixtures in their original precisions and tolerances. Preserve transactional invalid-input handling, material effects of small carried values, and the checkpoint representation.
2. Compare an isolated candidate to the retained implementations on the original six workloads and both entry points. Record synchronized costs, paired dispersion, and remaining overhead against the rounded-only baseline.
3. Identify which operations the candidate saves. Document any precision-specific fast path, its error bound and its behavior outside that bound. Keep any optional capability explicit.
4. Give the downstream learner enough evidence to choose a default. Separate recurring compute from checkpoint I/O.

## Methodological Suggestions — HOW

The architecture change paths are Learnable or checkpoint feature and Runtime numerical or status invariant. Check public eager, JIT, mapped, and scanned score/update at float32 and float64. Cover disabled support, low-word exhaustion, large coordinates and nearly singular observations. Engine sweeps, serialization layout and declared-family dispatch are outside this standalone kernel. An analytically bounded matrix perturbation can avoid per-component factorizations in ordinary well-conditioned banks.

## Future work

car-9ccf consumes this timing decision for mixed learning. The Gameworld reproduction in car-zieb can choose its working precision and budget once it has the same-workload comparison. Compact checkpoint I/O has a separate owner.

## Definitions

A low word stores the fraction lost when a number rounds in its working precision. A responsibility is an observation’s fractional assignment to a component. JIT compiles a function, and a scan applies it repeatedly inside one compiled call.


## Notes

**2026-11-24T05:50:46Z**

Qualified candidate 089f377 preserved all frozen accuracy tests and recovered 30–47% of elapsed time against the first compensated implementation. All twelve original/optimized comparisons remain above the human decision boundary. The paired compute scan costs +21.7% to +54.2%, public returned-array calls +22.5% to +50.2%. Exact numbers and 95% intervals in ~/agent-share/axiom-online-precision/compute-confirmation/summary.json. A request to Cooper to choose the default is pending. Continue independent AXIOM design and fixed-action validation on the evaluation branch.
