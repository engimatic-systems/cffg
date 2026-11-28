---
id: cpgh-tday
status: closed
deps: []
links: []
created: 2026-11-28T05:18:33Z
type: task
priority: 1
assignee: coopwilliams
parent: cff-c5qf
---
# Compare exact partial-goal smoothing and a Gaussian approximation on the retained hybrid fixture

A position-only goal can influence earlier velocity and action through the dynamics. The current mixing rule discards a backward likelihood flat in one direction. Quantify what information the existing graph actually contains before changing its solver.

## Design

Research calculations and executable reference only. Keep the original A, offsets, covariances, action prior, initial state and observations. Use H=1,2,15 and change only the final position reading from 0 to 2. Compare current declared collapse, exact summation over discrete action sequences, and moment-matched Gaussian future control offsets on the shared informative subspace. Double-check exact H=1,2 with an independent joint-Gaussian integral; retain original normalizers and exact-zero action support, include f32 sensitivity. No artificial velocity goal, ridge or full-tilt EC runtime.

## Acceptance Criteria

Report all first-action probabilities and changes, modal decisions, log evidence where meaningful, and the cost of each method. Explain which result is exact for the declared graph, which belongs to the surrogate, the null-direction behavior, and conditions under which the method fails. Keep a separate gated implementation decision and unchanged production runtime.


## Notes

**2026-11-28T05:19:01Z**

Change paths: numerical contract and math oracle only. Current public engine records use the retained default inference path; the independent enumerator uses Kalman innovations, with direct dense Gaussian conditioning as the second reference. Enumerate all 3^15 paths via fixed-prefix batches, never prune. All schedules and current/legacy codecs remain owned by existing runtime; no serialization or production code in this review.

**2026-11-28T05:24:45Z**

Compared all six registered cases through Engine.run (converged in H+1 sweeps), a path-by-path normalized Gaussian reference, and the future-offset approximation. The exact two-step first move changes 0 to 2. The approximation captures this switch, but its 15-step log evidence at final goal 2 is 67.785 nats too high. Full policies and scores are reproducible with uv run python tools/partial_goal_comparison.py --native. Local receipt ~/agent-share/cffg-partial-goal-review/retained.json SHA256 4e6622e0a641cfa2784a1a9916a71a5a24016906343c58bc92dd87d49137d932. Five research checks plus the existing public planning regression passed. Exact 1/2/3-step densities, sparse/singleton action support, common-direction admission and consumer f32 were exercised. Two independent reviews reproduced the 2-step numbers. See D3.3 for the derivation and scope. Isolated comparison imports the unmodified fixture and public runtime. No default test, execution, dependency or codec changes; validation used targeted runs.
