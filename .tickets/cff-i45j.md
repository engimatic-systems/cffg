---
id: cff-i45j
status: open
deps: []
links: [cff-cj5h]
created: 2026-07-08T22:00:50Z
type: bug
priority: 1
assignee: coopwilliams
---
# RockSample pathology: GFE agent checks the least-informative rock forever

Diagnosed 2026-07-08 from experiments/cffg-rl-0 artifacts (episodes=8, horizon=4, 3x3, 3 rocks): the GFE agent spends ALL 10 steps on check_2 — the FARTHEST rock (distance 4 from start, sensor accuracy ~0.568, barely above coin-flip) — never moves, never samples, return -1.0. The BFE ablation never checks: south-south-sample blind (62% success = p_good luck). Neither balances epistemic and pragmatic value; and picking the LEAST informative check suggests something worse than over-exploration — candidate: the near-uninformative check maximally PRESERVES the epistemic term for future slices (a curiosity trap: the composite scores expected info gain per slice, and a slow-resolving sensor keeps that term alive every tick), or the policy posterior is near-flat and MAP tie-breaking lands on it (cff-cj5h's diagnostic would tell us). CANDIDATE CAUSES to discriminate, in order: (1) no termination/absorbing-state modeling — env ends on sample but the model plans past it, so cashing in is never urgent (receding-horizon procrastination; the papers' hard-horizon caveat); (2) epistemic term does not saturate along a planned path — T-maze works because the cue resolves in ONE perfect observation; noisy repeated checks are outside the papers' validated regime; (3) under-convergence: 4 P-substituted composites + damp 0.5 + 220 sweeps may limit-cycle (check FE trace + message deltas); (4) MAP tie-breaking on a flat posterior. EXPERIMENTS: hand-score per-action GFE from the start belief (which term prefers check_2?); log policy posteriors per tick; horizon sweep 2..6; utility_scale sweep; add an absorbing 'done' mode to the model and re-run. NOTE: this is a PLANNING-side weakness — RockSample's parameters are exactly known, so the learning tickets (cff-cd1i etc.) will not fix it. DELEGATION: the scoring/diagnosis experiments are Fable-level; Sonnet can run parameter sweeps once the diagnostic script exists.

## Acceptance Criteria

Root cause identified with a written ticket comment + minimal reproducing test in the cffg repo; either a fix (engine/model-pattern) or a documented modeling recipe (absorbing modes, check budgets) that gets GFE RockSample to check-then-sample behavior beating the BFE ablation.

