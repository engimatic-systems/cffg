---
id: cff-4oq8
status: open
deps: []
links: [cff-fxmj, cff-hp3y, cff-gul8, cff-cj5h]
created: 2026-07-10T18:05:28Z
type: task
priority: 1
assignee: coopwilliams
parent: cff-mwjo
tags: [adversarial-reviewer, tests, epistemics]
---
# Behavioral epistemics suite: relevance, satiation, cost crossover, and realized return

ADVERSARIAL REVIEWER RECOMMENDATION (2026-07-10). Existing observation-GFE coverage is stronger than rule goldens: LAIF posterior parity, T-maze cue-first/BFE-ablation, and a 30-trial reward-learning ramp establish genuine information-seeking in that canonical model. What remains untested is whether curiosity is task-relevant, calibrated, and self-terminating outside it. The external RockSample run is the warning: GFE checks on all 10 steps in every episode, earns -1 return and 0% success, while the no-GFE arm succeeds 62.5%; the model omitted check costs from preferences, but the engine has no regression tier that would expose this as an unacceptable curiosity/pragmatic balance. Transition-novelty tests are currently local or symmetric-count traces, and cff-hp3y invalidates their engine integration until fixed.

## Acceptance Criteria

After cff-hp3y: 1. Costly two-step cue task: GFE senses then exploits; BFE skips sensing; realized return/regret beats ablation across seeds. 2. Relevance intervention: equally noisy distractor observation independent of the reward-relevant latent is not preferred over an informative cue. 3. Satiation: after a sufficiently informative observation or accumulated counts, repeat-sense probability falls and exploit probability rises. 4. Cost crossover: chosen sensing frequency is monotone in cue cost and crosses a hand-computed Bayes-optimal threshold. 5. Sensor-quality/data-processing test: realizable epistemic value weakens with observation noise; distinguish exact observation GFE from the documented optimistic transition bound. 6. Preference-scale invariance/calibration cases prevent novelty from swamping goals indefinitely. 7. Report posterior MI, realized information gain, task return, convergence status, and BFE/GFE ablations; no implementation-restatement oracle.
