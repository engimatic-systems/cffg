---
id: cff-4oq8
status: closed
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

## Notes

**2026-07-15T03:35:05Z**

Planning reconciliation (2026-07-14): this ticket predates the merged cff-hp3y cavity/ledger repair and cff-j4mu truthful RunResult statuses. cff-2gev depends on this suite; it is not a blocker. The suite will therefore use the repaired exact observation-GFE path and assert public convergence statuses.

Hostile-input checklist before implementation:
- Sparse/zero support: the decision model deliberately uses exact-zero deterministic transitions and likelihood support; every actionable run must converge with finite messages/FE.
- Vacuous/improper inputs: runs begin from Engine.init_state() (vacuous messages); malformed categorical arrays remain compiler validation, not a behavioral oracle.
- float32: one end-to-end decision case will run with x64 disabled and preserve the qualitative action ordering/status.
- 10-100x scale: no new numerical rule is introduced; GFE Newton already has K=80 scale coverage in test_gfe. This suite instead varies sensor quality/cost across a parameter grid on the minimal binary Bayes decision, so enlarging the latent domain would change the intervention rather than stress a new kernel.

Oracle: a standalone binary environment supplies the hand-computed value-of-information threshold and realized return/regret. Posterior MI and realized posterior KL are computed from normalized probabilities, never from engine messages or GFE rule internals. Acceptance is behavioral: GFE/BFE action and return differences, relevance, satiation, monotone quality/cost responses, finite FE, and explicit RunStatus.

**2026-07-15T03:54:25Z**

Completed cff-4oq8 (2026-07-14 local).

Implementation:
- Added cffg.experiments.behavioral_epistemics: serialized-source two-step cue/distractor task, independent binary Bayes threshold, normalized-joint MI/posterior-KL oracle, simulator, explicit DecisionPlan, and GFE/BFE BehavioralReport.
- Added 7 adversarial tests covering realized return/regret across 6 seeds x 256 episodes, relevance, satiation, cost crossover, sensor quality, preference offset/scale calibration, exact-zero support, vacuous initialization, float32, finite FE, and RunStatus.
- Updated README, ARCHITECTURE H5 test boundary, LIMITATIONS S1c exact-observation/transition-bound distinction, and dashboard M4 routing.

Measured acceptance:
- GFE u1=[.2062,.2062,.3208,.2668] chooses cue; BFE u1=[.3293,.3293,.1707,.1707] guesses an arm. Cue-conditioned replans choose the indicated arm.
- GFE mean return=.9648 versus BFE=-.0078; every GFE seed beats its BFE counterpart by >.65 utility. Bayes expectation=.95.
- Cue posterior MI and realized first-step IG=.3681 nats; distractor MI=0. Direct-arm MI=.6931 but arrives with the terminal reward, which is why reporting return alongside information is load-bearing.
- Repeat-cue probability falls after observation; exploit mass rises. Sensing probability decreases strictly with cost and switches between c=1.20 and 1.30 around the hand Bayes threshold c*=1.20.
- Exact observation epistemics weakens strictly with sensor quality and equals the random distractor posture at q=.5. High goal contrast eventually dominates curiosity.
- Full suite: 1138 passed, 4 existing dependency warnings, 237.45s. Dashboard refreshed: 1138 tests, 0 failing, 7 new.

No engine rule fix was needed after the merged cff-hp3y/cff-j4mu repairs.

**2026-07-15T04:05:23Z**

Post-fetch verification correction (2026-07-14 local): the branch was initially cut from a stale local main. After fetching, it was rebased cleanly onto origin/main@52b43d4, which includes the merged cff-kut1 one-pass dashboard workflow and cff-44we arena segmentation. The authoritative current-base landing command was uv run python tools/dashboard.py --run-suite: 1168 passed, 4 existing dependency warnings, 552.34s; dashboard reports 1168 tests, 0 failing, 30 new. This supersedes the earlier 1138-test pre-fetch count.

**2026-07-15T04:05:55Z**

PR: https://github.com/engimatic-systems/cffg/pull/30
