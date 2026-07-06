---
id: cff-t2g3
status: in_progress
deps: []
links: []
created: 2026-07-05T23:57:38Z
type: task
priority: 3
assignee: coopwilliams
parent: cff-21ta
---
# Optimize glucose benchmark with CFFG agent

Replace ReactiveController with a receding-horizon CFFG agent (discretized BG/IOB/carb states) once M5 runtime lands. Target: beat 87.6% TIR and cut hypo below 1%.


## Notes

**2026-07-06T00:06:03Z**

THE BAR (from Cooper, 2026-07-05): his VERSES-era hand-tuned experiments = 67% TIR on unannounced ADULTS + ADOLESCENTS, 0 critical episodes, 8-virtual-hour episodes. Goal: beat that with a continual-learning white-box CFFG agent + modular physiological world model. Current reactive controller (87.6%) is NOT comparable (adults-only, 24h, no critical-episode metric). Do not spend effort optimizing until CFFG core is done. When starting: extend benchmark with adolescent cohort, 8h episodes, critical-episode counter (BG<54 or <40 thresholds - confirm definition with Cooper).

**2026-07-06T04:43:01Z**

Experiment branch simglucose-experiment: added src/cffg/experiments/simglucose_basal.py for Cooper's 8h adults+adolescents basal-only unannounced-meal protocol, plus numbered run artifacts under experiments/simglucose-basal-2026-07-06. Best run is 005_personalized_v2_full: 86.66% TIR, 0.00% low, 13.34% high, 1.16% critical time, 37 critical samples. This beats the 67% TIR / zero-critical historical bar on TIR and achieves zero hypo, but not zero criticals because remaining failures are hyperglycemia in adult#004, adolescent#005, adolescent#007. Final targeted pass found stronger basal trades hyperglycemia for lows before approaching 94%, so this looks like the basal-only frontier for this simple white-box controller. Tests: uv run pytest; dashboard refreshed at ~/agent-share/cffg-tests.html (108 tests, 0 failing, 2 new).

**2026-07-06T16:49:12Z**

Randomized-meal follow-up on simglucose-experiment: added randomized 100-episode protocol (balanced 5 episodes per adult/adolescent virtual patient), variable hidden meal times/grams, per-trial meal metadata in artifacts, and two additional controller families. Results with seed 20260706: exp008 personalized-v2 under randomized meals = 80.02% TIR, 9.25% low, 4.36% critical time; exp009 randomized-hazard-v1 = 79.91% TIR, 4.56% low, 3.86% critical time (best TIR/safety tradeoff); exp010 reactive-physiology-v1 = 72.38% TIR, 15.08% low, 10.87% critical time; exp011 randomized-hazard-safe-v1 = 72.58% TIR, 0.17% low, 5.91% critical time (safety frontier, mostly hyperglycemia criticals). Conclusion: exact meal-clock policy was overfit; broad hazard prior is more robust, but basal-only global tuning cannot hit 94% or zero criticals under random meals. Next modeling step should learn patient-specific insulin sensitivity and carb absorption states, with a hard safety mask. Tests: uv run pytest = 110 passed; dashboard refreshed at ~/agent-share/cffg-tests.html.
