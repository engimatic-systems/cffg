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
