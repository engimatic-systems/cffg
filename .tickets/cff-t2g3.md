---
id: cff-t2g3
status: open
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
