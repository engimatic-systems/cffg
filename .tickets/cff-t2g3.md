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

