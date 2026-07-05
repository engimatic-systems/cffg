---
id: cff-21ta
status: closed
deps: []
links: []
created: 2026-07-05T23:57:38Z
type: epic
priority: 2
assignee: coopwilliams
---
# SG: simglucose TIR benchmark (side project)

Unannounced-meal benchmark per the overnight agent-share report: fixed 3-meal scenario, Dexcom+Insulet, 24h, 10 adults, BG-based TIR>=70% (FDA bound). Landed: gym-3.12 shim, benchmark harness, ReactiveController (CGM trend meal detection + quest-scaled corrections + hypo suspend) at cohort TIR 87.6%, low 1.7%. References: announced-BB 95.4% ceiling, basal-only 33.5% floor, prior pymdp 74.7%. LATER: optimize with a CFFG agent (rolling-horizon glucose model), tighten hypo on adult#006 (5.8% low).


## Notes

**2026-07-05T23:57:38Z**

Harness + reactive controller landed and green; epic stays open for the optimization stage (cff-t2g3).
