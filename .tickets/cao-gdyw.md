---
id: cao-gdyw
status: in_progress
deps: [cao-urtu, cao-fs5s]
links: []
created: 2026-11-24T14:51:38Z
type: feature
priority: 2
assignee: coopwilliams
parent: car-zieb
tags: [axiom, evaluation]
---
# Run full-budget AXIOM decisions from an exactly restorable campaign actor

## Background

A 24-action smoke connected the scene learner and policy search. The campaign needs a persistent actor at the published budget, a single real-action clock and a checkpoint that can resume before or after a decision.

## Outcomes

Implement Actor(initial_rgb,game,actions,seed), choose, observe, exact state_dict/load_state_dict. Preserve the proposal, explicit planning key, pending action, model scene, and complete reduction history. A true reset consumes an observation with learning disabled and preserves the real clock. Run the scheduled reducer after each due real action. A rejected step preserves the preceding actor state.

Use all 512 policy strings, 32 imagined steps and 3 draws per string. Limit concurrent policies while retaining the keyed samples, separate reward and novelty, rankings, proposal and selected action. Qualify against existing unchunked output on a small learned development case, including zero tracking; check unequal score, checkpoint after choose, invalid inputs, and restart. Measure a first full-budget plan then warmed development ticks, peak workspace and trajectory occupancy. Bound memory as support grows and disclose workload-dependent latency.

## Method

Use the Runtime numerical or status invariant and Learnable or checkpoint feature paths. Validate compiled chunked against ordinary search, original against restored actor, including real and reset clocks. Flooding and graph message schedules do not participate. Model/OOM failures must not become a valid action. Keep the supported model learning and release score semantics intact.

## Future work

The parent launches and monitors the measured campaigns. This ticket owns the actor, its performance checks and downstream API handoff.

## Definitions

A pending action is one chosen by the actor whose observed transition has not yet arrived. A batch processes a subset of independent policy strings at a time. An exact restart restores every stored numerical word.


## Notes

**2026-11-24T14:51:44Z**

The campaign actor uses the seed-key scheme already qualified by the acted smoke. Gauss owns the reduction scheduler and archive; the actor will call those hooks. The parent owns remote or long measured runs.
