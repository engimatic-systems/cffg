---
id: cff-olvl
status: closed
deps: [cff-1d6t]
links: []
created: 2026-07-05T22:12:32Z
type: epic
priority: 2
assignee: coopwilliams
---
# M5: Agent runtime + colony execution

Receding-horizon loop (constraint flip goal->data, glue new slice), continual learning posterior->prior, checkpointing; Colony layer: instance table, structure-hash batching via vmap, host port routing. Demo: gridworld + N-instance colony benchmark.


## Notes

**2026-07-06T01:37:12Z**

M5 COMPLETE: RollingAgent (infinite-horizon receding loop, 2 engines/agent, checkpointing, optional continual A-learning) + Colony (structure-hash batching, vmapped ticks, host routing) + gridworld second model. The learning-goo execution pattern is real: spawn many templates, tick them batched, couple through ports.
