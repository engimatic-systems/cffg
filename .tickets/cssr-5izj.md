---
id: cssr-5izj
status: open
deps: [cssr-h31p]
links: []
created: 2026-11-28T05:28:14Z
type: task
priority: 2
assignee: coopwilliams
parent: cff-cynx
tags: [structure-learning]
---
# Serialize typed fragments for object persistence, contact and agency

A fragment is a reusable graph with named connection points. Proposals need enough provenance to reconstruct exactly which version was inserted.

## Design

Cover items 21–24. Store typed ports, source and a stable content identifier with each fragment. Build the three named examples with existing Template and compose APIs. Architecture row: New IR field if serialization changes. Check save/load, typed ports, scoped composition and IDs. Pure host graph assembly requires no separate JIT message, scheduling or learning-checkpoint path.

## Acceptance Criteria

A fresh process loads all three examples, validates their graphs and composes them with compatible ports. Incompatible ports fail clearly and cannot alter the host graph. Test ID stability across save/load, distinct content, and composition at 10× the smallest example. Document what each fragment asserts about its observations.
