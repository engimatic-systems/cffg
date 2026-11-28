---
id: cssr-5izj
status: closed
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

## Notes

**2026-11-28T06:45:55Z**

Host archive layer using existing Template, typed-port composition, model bundles and proposal replay. The New IR field change-path row was checked; fragment meaning/provenance belongs to a separate host record. No JIT/flood/scheduled, Colony or learning-checkpoint duplication. Validate numeric content, port roles/domains and category meaning after a fresh-process reload, ten-fold reuse and failed composition. Keep observed DATA distinct from open glue and report each of the three examples as a declared conditional model.

**2026-11-28T06:50:52Z**

The three fixed binary examples, portable archive and GLUE attachment API are implemented in cffg.structure_fragments. IDs include complete Template, source, description, named ports with ordered outcomes and exact numeric content. All three loaded and composed in a fresh interpreter. Native posterior and ordinary Bethe evidence matched independent probabilities for hit, miss and missing readings and both selector values. A ten-slice persistence graph retained all ten sensors and an unrelated host DATA port. Swapped numeric contents, altered dtype, reversed labels, incompatible role/domain and wrong expected ID were exercised. 39 focused fragment, proposal, edit, compose and serialization tests passed. No application or scoring campaign ran.

**2026-11-28T13:53:01Z**

Implemented at 1354a21. All 39 focused checks passed, including fresh-process restore, ten composable copies, and native posterior and energy checks for each of the three fragments. The serial dashboard gate passed all 2,820 selected tests in 1,202.498 s. The shared 18 opt-in marks from d71952b (local fc35a77) cover 14.691 measured case seconds, giving an estimated 1,187.807 s. Collection verified 2,802 short plus 528 slow, all 3,330 IDs retained. Dashboard and per-ID durations are in ~/agent-share/cffg-backlog-gates/cssr-5izj/.
