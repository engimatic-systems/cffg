---
id: cff-mjku
status: open
deps: [cff-9xmq]
links: [cff-b47g]
created: 2026-07-10T03:31:28Z
type: feature
priority: 3
assignee: coopwilliams
---
# Modeling ergonomics: fan-out primitive + observation branches for interior latents

## Plain English

This ticket turns two common graph edits into reusable operations: sharing one
variable with many factors, and attaching an observation port to a latent
variable inside a graph. Model builders can request either structure directly
with one declarative call.

Two friction points every large model hits: (1) a variable consumed by >2 factors forces the modeler to hand-build ternary equality chains (a 16-leaf class fan-out inside one fragment took 15 hand-written equality factors) — promote the Broadcast synthesis into a general fan_out(variable, n) template helper (balanced tree per cff-9xmq); (2) observing an interior (degree-2) latent requires manually inserting an equality to open a clampable DATA branch (e.g. a supervised label on a latent class edge) — add an edit action / builder helper open_observation_branch(variable) that does the equality+port dance.

## Acceptance Criteria

Both helpers in core/edits or compose with tests; the 16-leaf fan-out expressible in one call; supervised-clamp-on-latent expressible in one call; validate/serialize round-trip.
