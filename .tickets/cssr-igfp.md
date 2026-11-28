---
id: cssr-igfp
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
# Build replayable add, split, prune and merge candidates

A proposed edit changes the number or connectivity of latent states. A change to a prior can score a possible edit, but a ranked structural candidate needs an actual graph that can emit the same observations.

## Design

Cover items 17.1–17.13 using the reviewed equations in cff-du6m. Reuse the AXIOM mixture and reduction code where its statistical assumptions apply. Distinguish equivalent changes of labels, changes of prior on the same model, and edits that require scoring a new likelihood. Use the shared replay record. Architecture rows: New IR field and Learnable or checkpoint feature if an edit transports fitted parameters; mark unrelated host/JIT or checkpoint surfaces explicitly.

## Acceptance Criteria

On small independently scored fixtures, produce valid add-regime and split-state edits, prune a Dirichlet state, Gaussian edge and Gaussian regime, and merge two Dirichlet states. Report invalid input as a structured rejection. Check zero-support donors, improper priors, genuine f32 and 10× cardinality. Preserve observation alphabet across a comparison, track fitted arrays and reproduce each candidate from its record.
