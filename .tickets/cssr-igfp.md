---
id: cssr-igfp
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
# Build replayable add, split, prune and merge candidates

A proposed edit changes the number or connectivity of latent states. A change to a prior can score a possible edit, but a ranked structural candidate needs an actual graph that can emit the same observations.

## Design

Cover items 17.1–17.13 using the reviewed equations in cff-du6m. Reuse the AXIOM mixture and reduction code where its statistical assumptions apply. Distinguish equivalent changes of labels, changes of prior on the same model, and edits that require scoring a new likelihood. Use the shared replay record. Architecture rows: New IR field and Learnable or checkpoint feature if an edit transports fitted parameters; mark unrelated host/JIT or checkpoint surfaces explicitly.

## Acceptance Criteria

On small independently scored fixtures, produce valid add-regime and split-state edits, prune a Dirichlet state, Gaussian edge and Gaussian regime, and merge two Dirichlet states. Report invalid input as a structured rejection. Check zero-support donors, improper priors, genuine f32 and 10× cardinality. Preserve observation alphabet across a comparison, track fitted arrays and reproduce each candidate from its record.

## Notes

**2026-11-28T06:38:00Z**

Implementation handoff at b76ef27. For a hidden categorical selector connected to one ordinary emission factor, resize its prior and the matching emission axis together. Split and merge transport concentration mass and preserve the observed alphabet. Actual Gaussian predictor deletion must drop the dependence in the graph. Prior-only Bayes factors, held-out scores for edited likelihoods and parameter initialization have distinct meanings. Reuse the retained proposal inputs and immutable edit bundles. Architecture change paths: model construction and parameter ownership. Schedules, batched kernels and checkpoint formats consume ordinary replayed Templates and need no edits. Exercise sparse donors, nonpositive priors, native f32/f64, and at least 10x cardinality against hand-computed distributions.

**2026-11-28T09:19:55Z**

A caller can propose a discrete state split or merge, a state or regime prune, Gaussian regime growth, or deletion of an affine Gaussian input. Each accepted edit identifies its parent, serialized operations and numeric inputs and preserves the observed alphabet. Rejected proposals carry a code and reason. The scorer can fit and compare each reconstructed graph on retained data.

The 26 focused checks cover mass conservation, duplicate emissions, zero support, genuine float32 and float64, a 40-regime mixture, and normalized Gaussian observations after deleting an input. Combining two different emission columns preserved a single-observation mean while changing the two-success probability from 7/24 to 1/4 in the Dirichlet fixture. This is why the changed likelihood needs scoring. The compact interface takes fixed or explicitly fitted source parameters.

The required short suite passed 2,821 cases in 1191.174 seconds at 660fefed7c22.
