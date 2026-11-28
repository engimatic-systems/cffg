---
id: cssr-aw99
status: in_progress
deps: []
links: []
created: 2026-11-28T05:28:14Z
type: task
priority: 2
assignee: coopwilliams
parent: cff-cynx
tags: [structure-learning]
---
# Fit candidate models on training data and compare their held-out Bethe scores

Model comparison asks how well each model explains the same new observations. Reusing training observations or scoring a preference-based acting objective would answer a different question.

## Design

Cover items 1, 2, 5–14, 18–20 and 28–38. Begin with explicit candidates supplied by the caller. Split fitting and evaluation data by retained artifact identity and use the public ordinary Bethe evaluator for comparison. Group compatible candidates with Colony and verify per-candidate parity with Engine. Architecture rows: Learnable or checkpoint feature for fitted state and Runtime numerical or status invariant for status transport. Inspect single/grouped and flooding/scheduled results. Keep comparisons on the host and leave existing checkpoints readable.

## Acceptance Criteria

An analytically scored tree, label-permuted copy, and a loopy model with a visible approximation note exercise ranking. Every result identifies partitions, evaluation graph, raw energy and status. Invalid, infeasible, exhausted or diverged candidates have no finite rank. Test exact-zero support, empty evidence, true f32 versus f64 and 10× candidate count/cardinality. Prove held-out events never update the fitted state, and a score equals the public solver's own energy.

## Notes

**2026-11-28T05:43:59Z**

Use the Learnable or checkpoint feature and Runtime numerical or status invariant rows. Fit only declared Learnables from public Engine results, then freeze their written parameters. Score a disjoint held-out partition with the public sum-product Bethe ledger. Exercise single Engine and grouped Colony, exact/colored and flooding routes, real native float32/float64, impossible and vacuous observations, invalid inputs, exhaustion and divergence, and at least tenfold candidate/cardinality scale. Preserve raw native energies and status even when ineligible to rank. P-substitution and planning evidence are ineligible. New checkpoint or template decoding, stochastic keys and device-side ranking are outside this host orchestration.

**2026-11-28T05:51:28Z**

The 10 focused structure-scoring checks passed, including the optional flooding comparison, in 10.42 seconds. The learned-tree oracle computes digamma losses from the hand-counted posterior and compares held-out repetitions and reversed event order. A separately solved latent tree and its relabeled copy agree in real native float32 and float64. The true Gaussian loop without a location anchor retains the native failed status and undefined energy. A scale fixture ranks forty explicit candidates with twenty latent labels each and checks every total analytically. The round rejects overlapping artifact IDs and incompatible measured domains. All results are synthetic; identical observed values in independent artifacts are allowed. The canonical suite is next.
