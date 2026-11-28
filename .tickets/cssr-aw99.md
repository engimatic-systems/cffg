---
id: cssr-aw99
status: open
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
