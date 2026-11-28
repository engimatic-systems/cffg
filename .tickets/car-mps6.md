---
id: car-mps6
status: in_progress
deps: [car-k6oi, car-8rgl, cff-pksr, cao-0pbk]
links: [cff-du6m]
created: 2026-11-23T21:48:38Z
type: feature
priority: 2
assignee: coopwilliams
parent: car-zieb
tags: [axiom]
---
# Learn AXIOM motion, interactions and rewards and reduce redundant rules

## Background in Plain English — WHY

An object can move the same way for many frames, then bounce or disappear when another object comes close. AXIOM keeps a small catalogue of motion rules and learns which interaction predicts the next rule and reward. It periodically combines redundant interaction rules. CFFG’s existing switching Gaussian factors provide part of the numerical machinery.

## Outcomes / Acceptance Criteria — WHAT

1. From a causal object tape, score active affine motion models (tMM), infer the motion class and initialize a new affine rule from a novel observed transition. Preserve the release’s preseeded rules and the fixed matrices of existing rules. Verify matched motion, new motion, repeat/reuse and exhaustion on independent small cases.
2. Implement the recurrent interaction mixture (rMM) using the specified object and neighbor features, identities, executed action, observed motion class and realized reward. During prediction omit the unknown switch and reward from component selection and predict both from the resulting distribution. Match release ordering for tracked objects sharing a learner; check the soft assignment, selected writes (including the responsibility and count cutoffs) and every head’s counts against a trace. Compare default/tuned neighbor construction when the tuned variant is selected.
3. At the specified 500-frame cadence generate the reference pseudo-observations, retaining the earlier generated batches as in the release. Propose at most 2,000 eligible pairs and evaluate them sequentially on the same sample bank within a pass. Select the component from input fields, score the held-out reward and switch fields and subtract the mixing-weight Dirichlet KL. Accept a merge if its score is at least as good as the current score. Log proposed pairs, both scores and active counts. This score has its own name and units, separate from the CFFG graph’s free energy.
4. Trace a novel contact, its learned reward or switch and its reuse on the next contact. Demonstrate accepted and rejected merges, donor reset, post-merge prediction, and exact replay from a checkpoint. Include impossible categories, vacuous evidence, full capacity, float32/64 and the reference 500 motion/5,000 interaction slots. A failure returns a usable diagnostic and does not corrupt the preceding model.

## Methodological Suggestions — HOW

Use cff-pksr for switching Gaussian inference, car-k6oi for transactions and car-9ccf for mixed evidence. Begin with injected object tracks from car-8rgl so pixel inference can develop in parallel. cff-du6m covers the paper’s reduction math; compare the released partial output score against a hand-worked example in which whole-graph scoring would give a different answer. Review “Learnable or checkpoint feature” and any new rule’s host/JIT, single/batched and schedule owners. Qualify prediction and real-data update separately.

## Future work

Jointly refitting every affine matrix and using a general model-search objective would be new algorithms. Track them as variants if pursued after reproducing the released rule.

## Definitions

An affine rule maps the previous state to a matrix times that state plus an offset. A switch chooses a motion rule. A pseudo-observation is drawn from the current model. KL divergence measures a difference between probability distributions.

## Notes

**2026-11-24T06:00:54Z**

Live frames now train all motion and interaction components, with delayed identity writes and actual applied action and reward. The learned sparse rollout bridge is separately qualified. The released scale calculation includes a global diagonal repair after sparsifying Gaussian evidence. Training, rollout and reduction must share its result across all components; sampling uses the distinct unshifted projection. The scheduled model reduction and its assembled integration remain their own gate.
