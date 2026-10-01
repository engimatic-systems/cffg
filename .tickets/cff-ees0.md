---
id: cff-ees0
status: open
deps: []
links: []
created: 2026-09-25T03:46:25Z
type: feature
priority: 1
assignee: coopwilliams
parent: cff-xa6z
tags: [pcgym]
---
# Expose exact finite parameter-hypothesis updates as a supported library feature

## Background in Plain English

A bank of 64 physical parameter hypotheses worked in PCGym using one shared categorical selector and Gaussian likelihood factors. Its posterior matched independent enumeration. Model authors should not need to hand-build the equality tree and observed emissions. Exactness is conditional on the fixed candidate set and the declared likelihood model.

Evidence comes from the PCGym report at `/Users/ravd/agent-share/PCGymCFFG/` and experiment code at `/Users/ravd/experiments/cffg-testing/pc-gym/00/cffg/`, reviewed against CFFG `fce60bd`. Copy only the minimal reproducible inputs into maintained fixtures; do not add external training artifacts or pickle dependencies.

## Outcomes / Acceptance Criteria

1. A supported public builder or facade represents a fixed parameter bank using reconstructible Template declarations plus numeric data, without an external inference implementation.

2. Posterior weights and log evidence match independent finite enumeration on small fixtures.

3. Sequential updates match a single combined batch when each observation is counted once; reset and prior replacement are explicit.

4. Canonical carried weights remain in log space with explicit exact support; very unlikely live hypotheses can recover after contradictory evidence without probability-round-trip underflow.

5. Exact-zero prior support stays zero; impossible evidence has an explicit failure; float32 and banks 10-100 times larger than a small fixture are covered.

6. The API defines shapes, likelihood covariance ownership and which numeric candidate arrays are fixed or replaceable; invalid inputs fail before partial updates.

7. Serialization/bundle restart preserves candidate identity and accumulated evidence, with a standalone nonlinear-physics example.

8. Documentation says finite-bank exactness does not imply an exact continuous-parameter posterior or validate arbitrary jitter.

## Methodological Suggestions

Reuse common/graphs.py:hypothesis_batch's categorical/equality/mixture/DATA pattern without importing experiment code. Architecture: Learnable or checkpoint feature; New IR field only if a new persisted descriptor is necessary. Pairs host/JIT inference/readout; flooding/scheduled equivalence; single/vmap independent banks; current/legacy decode explicit if schema added. The experiment's moment-collapsed Gaussian mixture ledger is not automatically exact log evidence. Verify evidence separately; an exact per-candidate observed-Gaussian graph plus a finite log-sum is a valid alternative.

These are suggestions; another method is acceptable if it satisfies the outcomes. Work branches continue from the preceding validated PR in the PCGym chain, whose initial base is `cff-ucvx-readme-capability-map` (PR #161). Update the intended base before branching. Code changes require focused tests and one passing final short suite; documentation-only work validates its changed artifacts.

## Future work

- cff-wi2o — Specify valid resampling and rejuvenation for nonlinear parameter particles.

## Definitions

Hypothesis bank: a finite set of candidate parameter vectors with probabilities. Log evidence: log probability of observations under the candidate mixture. Equality tree: factors sharing one logical variable while respecting degree two. Finite enumeration: direct summation over every candidate.

Shared terms: CFFG means constrained Forney-style factor graph; JAX is the numerical array/compiler library; dtype is a numeric storage type; free energy is the model's inference objective; host/JIT compares Python-side and compiled execution; single/vmap compares one instance with vectorized instances; N/A means not applicable.

## Notes

**2026-09-25T03:47:40Z**

Research handoff (question -> answer -> source -> confidence): Does the experiment's exact selector posterior imply exact Engine free energy/evidence? No. Small K=2/B=1 check (means [-2,+2], prior [.3,.7], component variance .6, emission variance .4, observation .4) converges with selector [0.07963638,0.92036362] matching enumeration, but BFE=2.0654715530983245 while exact negative log evidence=2.47262703346254. Source: common/graphs.py:hypothesis_batch; LIMITATIONS S1d; independently executed small fixture. Confidence: reproduced. Do not relabel the collapsed-mixture BFE as exact evidence. An ordinary per-candidate observed Gaussian graph with A=0 emission and candidate-specific offset gives exact likelihood; finite log-space aggregation is a bounded alternative. Carry log weights plus exact support to avoid probability underflow irreversibly killing a hypothesis before later contradictory evidence.
