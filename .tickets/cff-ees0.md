---
id: cff-ees0
status: closed
deps: []
links: [cpc-7tbv, car-k6oi]
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

**2026-11-28T04:04:57Z**

Pre-implementation. Architecture row Learnable or checkpoint feature. The PCGym hypothesis_batch demo ties equal selectors through moment-collapsed Gaussian mixtures and adds a positive observation covariance. Its selector can be exact while its ledger is approximate. The current library already supplies the normalized Gaussian node energy for a bound equality coordinate (gaussian_taps). Use this density per fixed candidate, with one explicit observation covariance and log-sum-exp over the bank. Publish an executable conditional Gaussian Template and compare Engine schedules, native vectorized energies and an independent density oracle. Host commit validates every row before publishing state. Exact zeros use explicit support; f32 and 100x candidate counts, underflow/recovery, invalid covariance, impossible likelihoods and missing rows are qualification cases. A versioned restart binds candidate order, Template and numeric carry. No prior finite-bank snapshot schema, general IR extension, stochastic update or topological change. Parent owns combined default-suite gate.

**2026-11-28T04:14:08Z**

Ready for the combined gate. cffg.finite_hypotheses.FiniteHypotheses owns an ordered candidate matrix, shared or per-candidate total observation covariance, reset prior, normalized log weights and cumulative log evidence. Its Gaussian update runs the same normalized point-density primitive as a CFFG Gaussian factor bound at an equality DATA tap. conditional_model exposes the executable conditional tree. The bank sums across candidates explicitly. Rows flagged missing contribute zero. Impossible evidence raises its own exception, and invalid batches leave the full carry intact. save/load write an atomic, digest-checked version-1 NPZ containing the conditional Template, all candidate values, covariance, prior, support, posterior and evidence. Individual conditional templates also round-trip through ordinary CFFGZ. The supplied prediction function remains the caller’s responsibility on resume. No jitter, resampling or continuous-parameter inference.

Validation on this branch with UV_PROJECT_ENVIRONMENT=/Users/ravd/projects/cffg/.venv PYTHONPATH=src uv run --frozen --no-sync: pytest -q tests/test_finite_hypotheses.py tests/test_readme.py passed 15. Full-density SciPy enumeration agreed in f32/f64 for correlated, unequal covariances. Separate native Engine flood and two_pass ledgers agreed with the absolute per-candidate log density. Compiled batch and eager sequential updates agreed. In a 200-row bank, 198 live alternatives underflowed at the weight readout and recovered after opposite evidence, with the exact zero still excluded, both uninterrupted and after restart. The PCGym two-candidate case returned posterior [.07963637612143895,.9203636238785611] and NLL 2.47262703346254 in f64 (f32 NLL error 8.64e-8). The standalone pendulum example selected the 1.0 m candidate. pytest -q tests/test_gaussian_tap_oracle.py tests/test_amortization.py tests/test_neural_training.py::test_bundle_exports_initializers_overrides_and_trained_weights passed 4 with 5 existing slow checks deselected. Whole-bank transactions are host-side; vmap and JIT batch the per-reading/per-candidate density, and separate host bank instances own separate posteriors. tk dep cycle and git diff --check passed. Parent owns the final short suite, combined PR and ticket closure.

**2026-11-28T04:33:20Z**

At 2bfe84b the final four-worker short suite passed all 2,855 tests in 484.55 seconds wall. This includes the restored log-weight evidence, sparse support, Gaussian tree and restart cases. The review branch is based on #208.
