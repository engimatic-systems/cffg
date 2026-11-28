---
id: cff-wi2o
status: in_progress
deps: [cff-ees0]
links: []
created: 2026-09-25T03:46:26Z
type: task
priority: 1
assignee: coopwilliams
parent: cff-xa6z
tags: [pcgym]
---
# Specify valid resampling and rejuvenation for nonlinear parameter particles

## Background in Plain English

The PCGym bank resamples survivors, jitters them and resets weights, including an elite hypothesis. That is a useful search heuristic, but not proof of a Bayesian continuous posterior. A first-class particle feature needs an explicit target, weight correction and evidence-accounting contract.

Evidence comes from the PCGym report at `/Users/ravd/agent-share/PCGymCFFG/` and experiment code at `/Users/ravd/experiments/cffg-testing/pc-gym/00/cffg/`, reviewed against CFFG `fce60bd`. Copy only the minimal reproducible inputs into maintained fixtures; do not add external training artifacts or pickle dependencies.

## Outcomes / Acceptance Criteria

1. The design distinguishes an exact fixed-bank posterior, unbiased resampling of a weighted empirical measure and approximate continuous posterior inference.

2. A declared prior and likelihood define the continuous target, including parameter transforms, Jacobians and constraints.

3. Any rejuvenation kernel has a justified invariant target or explicit importance-weight correction; elite retention and uncorrected jitter are not silently called Bayesian.

4. The design specifies retained-data/replay needs, online evidence accounting, effective sample size, explicit random keys, persistence, failure and degeneracy semantics.

5. Independent low-dimensional posterior/evidence oracles and calibration checks cover non-identifiability, zero support, float32 and larger particle sets.

6. DERIVATIONS contains equations with a plain-English companion and executable small tests; the implementation ticket receives a bounded v1 scope and acceptance tolerances.

## Methodological Suggestions

Do not copy the experiment's _rejuvenate as a statistical guarantee. Architecture: Learnable or checkpoint feature; new model fields only if required by target identity. Pairs host/JIT target agreement; flooding/scheduled likelihood agreement; single/vmap particle evaluation; current/legacy checkpoint identity. Background research goes to subagents with specific mathematical questions.

These are suggestions; another method is acceptable if it satisfies the outcomes. Work branches continue from the preceding validated PR in the PCGym chain, whose initial base is `cff-ucvx-readme-capability-map` (PR #161). Update the intended base before branching. Code changes require focused tests and one passing final short suite; documentation-only work validates its changed artifacts.

## Future work

- cff-iplr — Implement the validated nonlinear parameter-particle learning contract.

## Definitions

Particle: a weighted parameter hypothesis. Resampling: drawing particles according to weights. Rejuvenation: moving particles to improve coverage while respecting a target distribution. Effective sample size: a weight-concentration measure. Jacobian: the volume correction for a parameter transformation. Calibration: agreement between uncertainty claims and repeated outcomes.

Shared terms: CFFG means constrained Forney-style factor graph; JAX is the numerical array/compiler library; dtype is a numeric storage type; free energy is the model's inference objective; host/JIT compares Python-side and compiled execution; single/vmap compares one instance with vectorized instances; N/A means not applicable.

## Notes

**2026-11-28T04:58:11Z**

Change paths: Learnable or checkpoint feature and Runtime numerical or status invariant. This task is a mathematical contract with executable reference tests. The runtime ticket owns host/JIT, single/vmap, schedule likelihood parity, and versioned state. Before scoring: use a probability in (0,1) with a beta prior, a bounded nonlinear observation probability, exact polynomial integration, seeded repeated runs, vacuous observations, population collapse and 10× population size. No empirical PCGym search result is a posterior oracle.

**2026-11-28T05:01:55Z**

Question: does applying an invariant move only at low ESS preserve finite-population evidence? Independent review and a two-state exact enumeration give expected normalizer 1/4 instead of 1/2. The contract uses the same predetermined posterior-invariant kernel in both ESS branches; stratified draws have unbiased offspring counts. Sources: PCGym common/physics_agent.py _rejuvenate (elite plus jitter), D6d induction and test_static_particle_derivation.py. The zero estimate from an exhausted population is distinguished from analytic impossibility. Continuous runtime implementation remains cff-iplr.
