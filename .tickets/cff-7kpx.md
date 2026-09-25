---
id: cff-7kpx
status: open
deps: [cff-04mb]
links: []
created: 2026-09-25T03:46:25Z
type: task
priority: 1
assignee: coopwilliams
parent: cff-xa6z
tags: [pcgym]
---
# Capture and classify the RK4 nonlinear CSTR planning failure

## Background in Plain English

The experiment reports failure on every nonlinear CSTR plan even with the true constants, sensitive to goal and process precision. The exact failing invocation and state were not retained. A smaller cubic script has a wrong parameter shape; after correction it converges, so it cannot justify a solver repair.

Evidence comes from the PCGym report at `/Users/ravd/agent-share/PCGymCFFG/` and experiment code at `/Users/ravd/experiments/cffg-testing/pc-gym/00/cffg/`, reviewed against CFFG `fce60bd`. Copy only the minimal reproducible inputs into maintained fixtures; do not add external training artifacts or pickle dependencies.

## Outcomes / Acceptance Criteria

1. A compilable self-contained fixture records the registered physical function, constants, observations, initialization, process covariance, goal covariance, dtype and iteration settings.

2. The reported failure is reproduced, or a bounded investigation records precisely which missing inputs prevent reproduction without substituting the invalid cubic artifact.

3. The first failed cavity and its spectrum are captured across a declared grid of process/goal precision ratios for Taylor and sigma-point approximations.

4. Evidence distinguishes transient initialization/order failure, an unsupported improper cavity and non-finite physical dynamics.

5. An affine limiting case agrees with exact linear-Gaussian inference; sparse/vacuous information, float32 and a larger domain are part of the investigation.

6. The downstream repair ticket receives a concrete mathematical admissibility contract, retained fixtures and proposed acceptance thresholds before implementation.

## Methodological Suggestions

Use common/physics.py:nl_window_chain and the CSTR RK4 registration; request no entire training rerun. Architecture: Runtime numerical or status invariant and New factor or rule family for admissibility analysis. Pairs: host/JIT local status comparison; flooding/scheduled initialization comparison; single/vmap local rule comparison; checkpoint N/A. Do not claim global posterior properness proves every local cavity proper.

These are suggestions; another method is acceptable if it satisfies the outcomes. Work branches continue from the preceding validated PR in the PCGym chain, whose initial base is `cff-ucvx-readme-capability-map` (PR #161). Update the intended base before branching. Code changes require focused tests and one passing final short suite; documentation-only work validates its changed artifacts.

## Future work

- cff-toqa — Implement the evidence-backed nonlinear planning admissibility contract.

## Definitions

CSTR: continuous stirred-tank reactor. RK4: fourth-order Runge-Kutta integration. Precision: inverse covariance. Spectrum: matrix eigenvalues. Taylor/sigma-point approximations: local methods for propagating uncertainty through a nonlinear function. Affine: a linear map plus an offset.

Shared terms: CFFG means constrained Forney-style factor graph; JAX is the numerical array/compiler library; dtype is a numeric storage type; free energy is the model's inference objective; host/JIT compares Python-side and compiled execution; single/vmap compares one instance with vectorized instances; N/A means not applicable.
