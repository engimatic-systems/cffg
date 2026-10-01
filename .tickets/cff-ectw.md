---
id: cff-ectw
status: open
deps: [cff-04mb]
links: []
created: 2026-09-25T03:46:25Z
type: bug
priority: 1
assignee: coopwilliams
parent: cff-xa6z
tags: [pcgym]
---
# Stabilize the reproduced hybrid planning failure without changing its model

## Background in Plain English

The saved nonsmooth-control seed-1 model diverges after two undamped sweeps from the first evaluation state; restarting that same state with damping 0.5 converges after 69 sweeps. The noise covariance is positive definite. Changing its noise floor changes the model and is not a solver repair.

Evidence comes from the PCGym report at `/Users/ravd/agent-share/PCGymCFFG/` and experiment code at `/Users/ravd/experiments/cffg-testing/pc-gym/00/cffg/`, reviewed against CFFG `fce60bd`. Copy only the minimal reproducible inputs into maintained fixtures; do not add external training artifacts or pickle dependencies.

## Outcomes / Acceptance Criteria

1. A small repository fixture reproduces the saved failure without PCGym installation, training, or pickle loading.

2. The first failed factor or matrix is identified and the mechanism is documented using the new failure diagnostics.

3. A safeguarded path converges on the retained fixture with finite posterior moments and free energy while retaining the original parameters and objective.

4. Any automatic retries start from a documented pristine state, have finite attempt/iteration budgets, and expose attempts and final failure when none succeeds.

5. Neighboring parameter values, sparse selectors, vacuous/improper messages, float32 and float64, and a ten-times-larger horizon are covered with truthful statuses.

6. Existing convergent hybrid fixtures retain posterior and free-energy agreement within preregistered dtype-aware tolerances; Engine and Colony behavior agree.

## Methodological Suggestions

Compare update ordering, local conditioning, and damped candidates before choosing a method. No unconditional claim that damp=0.5 works for all models. Architecture: Runtime numerical or status invariant; New factor or rule family only if equations change. Pairs: host/JIT, flooding/scheduled, single/vmap required; checkpoint N/A absent new persisted state.

These are suggestions; another method is acceptable if it satisfies the outcomes. Work branches continue from the preceding validated PR in the PCGym chain, whose initial base is `cff-ucvx-readme-capability-map` (PR #161). Update the intended base before branching. Code changes require focused tests and one passing final short suite; documentation-only work validates its changed artifacts.

## Future work

- cff-wdhj — Implement the continuous and hybrid rolling-agent facade.

## Definitions

Hybrid model: a graph containing both categorical and continuous variables. Damping: mixing an old message with an update. Positive definite: all eigenvalues are positive. Pristine state: the state before a failed attempt.

Shared terms: CFFG means constrained Forney-style factor graph; JAX is the numerical array/compiler library; dtype is a numeric storage type; free energy is the model's inference objective; host/JIT compares Python-side and compiled execution; single/vmap compares one instance with vectorized instances; N/A means not applicable.
