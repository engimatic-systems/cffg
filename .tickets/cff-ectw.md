---
id: cff-ectw
status: open
deps: [cff-04mb, cff-vxza, cff-03dn, cff-u9un]
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

## Notes

**2026-09-25T07:25:29Z**

Research question -> answer -> source -> confidence: Why does the saved seed-1 undamped hybrid plan fail? Current mixed-schedule replay diverges at sweep 2 with nonfinite_objective, unknown factor, ledger phase; pristine damp=.5 converges at sweep 69, FE 259.09087910280937. Individual ledger inspection identifies goal15 (+inf) and interior x15 precision diag(400,0). dyn14 receives rank-one diag(400,0); Schur reconstruction yields roundoff eigen 4.38e-16, strict _pd_ok admits it; collapsed covariance eigenvalues [-0.00333657,1.66532e15] fail propriety, yet _pd_inverse safe identity escapes as precision 6.083413961e-16*I. eq14 total-minus-self changes this to4.4408920985e-16, and dyn15 later returns zero precision. Damping changes the same rank-sensitive placeholder branches; it is not proof of a clean arithmetic path. Source: read-only pcgym_numerics_review replay, mixg.py forward/backward/_collapse, gauss.py Schur/equality, efrules.py safe inverse, ledger_terms.py Gaussian node and interior-edge terms. Confidence high from measured intermediates; last-bit eigenvalues diagnose conditioning, not mathematical rank. Split before implementation into cff-vxza exclusion sums, cff-03dn rank/collapse validity, and cff-u9un initialization/safeguarded control. No noise-floor change is authorized as a repair.

**2026-09-25T07:25:52Z**

Refined dependency map from tk dep tree --full, tk ready and tk blocked. Legend: prerequisite --> dependent; [R] ready, [B] blocked, [x] complete.

[x] cff-04mb diagnostics
  +--> [R] cff-03dn mixture rank/collapse ----+
[R] cff-vxza additive exclusion sums -------+--> [B] cff-u9un safeguarded solve
  (independent arithmetic work)                  --> [B] cff-ectw completion
                                                    --> cff-wdhj agent facade
                                                    --> cff-xa6z roadmap

Both arithmetic tickets and cff-u9un are explicit prerequisites of cff-ectw. Other independent PCGym roots remain ready. tk dep cycle reports no cycles.

Follow-up research: actual dyn_j full-rank input first arrives after j-1 mixed sweeps (dyn1 at priming0, dyn15 at14; diagnostic threshold16*d*eps*lambda_max). Existing prime/reconstruct traverse exact islands, not mixture factors. Correct rank exclusion should leave the early ledger undefined until proper prior information arrives; importantly, IterationProgress already allows initial undefined Gaussian FE while previous_fe=None, so no new initialization policy is justified until arithmetic repair is tested. Source: pcgym_numerics_review traced saved graph and inspected iteration.py/mixed schedule. Measured rank-arrival confidence high; behavior after corrected arithmetic remains an inference to validate in cff-u9un.

**2026-09-25T07:26:21Z**

Published refinement PR #169: https://github.com/engimatic-systems/cffg/pull/169, based on #168. Numerical investigation is complete; implementation starts with independent cff-vxza while rank/collapse contract review informs cff-03dn.
