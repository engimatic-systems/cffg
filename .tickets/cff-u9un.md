---
id: cff-u9un
status: open
deps: [cff-vxza, cff-03dn, cff-04mb]
links: []
created: 2026-09-25T07:25:11Z
type: feature
priority: 1
assignee: coopwilliams
parent: cff-ectw
tags: [pcgym, numerics]
---
# Establish a proper hybrid planning start and bounded safeguarded solve

## Background in Plain English

The PCGym seed-1 hybrid planner diverges after two undamped sweeps. A pristine-state retry with damping 0.5 converges after 69 sweeps, but deeper inspection shows the path can depend on invalid mixture-collapse placeholders. After that arithmetic is repaired, initialization and solver defaults must be assessed from the actual model rather than from the accidental recovery.

The parent cff-ectw owns the complete outcome. This ticket owns only run control and end-to-end certification after its arithmetic prerequisites.

## Outcomes / Acceptance Criteria

1. Retain the saved 15-transition fixture as plain numeric model inputs, with no training, PCGym or pickle dependency.
2. Establish whether proper information must propagate through the hybrid chain before free-energy certification; document and test the chosen initialization contract without treating nonfinite free energy as convergence.
3. A safeguarded path converges on the original fixture without changing its parameters, noise or objective. Compare posterior/FE against independently validated local calculations and a fresh undamped residual certificate.
4. If automatic retries are used, they restart from the caller's pristine state, consume one finite total work budget, expose attempt diagnostics and provide an explicit opt-out. If a different method avoids retries, document why and retain equivalent bounded work/status evidence.
5. Initialization failures, infeasible support, explicit damping, zero/fixed budgets, hardening, functional updates, recurrence fitting and adaptive/GFE solvers have explicit policy boundaries; do not silently stack independent retry owners.
6. Engine, singleton Colony and genuine batched Colony agree. Mixed/flood schedules, neighboring parameters, sparse selectors, vacuous/improper inputs, float32/float64 and a tenfold horizon retain truthful statuses.
7. Preregister dtype-aware posterior/FE tolerances; preserve existing successful hybrid behavior and report unresolved cases honestly.

## Methodological Suggestions

Choose another sound method if it meets the outcomes. Follow the Runtime numerical or status invariant row of docs/ARCHITECTURE.md. Before implementation, list local kernels, host/JIT, flooding/scheduled/exact, Engine/Colony and single/vmap surfaces; checkpoints are N/A unless stored state changes. Keep numerical guards relative to dtype and scale. Continue the validated PCGym PR chain, update the intended base before branching, and run focused tests plus one final short suite.

One 0.5 retry repairs the original stored fixture but is evidence for a candidate, not a universal default theorem. Keep failed-attempt traces separate from a successful attempt so nonfinite history and across-restart FE jumps do not corrupt its certificate. Inspect mixed schedule priming before choosing a retry-only policy.

## Future work

- cff-ectw closes only after this and its arithmetic prerequisites pass.
- cff-wdhj consumes the stable policy for continuous/hybrid agents.

## Definitions

Damping blends an old message with its update. Pristine state means before any failed attempt or priming. FE is the model's free-energy objective. A residual certificate measures whether a returned state satisfies its undamped fixed-point equations. JIT is compiled execution; vmap batches instances; dtype is numeric storage precision. GFE is generalized free energy.



## Notes

**2026-09-25T07:25:30Z**

Retained-fixture source: common/graphs.py hybrid builder and common/hybrid_agent.py parameter installation in /Users/ravd/experiments/cffg-testing/pc-gym/00/cffg. Extracted saved seed-1 constants: A=[[0.9872294577831741,0.08338833118938865],[-0.175619222344668,0.7290274178747534]] repeated for three selectors; b=[[-0.004707853348770677,-0.08696721726375882],[-0.00017546460387765,-0.0004393965675551424],[0.004356924141015377,0.08608842412864852]]; Q=[[0.00011554271987053177,2.0576762787578282e-07],[2.0576762787578282e-07,0.00019275601503216264]] repeated for all three. Horizon15, m0=[1,-1], P0=1e-4*I, uniform selectors, scalar goal matrix[[1,0]], zero offset/observations, goal covariance[[0.0025000000000000005]]. Preserve these installed values exactly; original installation already included a1e-8 covariance floor before blocking. Float64, budget200,tol1e-6 reproduces 2-sweep failure versus69-sweep .5 recovery. Do not load pickle in maintained tests.

**2026-09-25T08:45:32Z**

Research replay after cff-61ag arithmetic repair: does the unchanged saved H15 model still need retries? No for retained f64 fixture. pcgym_numerics_review rebuilt from plain constants/source, no pickle/training: default undamped CONVERGED16, first finite FE at sweep15, two finite FE entries258.2408051611355, residual0, failureNone. Explicit damping0.5 CONVERGED69, first finite FE at15, final258.2408051533387, residual4.8439e-9, failureNone. Every Gaussian node readout proper; z0 about[.78692114,.17462167,.03845719]. High confidence local execution of original model, no claim about neighboring/f32/H150/Colony yet. Next implementation can retain this fixture and prove existing bounded initialization policy rather than introducing unnecessary retries. Earlier pre-repair damped FE259.0908791 was influenced by invalid intermediate arithmetic and is not the new correctness oracle.
