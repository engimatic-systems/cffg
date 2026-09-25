---
id: cff-u9un
status: in_progress
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

**2026-09-25T09:14:53Z**

Pre-implementation handoff: Runtime numerical or status invariant row. Arithmetic prerequisites are closed; unchanged retained H15 now converges under existing default policy, so begin with maintained numeric fixture, independent Gaussian-mixture message/ledger oracle and fresh undamped residual check, not a speculative retry owner. Required pairs: host/JIT, mixed/flood, Engine/singleton and genuine batched Colony; current/legacy checkpoint schemas N/A because no persistent state change is planned. Cover original H15 default/damped, initialization budgets including zero/fixed, neighboring noise and sparse selector support, f32/f64 truthful status, and tenfold horizon as opt-in long behavior. Preserve existing explicit damping, hardening, functional updates, recurrence and adaptive solver owners; inspect their policy boundaries before deciding whether production edits are needed. Preregister f64 message/posterior atol/rtol1e-8 and FE1e-7; f32 comparison tolerances scale with dimension/epsilon and solver residual, never label nonfinite FE converged. Stop and split if a newly discovered solver mechanism requires a separate repair.

**2026-09-25T09:19:27Z**

Important semantic boundary confirmed by independent NumPy oracle and research replay: H1, H15 and H15 with only terminal goal changed0to2 have exactly identical z0=[.7869211413656841,.1746216733306407,.038457185303675194]. Terminal z14 and FE change (258.240805161 to669.184104862), proving data were bound. Every backward dynamic component stays rank1 under scalar position goals and D3 deliberately emits vacuous messages. Arithmetic convergence therefore does not establish useful multi-step planning. New maintained reference computes the forward collapsed moments, local joint-mixture node energies and entropy corrections independently and gives FE258.2408051611357. cff-ulbz full-tilt EC was closed as not pursuing at Cooper direction, not implemented; do not revive it implicitly. User has been asked whether to document this limitation or investigate a separate partial-goal workstream. Current ticket remains numerical initialization/certification, with explicit limitation in README/LIMITATIONS and executable distant-goal invariance.

**2026-09-25T09:22:24Z**

Focused integration results so far: all11 non-float32 cases passed, including explicitly opted-in H150 (151 sweeps, first finite FE150) and H15 singleton/genuine batched Colony comparisons against independent NumPy moments/ledger. Iteration and run-strategy boundary files:38passed. New f32 neighbor oracle found posterior error about6.0e-5 above the provisional256eps bound; assessing cancellation/conditioning before changing acceptance. No production tolerance or status has been relaxed.

**2026-09-25T09:23:48Z**

Independent-oracle review (failure_provenance): correct for the retained rank1-goal fixture. Selector prior plus selector edge entropy cancel mixture categorical entropy, leaving expected transition NLL minus joint Gaussian entropy plus KL(rho||habit). At each interior state, equality contributes -H, three Gaussian edges +3H, observed goal -H plus NLL: net +H plus NLL. Terminal edge H cancels goal -H; prior -H cancels x0 edge H, leaving prior NLL. Joint conditioning uses [x,y] block covariance and observation [0,C]. Source: tests/hybrid_planning_fixture.py, D3 and direct graph incidence; high-confidence mathematical/code review, no production math calls in the reference.

**2026-09-25T09:27:33Z**

Final focused integration validation:13 passed including both opt-in slow tests (H150 and genuine H15 batched Colony). Shared iteration/run-strategy boundaries:38 passed. Mean/covariance/FE and fresh undamped residual checks retain their original bounds; only the binary float32 selector comparison gets an explicitly measured cancellation allowance, independently computed from Gaussian normalizers and stable likelihoods (two log-odds ULPs, sigmoid derivative bound1/4, plus rounding). Direct log-odds checks make that allowance observable; cff-d8jj owns stable evaluation and tightening it. No production convergence threshold, damping default or mathematical objective was changed. Partial-goal scope is explicitly documented and independently tracked in cff-c5qf pending user decision. Final short-suite gate follows.
