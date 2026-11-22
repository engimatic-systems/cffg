---
id: cff-wt1c
status: open
deps: [cff-toqa, cff-iplr, cff-fl7h, cff-wdhj, cff-csmz, cff-4yr6, cff-qtub, cff-21vm, cff-qu2j, cff-3kdk, cff-wbhu, cff-c5qf, cff-d8jj, opt-ewtf, cfp-lsxw]
links: [cff-d63w, opt-ewtf, cpt-ydse, cpt-j81k]
created: 2026-09-25T03:46:26Z
type: task
priority: 2
assignee: coopwilliams
parent: cff-xa6z
tags: [pcgym]
---
# Validate the PCGym follow-up chain with matched control and runtime comparisons

## Background in Plain English

PCGym results mix library behavior with information access, planner choices, model assumptions and concurrent CPU load. After the improvements land, a bounded comparison must show which capabilities and costs actually changed without treating every reward gap as a library defect.

Evidence comes from the PCGym report at `/Users/ravd/agent-share/PCGymCFFG/` and experiment code at `/Users/ravd/experiments/cffg-testing/pc-gym/00/cffg/`, reviewed against CFFG `fce60bd`. Copy only the minimal reproducible inputs into maintained fixtures; do not add external training artifacts or pickle dependencies.

The fixed-model reactor comparison is owned by cfp-lsxw and can be published when it finishes. Cite its recorded runs here. The other prerequisites govern the remaining comparison arms and final synthesis. Use cfp-okq2 evidence for any later claim about automatic production dispatch. Keep elapsed launch, warm executable reuse and reuse of a previous belief state separate.

## Outcomes / Acceptance Criteria

1. The comparison records exact revisions, hardware, seeds, model knowledge, setpoint preview, objective, horizon and compute budget.

2. Separate ablations isolate nonlinear planning, fixed/particle parameter uncertainty, bound-aware decisions and facade equivalence.

3. Runtime reporting separates setup, binding, inference, readouts, learning and total wall time with explicit synchronization and concurrency.

4. Numerical success includes status, posterior/energy oracle checks and failure counts, not reward alone.

5. Short deterministic smoke workloads run by default; long multi-trial experiments remain opt-in slow or external artifacts.

6. Results are summarized in this ticket and existing canonical documentation where capability claims change; no new repository session/report markdown is created.

## Methodological Suggestions

Reuse small retained fixtures before optional full PCGym seeds. Architecture: Runtime numerical or status invariant and Learnable or checkpoint feature consumer validation. Pairs host/JIT and flooding/scheduled where comparable; Engine/Colony/facade equivalence; current/resumed checkpoints. Include zero support, float32 and larger workload cases. Do not interpret PPO's end-to-end runtime as a matched inference microbenchmark.

These are suggestions; another method is acceptable if it satisfies the outcomes. Work branches continue from the preceding validated PR in the PCGym chain, whose initial base is `cff-ucvx-readme-capability-map` (PR #161). Update the intended base before branching. Code changes require focused tests and one passing final short suite; documentation-only work validates its changed artifacts.

## Future work

No additional future work is identified within this ticket; the completed outcome contributes to epic cff-xa6z.

## Definitions

Ablation: a comparison changing one component. Compute budget: allowed time or computational resources. Microbenchmark: timing one defined operation. Oracle: independent mathematical reference.

Shared terms: CFFG means constrained Forney-style factor graph; JAX is the numerical array/compiler library; dtype is a numeric storage type; free energy is the model's inference objective; host/JIT compares Python-side and compiled execution; single/vmap compares one instance with vectorized instances; N/A means not applicable.

## Notes

**2026-11-22T03:38:43Z**

Fixed-model reactor arm cfp-lsxw is complete. The paired deadline protocol favored the directed reference on all ten seeds (+0.353259 mean return); all fifty directed candidate replays passed stock checks. Startup accounted for much of the observed gap. See ~/agent-share/cffg-reactor-paired/index.html. Other arm prerequisites remain open.

**2026-11-22T20:53:49Z**

PCGym comparison inputs are available from the integrated directed and bounded implementation. For P4/H15 use Engine.run with the default schedule and explicit tolerance, read gaussian_marginal on the current action edge, and log termination_reason plus startup timing boundaries. For an H2 clipping-vs-box comparison use the same graph, data and directed solve with tools/cstr_bounded_comparison.py. Do not use separate scalar boxes as a proxy for the correlated 15-command box. Keep seed/noise/initial conditions/goals/horizon, physical actuator constraints, deadline start and fallback identical across libraries. Existing 599-window replay validates numeric parity; closed-loop return and cross-library performance remain to be measured.

**2026-11-22T21:21:40Z**

Four-library timing pilot cpt-ydse qualified CFFG directed Gaussian inference, native RxInfer streaming linearization, NumPyro joint Laplace, and sparse GTSAM LM on three common affine Gaussian windows apiece. Seed 0 completed all eight fixed-model control runs and 48 forecast origins. Use tools/pcgym_timing/ and ~/agent-share/pcgym-four-library-timing/index.html for the frozen pilot protocol, gate tolerances, native version locks, per-tick traces and projection. Both lengths plus twelve forecasts per arm on 30 seeds project to 51 min serial on the M2 Max under one warm deadline. The one-seed returns and 12 overlapping forecast origins do not establish a population ranking, calibration, or adaptation to parameter drift.

**2026-11-22T22:25:59Z**

Fixed-model four-library comparison cpt-j81k finished on 30 paired seeds. CFFG tied RxInfer on all 30 long runs and achieved +0.6245 mean return against each of NumPyro and GTSAM (95% paired-seed intervals +0.6141 to +0.6349). Short-run differences crossed zero. All 78,960 commands met the warmed one-second deadline and all 1,440 forecasts scored. The report and raw evidence are at ~/agent-share/pcgym-paired-30/. These arms vary sensor noise on a fixed reactor with fixed model parameters. Other model, bounds and learning comparisons under this parent retain their own prerequisites.
