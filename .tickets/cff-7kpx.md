---
id: cff-7kpx
status: in_progress
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

## Notes

**2026-09-25T10:43:38Z**

Pre-implementation handoff: Runtime numerical or status invariant and New factor or rule family rows, for investigation only; no production equations changed here. Host/JIT local results, flooding/scheduled initialization and single/vmap checks apply; current/legacy decoding N/A. Retain the CSTR RK4 function and portable graph inputs in a maintained numeric fixture, with declared synthetic choices for missing incident data. Hostile checks: partial/vacuous information, float32/float64, finite versus overflowing physics, process/goal precision grid, affine limit and larger domain. The exact incident replay is not promised. A bounded investigation will classify observed mechanisms and pass a concrete contract to cff-toqa. Focused checks and a final full gate if maintained test code changes.

**2026-09-25T10:43:38Z**

Research question: is the exact failed invocation reconstructible? Answer: no. The retained common/physics.py contains the RK4 registration (lines57-101), constants (46-54,69-76) and nonlinear window builder (110-154), but no saved caller establishes m0/P0, Q/Qp, observations, FunctionSpec configuration/substeps, schedule, damping or run budget. FEEDBACK.md line54 reports P4/H15, Taylor or sigma_point,59/59 first-sweep IMPROPER_CAVITY, temperature-goal variance.01 passes versus1 fails. Later common/physics_agent.py46-54 constrains Ca and actuator, not temperature; it is neighboring workaround evidence, not original provenance. Original RK4 clips only its input, not intermediate/final stages; do not substitute common/plants.py integrator. Source: pcgym_numerics_review bounded code/report audit; high confidence. Proposed deterministic fixture uses retained nominal constants plus explicitly labeled v3b neighboring settings and synthetic nominal observations.

**2026-09-25T10:50:03Z**

Retained fixture now records the original RK4 formula/topology without external imports or pickle. Four finite physical points match the retained _cstr_eval exactly in float32 and float64. Explicit reconstruction: nominal parameters, normalized [Ca,T,d_Ca,u],dt26/60,eight RK4 substeps,m0=0,P0=diag(1,1,1e-4,1),Q=Qp=diag(1e-4,4e-4,4e-6,.25),past missing,current[-1/3,.2],future goals zero. A synthetic T goal has variance.01 or1. Default P4/H15 auto rejects Taylor at tr10 with input spectrum[0,1,1,2500]; sigma-point rejects tr1. The temperature sigma point at normalized T=2 gives nonfinite output in float32 and finite magnitude3.13e266 in float64, whose squared moments overflow. These are distinct from the partial cavity. No incident-specific ratio threshold is established.

**2026-09-25T10:53:12Z**

Focused investigation including opt-in cases:21 passed in174.08s. The affine limit (both Taylor and sigma points) fails from fresh partial-goal initialization but converges after semantic factor/role transfer of an exact linear-Gaussian solution; moments and evidence agree with independent dense NumPy conditioning. Tests distinguish partial input from vacuity and compare host/JIT/vmap statuses in both dtypes. Eight full P4/H15 precision-grid runs cover48 configurations:Qp scales.01,1,100;T-goal variances.01,1;both methods,dtypes,schedules. Tenfold horizon150 preserves the partial-input mechanism. Long complete solver/grid cases are opt-in slow; direct local admission and RK4 range checks remain default. No production equation or solver behavior changed.

**2026-09-25T10:54:37Z**

Final investigation selection: all17 full reactor/affine/grid/long-horizon behaviors are opt-in slow and were explicitly exercised; four cheap float32/float64 local partial-information/JIT/vmap checks remain default. This supersedes the earlier note that RK4 model-range checks were default. Existing nonlinear rule hostile-input coverage remains selected. The precision grid now additionally pins each first-failed input spectrum, not just its rank. No production behavior changes are included in this investigation PR.

**2026-09-25T10:54:43Z**

Final spectrum-grid checks:8 passed,13 deselected in76.41s. Across both dtypes and both schedules, all Qp scales(.01,1,100) give Taylor first failure tr10 improper_cavity. T variance.01 gives eigenvalues[0,1,100,2500];T variance1 gives[0,1,1,2500]. Every sigma-point case first fails tr1 nonfinite_function with proper input eigenvalues[1,1,1,10000]. Failures occur within three iterations. Thus this reconstruction explains two concrete failure mechanisms but not the reported goal-variance pass/fail boundary. The missing original observations/configuration prohibit that claim.
