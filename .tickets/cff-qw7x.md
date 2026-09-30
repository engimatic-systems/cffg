---
id: cff-qw7x
status: in_progress
deps: [cff-36x7, cff-pl0v]
links: []
created: 2026-09-25T11:23:56Z
type: feature
priority: 1
assignee: coopwilliams
parent: cff-toqa
tags: [pcgym, nonlinear]
---
# Establish and document the nonlinear reactor planning envelope after startup repair

## Background in Plain English

The original failed CSTR caller is missing. Retained mechanism tests instead identify partial-input initialization and a separate unstable fixed-step RK4 evaluation. After the initializer is repaired, the library needs an honest, independently checked statement of which reconstructed planning cases work and which still fail.

## Outcomes / Acceptance Criteria

1. Supported reconstructed CSTR cases run through gaussian_nonlinear without caller-side linearization and satisfy finite-belief, adaptive-residual and free-energy convergence checks. Record their exact inputs and do not substitute them for the missing incident invocation.
2. Taylor and sigma-point cases are assessed across declared process/goal precision ratios and initialization choices in both dtypes. Unsupported cases remain explicit rather than hidden by parameter clipping or covariance floors.
3. The affine limit agrees with independent dense Gaussian conditioning. Preregister bounds: normalized residual at least the existing dtype floor of 64*eps, affine comparison tolerance 512*eps in float32 and 1e-8 in float64, with separate justified scaling for energy.
4. Wide-prior RK4 overflow and finite values too large for second moments remain attributable physical/approximation failures. Permanently unresolved partial inputs do not become converged plans.
5. Include the retained long-horizon, sparse/vacuous-information, host/JIT and scheduling checks; put long control/globalization regressions in opt-in slow tests and run those relevant to this change.
6. README, ARCHITECTURE, LIMITATIONS and executable examples describe the supported startup and nonlinear approximation envelope. The canonical model remains reconstructible from its function specification and arrays.
7. Close parent cff-toqa only after this integration contract and the prerequisite runtime work are verified. Do not claim the unretained report threshold or globally exact nonlinear inference.

## Methodological Suggestions

Another sound investigation is acceptable. Start from tests/cstr_planning_fixture.py and test_cstr_planning.py, including the independent affine oracle. A past=0 observed-current-state fixture is a separately declared candidate finite-physics case, not evidence about the unretained past=4 caller. Use Runtime numerical/status and New factor/rule family rows; list host/JIT, flood/scheduled and single/vmap impacts. Checkpoint schema is N/A unless the implementation changes it. Run focused and relevant opt-in cases, then one final default suite.

## Future work

cff-wt1c performs matched end-to-end control and runtime comparisons. cff-c5qf separately decides partial-goal hybrid mixture planning. Continuous parameter posterior learning and action bounds remain their own original roadmap tickets.

## Definitions

CSTR is a continuous stirred-tank reactor. RK4 is a fixed-step fourth-order numerical integrator. An affine limit replaces nonlinear dynamics with a linear map plus an offset. An oracle is an independent reference calculation. Eps is machine precision for the dtype. The planning envelope is the explicitly supported set of inputs and approximations, not a guarantee for every nonlinear model.


## Notes

**2026-09-30T18:21:03Z**

Pre-implementation handoff: Runtime numerical/status and New factor/rule family architecture rows. This integration assessment uses the existing portable mean/function registry and repaired Engine.run; ordinary equations stay unchanged unless new work is split into a separate ticket. Host/JIT mean/local updates and flood/mixed Engine runs apply; serial member/Colony readiness was verified in cff-36x7 and will be reused where unchanged. Flat/registerized adaptive scans checked N/A (iterative bindings excluded); current/legacy checkpoint decoding N/A (no persistent runtime change). Preregistered finite candidates use explicit past=0 current observations and declared P0/Qp/goal covariance arrays, not silent floors or incident replay. Float32 residual >=64*eps, affine 512*eps; float64 affine 1e-8. Sparse/vacuous inputs and H150 budget exhaustion stay explicit; wide RK4/sigma range failures remain retained. First assess exact parameter/noise/goal inputs, then publish supported envelope and executable example.

**2026-09-30T18:29:32Z**

First explicitly declared candidate, not incident replay: past=0,H2,P0=diag(.01,.001,1e-4,.1), Qp=diag(1e-4,4e-4,4e-6,.25), concentration/action-only goals, current observation [-1/3,.2], flood, budget60,tol1e-6. All four method/dtype runs converge in15 iterations: f64 Taylor root residual2.052e-15, energy46.49608643; f64 sigma1.919e-14, energy46.46345715; f32 Taylor1.760e-6, energy46.49592209; f32 sigma1.950e-6, energy46.46354675. Reported RunResult.residual is the difference between last accepted energy-history entries (f32 .00604/.01638), not the current-to-mapped ledger comparison used by solve_outer. Integration tests must validate the actual one-map FE certificate and separately justify dtype scaling, rather than compare that history field to tol. Sources: retained temporary four-case probe, outer.py ledger_stationary and outer_strategy residual assembly. Further ratios/schedules/init choices pending.

**2026-09-30T18:38:04Z**

Independent research: question—does narrowing P0 while retaining the reactor mean preserve source fidelity? Answer—yes as a separately declared new fixture, not the missing P4/H15 incident. Source tests/cstr_planning_fixture.py and retained experiment physics; confidence high. Current [-1/3,.2] means Ca=.8,T330K; zero goals mean Ca=.85,Tc298.5K. Clip evaluation inputs only: Ca[.1,1.6],T[200,450]K,Tc[275,322]K; no hard actuator constraint and intermediate/final RK4 remain unclipped. Independent DOP853 probes of prior/current-conditioned center and eight sigma points are finite in both dtypes; RK4(8) normalized error <=3.10e-8 f64 and1.33e-6 f32, reduced to1.87e-9 f64 by16 substeps. Actual solved cavity point clouds still require checks. State concentration as raw model units and dt26/60 model time units: upstream concentration/time labels are inconsistent. No globally exact nonlinear evidence claim.

**2026-09-30T18:49:39Z**

Preregistered physical-integration comparison for declared prior and solved cavity sigma points: independent NumPy ODE with SciPy DOP853, rtol=atol1e-12; maximum normalized-coordinate absolute discrepancy max(1e-5,512*eps). This is an integration-error check, not a posterior-accuracy bound; 1e-5 is1000 times smaller than the normalized current-observation SD. Required actual one-map residual max(1e-6,64*eps) and energy delta <=that*(1+abs(FE)); no golden energy or history-residual shortcut. Float64 auto grid at all6 Qp/Tgoal combinations and bothmethods converged4iterations with one-map residual <=2.04e-13 and FEdelta<=5.12e-13. Original wide-prior mechanism wrapper retains its P4/H15/defaultnoise/observations unchanged; public recipe defines distinct narrow prior.

**2026-09-30T18:59:11Z**

Integration discovery split into prerequisite cff-pl0v: f32 sigma/flood reports root residual1.950e-6 but next-map residual of returned state is6.617e-5 in the SAME frozen solver chart, above64eps7.629e-6; physical point clouds pass independent ODE checks. The solver certifies current but returns mapped in3 success paths. New-chart measurement is not used as proof. Public recipe/test work is retained while the independent return-point defect is repaired. Before rerunning integration, use the captured solver chart and reconstruct mixed owned slots; no tolerance relaxation.

**2026-09-30T19:37:02Z**

Prerequisite cff-pl0v complete: all3 outer success paths now return the certified point. Focused97pass, real f32 reactor frozen-chart certificate passes, mixed reconstruction2pass, final default gate2631pass157deselected wall1182.696s. Resume retained recipe/example from named stash on a branch based on the certified-return PR. Re-run relevant envelope opt-ins after this behavior change; the newly defined default-suite tests need one final gate at the envelope head. Continue actual cavity physics checks and truthful grid assessment; original missing incident remains unavailable.

**2026-09-30T19:44:13Z**

Resume on cff-qw7x-supported-reactor from current published PR#178 ac6d2bc after ff-only bootstrap. Restored source/example work; public builder defines narrow P0 while the original investigation wrapper retains its old defaults. Actual certificate uses captured fixed chart and mixed-slot reconstruction. Fresh-process bundle check confirms absent mean rejected, explicit trusted installation enables load/compile; installation must precede load_bundle, not just a later compile call. New default-focused selection21passed36deselected in18.99s; full relevant opt-ins running. Default suite not yet rerun for this separate implementation head.

**2026-09-30T19:50:21Z**

Relevant completed opt-ins after cff-pl0v: default auto cold/warm bothmethods xbothdtypes —4 passed15deselected in444.86s; initial slow batch has completed both f32 affine-oracle cases and all4 flood cold/warm cases before its grid. Retained affine fresh/exact-seeded, wide RK4, strict missing-information and H150 strict/budgeted startup checks —10 passed41deselected in209.98s. Standalone public recipe subprocess —1 passed16deselected in108.82s. No broad speed claim; cold compile/control is expensive and remains the measured-runtime roadmap. Grid still running; canonical prose is conditional on passing exact enumerated checks.

**2026-09-30T19:53:14Z**

Final bounded review question—do extraction/docs match exact inputs and honest scope? Answer—yes conditional on pending grid: P0/Qp/observations/goals/6 ratios/budget/dtype floor match; no incident/exact-posterior/actuator/closed-loop claim. Source example, original wrapper diff and canonical doc hunks; confidence high. Physics/topology/wide-prior defaults retained. Intentional callback identity is now cffg.example.cstr_rk4; previously saved INTERNAL test fixtures naming cffg.test.cstr_rk4 still need that old trusted registration, not automatic alias/migration. Current bundle schema unchanged. Arrays now own a copy of nominal theta so caller parameter edits cannot contaminate future recipe builds; focused ownership check added.

**2026-09-30T19:58:10Z**

Both float32 grid blocks have passed all6 Qp/temperature-goal pairs in the current opt-in run, with actual frozen-chart residual/FE checks and DOP853 comparisons of solved cavity sigma points. No tau relaxation, noise floor, physical clipping change or caller-side linearization. Float64 grid blocks still running. All remaining default checks after theta ownership change pass21/36deselected; no runtime changes in this envelope PR.

**2026-09-30T20:01:56Z**

The float64 Taylor grid block also passes its six enumerated combinations; only float64 sigma grid remains. Source/API and tests committed as d2f7f79 (default-focused checks passing; numerical equations unchanged). README/ARCHITECTURE/LIMITATIONS describe exact input scope, separate DATA reconstruction, clipping vs actuator bounds and approximate objective. D13 stale pre-startup text corrected to budgeted startup and frozen-chart certified-point handoff. No claim of original failure-threshold reproduction.
