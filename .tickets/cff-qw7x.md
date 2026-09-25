---
id: cff-qw7x
status: open
deps: [cff-36x7]
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

