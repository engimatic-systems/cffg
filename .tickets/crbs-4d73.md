---
id: crbs-4d73
status: closed
deps: []
links: []
created: 2026-11-23T04:25:49Z
type: task
priority: 2
assignee: coopwilliams
parent: crbs-duxy
tags: [benchmark, rxinfer, extraction]
---
# Build an extraction-column fixture with two visible outlets and independent hidden truth

## Background in Plain English — WHY
The existing extraction experiment already models ten interacting concentrations and controls two flows. It exposes all concentrations and uses one model hour per decision. At some flow settings many internal changes nearly disappear before the next sample. A credible comparison needs a sampling interval and flow pattern that let both libraries learn something from the outlets.

## Outcomes / Acceptance Criteria — WHAT
1. Pin PCGym and generate truth with its native five-stage differential equations and a qualified CasADi integrator. Give the adapter only noisy X5 and Y1, their timestamps, the applied liquid/gas flows L and G, and the public references when planning. Restrict the other eight concentrations, true Kla, plant internals, switch times and future noise to scoring. The reset interface follows the same rule.
2. Vary Kla inside the actual native transition and verify one uninterrupted episode against independently configured fixed-Kla one-step integrations before, at and after a change. PCGym's native feed disturbances X0 and Y6 are separate quantities. Check no reset across changes and independently replay the applied actions, terminal state, time, observation and reward alignment. Keep common random innovations across paired conditions.
3. Use development tapes to demonstrate attainable X5 setpoints, sustained independent variation in both flows, sensitivity of measured outlets to Kla and resolvable hidden-state information under the proposed sensor noise. Compare per-state errors with an observer using no sensor updates and an informed reference. Report which components stay weakly identified. If the informed learner has negligible tracking or prediction headroom, revise the fixture on development data before qualifying a comparison.
4. Give the pilot explicitly timed stationary and changing conditions. Start with nominal Kla = 5/hour, a step down to 4.5, a return, and a later ramp to and from the same changed value. Fix or bound all values, durations, independent change-time jitter and noise before fresh seeds. Parameter return and setpoint changes occur at different times. Plant controls and actions use the same clipped physical ranges in both arms.
5. Check the solver integration, controller transition and Jacobians against the plant at nominal, perturbed and flow-limit conditions. Select time step, substeps and horizon using actual process settling times, stability and physical reachability. Record a development wall-time estimate. Freeze an additive two-channel Gaussian sensor with calibrated scales shared by truth and likelihood, or explicitly document and qualify an alternative. The sibling experiment used multiplicative sensor noise.

## Methodological Suggestions — HOW
Use the existing sibling extraction_column/experiment_physics.py and common/envs.py for plant identity, units and intended X5 targets. The older predictor uses 160 RK4 substeps and state bounds. Check its errors against the native integrator before reusing it. A shared open-loop probing schedule lets the two flow controls vary independently. Keep hidden initial profiles varied and perturb process/parameter truth across seeds within the frozen design so a predictor cannot replay a known trajectory. Test that changing any scorer-only field cannot change a planner's next input.

## Future work
The development pilot chooses a sampling interval and whether the two-outlet task has enough learning headroom. Adding an interior sensor changes the benchmark and needs an explicitly revised development protocol.

## Definitions
A concentration profile is the ten levels of dissolved material across the five stages. Observability measures how changes in those levels affect available measurements. A Jacobian is a local sensitivity matrix. An informed reference is given the true transfer coefficient. An innovation is an independently drawn noise input.


## Notes

**2026-11-23T04:26:57Z**

Feasibility calculation on the installed PCGym 0.1.8 multistage_extraction right-hand side. At L=75, G=30 and Kla=5, a positive steady solution gives X5=0.3328292 and Y1=0.7179270. Linearize the 11 variables (ten concentrations, log Kla with zero derivative) at this point by central differences of size 1e-5, cross-checked at 1e-6. Stack H exp(k dt A) for k=0..10, where H selects zero-based rows 8 and 1. Whiten each output by 5% of its nominal level, and scale perturbations by 0.05 concentration and 0.1 log Kla. At dt=0.05 model hours the unweighted stack has numerical rank 11, but only 3 scaled singular values exceed 1 (largest 3.3664, smallest 0.002323). At dt=1 the numerical rank falls to 8 at this operating point. This tests local sensor sensitivity for 11 readings at constant flow. It does not establish nonlinear identifiability or forecast accuracy under excited conditions. Use varied flows and an informed reference to establish that before scoring.

**2026-11-23T04:26:57Z**

Plant review: existing sibling ~/experiments/cffg-testing/pc-gym/00/cffg/common/envs.py:34 exposes the ten concentrations; extraction_column/experiment_physics.py:22 and common/plants.py:113 carry older nonlinear model choices. Installed pcgym/model_classes.py:350 defines multistage_extraction, two physical flows and two feed disturbances. A changing Kla requires a wrapper. Raw info may expose hidden state. Confidence high for the source layout and the local linearization, unqualified for long-rollout identifiability.

**2026-11-23T04:52:07Z**

Architecture change path checked: runtime numerical/status invariant. The fixture uses only a local benchmark plant and offline qualification. Engine, model serialization, scheduler, FE and runtime API surfaces are N/A. Target the native RHS with an independent fixed-parameter integrator, full discrete Jacobians, sparse data and causal replay checks.

**2026-11-23T04:57:07Z**

Implemented tools/extraction/plant.py and plant_check.py. Plant uses PCGym 0.1.8 git d26cbc0690024efef488dc6a304d37654934e525 directly, through CVODES. Physical Kla is supplied with the applied flows on each transition. Scorer-owned distinct seeded streams supply async L/G probes, initial state, outlet and process noise, plant changes and public references. Each 600 transitions contains two matched abrupt visits and a ramp. The 6,000-step trace repeats the design with new jitter and the short trace is an exact prefix. Expose only the two noisy outlets in X5,Y1 order, including at reset; the scorer keeps the full tape.

**2026-11-23T04:57:07Z**

Development validation: uv run --frozen --no-sync python tools/extraction/plant_check.py --output ~/agent-share/pcgym-extraction-rxinfer/qual/plant-qualification.json passed all controls. Seeds 17101–17103 each have 600 transitions, 17104 has 6,000 per stationary/changing plant; no evaluation data. Independent fixed-coefficient SciPy DOP853 replay of all changing transitions matches native state, terminal, observation and post-command tracking cost within 2e-9. Six preview ticks (0.3 hours) reach the .295–.375 target range from nominal with either coefficient; slowest local 95% settling is 27.2 ticks, below the 62–68 tick abrupt dwell. Shared-tape reference EKF with known Kla improves hidden RMSE vs nominal by 87.7–88.1% on changed periods and improves pre-reading whitened physical outlet RMSE 78.0–81.9%. Adaptive reference hidden RMSE improves 81.8–85.4%. Conditional sensor gain is much smaller with the parameter known. On changed phases, X1 and X2 error decreased only 0.3% and 2.2% relative to a known-Kla observer without sensor updates. Six of eleven local scaled observability singular values exceed one. Physical concentration ranges remained inside [0,1] without clamps. Fixture checks took 6.99 s, excluding imports; the controller Jacobian and library gates are tracked separately by crbs-2xf6/crbs-jqe2. Isolated external experiment; modifies no library or test-selection code. py_compile, tk dep cycle and git diff --check passed.
