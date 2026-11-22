---
id: cpc-7tbv
status: in_progress
deps: [cpt-j81k]
links: [cff-ees0, cff-wt1c]
created: 2026-11-22T23:27:40Z
type: task
priority: 1
assignee: coopwilliams
parent: cff-xa6z
tags: [pcgym, benchmark, continual-learning]
---
# Pilot online learning of a changing reactor heat-transfer coefficient

## Background in Plain English — WHY
The four fixed-model controllers completed every decision on 30 noise seeds. That experiment supplied the physical parameters and each controller estimated the current state from a short window. Cooper now wants to know whether a controller can carry a learned physical parameter forward and recover control quality when the real plant changes. The first experiment should show whether the measurements reveal such a change, before scaling to four libraries.

## Outcomes / Acceptance Criteria — WHAT
1. Demonstrate a real change in one hidden physical coefficient on an uninterrupted simulated trajectory. Use nominal, changed and return-to-nominal periods and a stationary control. Keep the change times and values hidden from the learner. Verify the transition against an independently configured plant at the same state and coolant command and preserve time, state and noise across the switch.
2. On a common development tape of observations and applied controls, predict each new observation before using it. Update persistent parameter beliefs from each realized transition once. State how noisy measurements, latent bias and uncertainty in the previous state enter the likelihood. Test whether the coefficient is distinguishable at the observed noise level and under the available commands.
3. Compare a fixed parameter estimate, a learner and the same planner given the true coefficient. Preserve the planner settings, short-window state estimator, deadline, fallback and sensor access. Include a transparent independent one-dimensional reference. Score the stationary control as well as the change. If the informed planner offers no measurable advantage over the frozen planner, report that result before scaling.
4. Freeze the inference model, update cadence, change conditions and scoring windows on development data. Report prediction error before updates, coefficient error, recovery time, post-change tracking/return, and latency and failures including learning. If the learner chooses from known candidates, identify this as finite-regime tracking. For an unknown-parameter learning claim, also test an unseen coefficient value and expose the contribution of the persistent estimate by disabling its updates.
5. Record the fixed revision, seeds, plant truth available only for scoring, chronological raw data and replay instructions. State whether the evidence warrants a scored multi-seed comparison and which library functionality it actually tests.

## Methodological Suggestions — HOW
A transparent first target is UA, the heat-transfer coefficient times area. Keep the reaction constants fixed. Use independent simulator truth and a one-dimensional grid reference on the same causal data. A declared transition model for the coefficient can assign probability to a change after a long stable period. A stable finite bank with one fixed coefficient over the whole run answers a different question. A known-candidate graph or a cheap scalar point estimator can serve as a development control. Treat the posterior over a finite bank as exact only for its declared likelihood. Future goals in the planning window are preferences and cannot supply evidence about past physics.

Architecture change path is an external consumer of runtime parameter replacement, certified readout and the learning cadence. Host/JIT mathematics, vmap, bundle format and agent-facade changes are N/A for this pilot. Use the tested float64 reactor fixture. For a grid include valid support, failure on impossible evidence and a comparison with explicit enumeration. Continuously valued estimates carry their own uncertainty or are labeled point estimates. Figure out which library surface is earned by the experiment before generalizing it.

## Future work
cff-ees0 owns a public fixed-bank update. cff-wi2o and cff-iplr own continuous parameter-particle inference. cff-3wsw and cff-5k0n own supported physical point training and its cost measurements. cff-wt1c owns broader comparisons across libraries. None of the general features is required to establish the experiment on development data.

## Definitions
A frozen controller keeps its physical coefficients fixed while continuing to estimate the live plant state. An informed reference controller receives the true coefficient for comparison. A development tape is a recorded sequence of measurements and applied controls used by every method. Identifiability means that different coefficient values produce predictions that can be distinguished at the available sensor accuracy. A latent bias is an unobserved offset the controller already estimates.


## Notes

**2026-11-22T23:27:59Z**

At f3a78f9, I tested parameter replacement on the scored P4/H15 CFFG adapter. On one synthetic development window, increasing UA by 3% using set_params on tr1 changed the normalized command from -0.5829362795 to -0.3573894544. A separately constructed engine at the new coefficient returned exactly that value and restoring the original coefficient restored the first command. Both used the directed certificate. This checks parameter delivery to the planner, not inference of UA.

One-step simulator control, with identical initial state and coolant action: nominal temperature 326.48088126 K, after changing env.model.UA by 3% following reset also 326.48088126 K, and when constructing PCGym with custom_model=cstr(UA=51500) before reset 325.79631275 K. At reset PCGym builds a separate CasADi integration model. To implement a hidden switch, change the dynamics in the integrator actually used for that step and check continuity. The 3% probe is illustrative and does not preregister the study effect size.

Legend: prerequisite --> dependent. Lines are parallel workstreams.
[closed] cpt-j81k --> [ready] cpc-7tbv (changing-reactor pilot)
[ready] cff-ees0 --> [blocked] cff-wi2o --> [blocked] cff-iplr --> [blocked] cff-wt1c
[ready] cff-5k0n --> [blocked] cff-csmz --> [blocked] cff-wt1c
[ready] cff-3wsw --> cff-xa6z (broader PCGym roadmap)
cff-wt1c also has other open prerequisites. The pilot can use benchmark-local orchestration; a claim about a library feature requires exercising that supported feature. No merge or scoring run is part of this handoff.

**2026-11-22T23:28:16Z**

Read-only implementation probes found a no-fork simulator path. A PCGym custom_model can expose UA to its integrator as a scheduled disturbance; switching 50000 to 40000 with the same coolant changed the first affected temperature by 4.454 K in a noise-free smoke. PCGym consumes the disturbance at t+1 and appends it to its observation. Strip that channel and any oracle state from the learner, and timestamp the action/observation pair accordingly. Source: installed pcgym/pcgym.py around lines 292 and 404, pcgym/integrator.py around line 29. Verify again in the scored harness.

The earlier stationary experiment actually learned: cstr/runs_v3b/seed_0/train_summary.json reports 3333 learning updates in 100000 steps; common/physics_agent.py scores batches and common/physics_exp.py freezes the MAP estimate at evaluation. Use its finite-bank graph as an available control. Reweighting or switching among supplied regimes establishes regime tracking. Measuring physical parameter learning also needs a value the controller was not handed. In the retained four-library benchmark the CFFG adapter reuses Adapter.base and the separate forecast propagator reads the original arrays[theta] in tools/pcgym_timing/forecast.py. Supply the learner estimate to both and compare to a freshly constructed model on the same data. Conjugate Gaussian learnables cover affine coefficients but not these nonlinear RK4 constants. The separately checked small examples and source inspection support these engineering conclusions; the outcome on development and held-out drift data remains open.

**2026-11-22T23:41:00Z**

Development protocol fixed before collecting outcomes. Shared open-loop tapes use seed 6400 and the previous sinusoidal command schedule. Closed loop uses five paired development seeds 6401–6405, each with a stationary plant and a hidden 50000 → 48150 → 50000 schedule at transition indices 178 and 373 of 599. The changed value lies between points in the independent 250-unit grid and is not passed to the continuous learner. All controllers share the certified P4/H15 planner. Frozen and learned initially use 50000. The informed comparator is told only the current transition coefficient. The continuous estimator is a scalar extended Kalman update in log UA, with prior SD .10, per-transition motion SD .012, and bounds 40000–60000. The temperature innovation likelihood propagates uncertainty in the two previous sensor readings. The independent grid uses local diffusion and uniform restart probability .002. Both receive only observations and applied commands. This is benchmark-local learning feeding public CFFG parameter replacement. The raw records and hashes will live under ~/agent-share/pcgym-continual-learning.

Prespecified engineering screen for spending time on a larger control study: on the changed segment both the informed and learned controllers must each reduce concentration RMSE versus frozen by a median of at least 1%, and win at least 4/5 pairs. The stationary learned mean full-episode concentration RMSE may increase at most 2%; the learned mean coefficient error in settled changed operation must fall by at least 50%; the learner must improve changed-period one-step temperature prediction on the common tape; no controller may hold. Relative return and all component scores remain reportable on either result. Recovery is the first of five consecutive pre-decision estimates within 1% of true UA. The initial 25 transitions after each change are reported separately. No other libraries or held-out scoring are in this development run.
