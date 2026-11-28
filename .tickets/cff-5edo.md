---
id: cff-5edo
status: in_progress
deps: [cff-wbhu, clr-xtni]
links: [cff-fxmj, cff-j4mu, cff-i45j, cff-wbhu, cff-d56d]
created: 2026-07-10T03:31:09Z
type: feature
priority: 2
assignee: coopwilliams
parent: cff-ahps
---
# Long-rollout planning: cost and stability beyond ~20 slices

## Background in Plain English

This feature makes long receding-horizon plans converge reliably and reuse work between environment steps. It owns two changes in `RollingAgent`: choose a planning budget from the compiled plan depth while running to a convergence tolerance, and warm-start the next plan from the overlapping part of the previous converged plan.

The original evaluation used a 60-state aliased corridor and horizons from 5 through 25. GFE planning scored 0.786 success through horizon 20, above the 0.643 QMDP reference, but dropped to 0.571 at horizon 25. Median planning time per environment step grew from 0.61 seconds at horizon 5 to 2.47 seconds at horizon 25 and was dominated by message sweeps.

The historical diagnosis supersedes the original limit-cycle hypothesis. Messages travel about one slice per sweep. A 120-sweep cap left the horizon-25 plan with residual 5.5, while 400 sweeps reduced the residual to 4.9e-4. The poor action result came from using an under-converged posterior. `Engine.run` now reports explicit convergence states and uses free-energy stationarity plus message coherence, but `RollingAgent.act` still starts with uniform messages and applies one configured sweep cap on every call.

Open prerequisite `cff-wbhu` owns the compiled sweep loop and truthful per-instance stopping state needed before this feature measures steady-state cost. Closed tickets `cff-j4mu`, `cff-d56d`, and `cff-i45j` supplied explicit runtime statuses, schedule-aware propagation, and sparse-support GFE robustness. Downstream task `cff-t2g3` needs this feature for eight-hour glucose planning, and parent epic `cff-ahps` tracks its roadmap effect.

## Outcomes / Acceptance Criteria

1. `RollingAgent` plans to a documented convergence tolerance with a finite cap derived from the compiled plan's maximum scheduled message-path length. A caller may override the cap, but the default is not one horizon-independent sweep count.
2. Preserve the existing action gate and last-plan diagnostics when changing budgets and warm starts. The PR #185 branch already implements 2.1–2.3 in `RollingAgent.act` and `last_plan_result` (src/cffg/agent.py).
   2.1. `RollingAgent.act` returns an action after a `RunStatus.CONVERGED` plan.
   2.2. Budget exhaustion, divergence, and infeasibility remain distinguishable and cannot return an action that appears successful.
   2.3. The agent exposes the last planning `RunResult` for review.
3. The aliased-corridor evaluation records horizon, episode seeds, action-selection rule, sweep cap, tolerance, runtime status, free-energy trace, message residual, success, steps, and separately timed compile, first execution, steady-state planning, and readout phases.
4. The horizon-25 behavioral comparison uses the recorded aliased-corridor protocol.
   4.1. On at least 100 paired seeded episodes, the horizon-25 success point estimate is no more than 0.05 below the horizon-20 estimate.
   4.2. A paired 95% bootstrap confidence interval for the horizon-25 minus horizon-20 success difference includes zero.
5. Warm-started replanning shifts state across adjacent planning windows.
   5.1. The converged message and inner-solver state for overlapping horizon slices move into their next-step positions.
   5.2. The new planning window receives the new boundary beliefs and parameter values.
   5.3. The new terminal slice receives initialized message and inner-solver state.
   5.4. Expired observations and controls are absent from the new planning window.
6. A warm plan and a cold plan that receive identical templates, parameters, boundary beliefs, and random keys reach free-energy values, beliefs, and action probabilities within dtype-scaled documented tolerances. Deterministic MAP selection returns identical actions.
7. The horizon-20 warm-versus-cold benchmark records its latency threshold and provenance.
   7.1. Median steady-state planning time for warm-started replanning is at least three times lower than cold replanning.
   7.2. The comparison excludes compilation and first-execution time.
   7.3. Warm and cold measurements use identical convergence criteria and outputs.
   7.4. The benchmark reports hardware and dtype.
   7.5. The benchmark includes at least 100 paired environment steps after first execution.
8. The warm-start cache remains a derived optimization with explicit invalidation behavior.
   8.1. The cache is not model source of truth.
   8.2. Episode reset clears the cache.
   8.3. Checkpoint restore clears the cache.
   8.4. An incompatible template structure clears or rejects the cache.
   8.5. A horizon change clears or rejects the cache.
   8.6. A shape change clears or rejects the cache.
   8.7. A dtype change clears or rejects the cache.
   8.8. Restoring a checkpoint and cold-starting produces an action identical to an uninterrupted compatible run after both plans converge.
9. Warm-start storage stays bounded by the compiled planning-state size across a 10,000-step run; it does not grow with the number of environment steps.
10. Hostile-input tests demonstrate observable planning behavior.
   10.1. Exact log-zero support survives slice shifting, and a truly empty support reports infeasible rather than producing an action.
   10.2. Vacuous but proper boundary beliefs initialize a plan that either converges or reports budget exhaustion without a non-finite value.
   10.3. Float32 horizon-25 cold and warm plans use dtype-scaled tolerances and guards and report finite diagnostics when converged.
   10.4. A horizon-50 case, ten times the smallest recorded horizon, terminates with a truthful status under its declared cap; budget exhaustion is an acceptable result for this scale-survival case.
   10.5. A non-finite, structurally incompatible, or wrongly shaped warm cache is rejected or discarded before inference and cannot be reported as converged.
11. The implementation does not weaken true BFE/GFE evaluation, free-energy convergence, sparse-support semantics, or the runtime status contract to obtain the latency result.
12. The project documentation records the validated long-rollout behavior.
   12.1. `docs/ARCHITECTURE.md` documents the planning-budget and shifted-warm-start lifecycle.
   12.2. `docs/LIMITATIONS.md` replaces the unresolved horizon-25 warning with the validated envelope and records each measured limitation not resolved by this feature.

## Methodological Suggestions

The implementer may choose another sound method. The following approach is a starting point:

- Build a compiled mapping from each previous slice's message and inner-solver blocks to the preceding slice in the next planning window. Copy the overlap once, initialize the new terminal blocks, then rewrite boundary beliefs and parameters through the normal agent path.
- Keep the warm state inside `RollingAgent` as a cache. Clear it at episode reset and checkpoint restore unless the checkpoint format deliberately versions and validates every cached block. Prefer a safe cold retry when cache validation fails or a warm run does not converge.
- Use `Engine.run(state, cap, tol=...)` and inspect `RunResult.status`. Derive the default cap from compiled propagation depth with headroom for GFE outer coordination; do not infer success from sweep count or a message-delta trace alone.
- Compare warm and cold runs from cloned agent states after an identical observation. Pin free energy, marginals, action posterior, status, and deterministic action before interpreting timing.
- Time compilation, first execution, steady-state inference, and readout separately. Use paired repetitions on one machine, synchronize JAX work before stopping each timer, and report median latency plus a bootstrap confidence interval.
- Add focused coverage to `tests/test_agent.py` and the runtime-status tests. Put multi-episode aliased-corridor behavior and 10,000-step storage checks in `slow` tests. Run focused files during development, the applicable slow tests once for final behavior, and the repository's one required short suite after the final code change.
- The applicable `docs/ARCHITECTURE.md` change-path row is **Runtime numerical or status invariant**. Review `RollingAgent`, `Colony`, benchmark, estimator, and dashboard consumers. The single-instance/`vmap` and host/JIT pairs are affected through `cff-wbhu`; keep sequential and batched statuses and stopping semantics aligned. Flooding/scheduled parity is affected when the budget or cache path bypasses `Engine.run`; route both through `Engine.run`. The new/legacy-checkpoint pair is not affected when the warm cache remains derived and is cleared on restore.
- If the implementation requires an engine, schedule, or Colony change beyond the `RollingAgent` cache and budget policy, record that surface in this ticket before editing it. Keep slice-fused chain kernels in `cff-yl0a` rather than folding their separate throughput target into this feature.

## Future work

Open ticket `cff-yl0a` owns slice-fused chain kernels for the remaining strict specialized-smoother performance gap. This feature owns reuse and convergence at the agent layer, not that kernel-lowering target.

After this feature closes, `cff-t2g3` owns use of the long-horizon planner in the glucose controller, and parent epic `cff-ahps` owns the roadmap-level closure record. No other future work is identified.

## Definitions

- **Aliased corridor:** A corridor environment where different hidden states produce an identical observation.
- **Bethe free energy (BFE):** The node-and-edge variational objective used for ordinary CFFG inference.
- **Bootstrap confidence interval:** An uncertainty interval computed by repeatedly resampling the recorded paired episodes.
- **Cold replanning:** Starting a planning solve from initialized messages without reusing the prior planning solve.
- **Compiled plan depth:** A structural measure of how far information must travel through the compiled planning graph.
- **Convergence tolerance:** The dtype-aware threshold used to certify free-energy stationarity and message coherence.
- **Dtype:** The numeric data type used for arrays, such as float32.
- **Environment step:** One action, environment transition, and new observation in the agent loop.
- **Generalised free energy (GFE):** The active-inference objective used to score policies with goal and epistemic terms.
- **Horizon:** The number of future slices represented in one planning window; the historical notes call it `H`.
- **Inner-solver state:** State used by a nested numerical solve inside a message update.
- **JAX synchronization:** Waiting for asynchronous JAX computation to finish before recording elapsed time.
- **Limit cycle:** Iteration among repeating states without reaching the required fixed point.
- **MAP selection:** Choosing the action with the largest posterior probability.
- **Message coherence:** Agreement of outgoing messages with another update at the reported state.
- **Message residual:** A numeric measure of change or inconsistency remaining in messages.
- **`n_sweeps`:** The historical `RollingAgent` setting that caps message-passing iterations.
- **Posterior:** The probability distribution after combining model factors with observed data and messages.
- **QMDP:** A planning approximation that assumes uncertainty disappears after the next action.
- **Receding horizon:** Replanning after each new observation over a forward window of future states.
- **`RollingAgent`:** The repository's receding-horizon perception, planning, action, and learning controller.
- **RunResult:** The runtime result containing state, termination status, iteration count, traces, and residuals.
- **RunStatus:** One of converged, budget exhausted, diverged, or infeasible.
- **Slice:** One time-indexed copy of the state, observation, and control variables in a planning graph.
- **Sweep:** One scheduled round of message updates.
- **Warm-start cache:** Derived planning state reused as the initializer for the next overlapping planning window.
- **Warm-started replanning:** Initializing an overlapping planning window from the previous converged solve before running to convergence again.

## Notes

**2026-07-10T03:34:54Z**

DIAGNOSIS LANDED (evaluation round): the H=25 success drop is PURE UNDER-CONVERGENCE, not model error or limit-cycling — messages propagate ~one slice per sweep, and a fixed n_sweeps=120 leaves the deep-rollout control posterior unconverged (measured residual 5.5 @120 sweeps vs 4.9e-4 @400). Thread (1) resolves to: depth-aware or tol-driven sweep budgets in the agent planning loop (run-to-tol with cap exists in Engine.run; RollingAgent should use it). Thread (2) warm-started replanning unchanged. Also measured: per-step plan cost linear in H (sweeps x per-sweep, compile negligible).

**2026-08-08T04:26:21Z**

Bounded scan classifications: fixed=historical_finite_budget; reusable=artifact_reuse; expose=public_api; implement=planned_work; support=probability_support_or_hostile_numeric_survival; land=historical_component_landed; remaining=unfinished_work_or_residual_quantity_or_continued_property; approximate=named_mathematical_method; close=work_item_completion; certify=runtime_certificate; check=automated_check; converge=runtime_status_criterion_or_historical_under_convergence; estimate=empirical_measurement; exact=literal_zero_or_support; fixed_point=runtime_convergence_state; parity=behavioral_equality; pass=message_data_flow; robust=hostile_numeric_survival; sound=optional_method_choice; stable=numerical_or_convergence_stability; test=automated_execution_or_scientific_experiment; validate=input_or_artifact_or_scientific_validation; infer=state_or_policy_posterior_or_logical_deduction; learn=integrated_learning_workflow; performance=runtime_latency; scale=dtype_numeric_magnitude_or_horizon_size; small=horizon_size; may=permission_or_optional_requirement; should=historical_design_intent; explicit_owner=current_or_named_ticket_owner; named_ticket_handoff=concrete_current_or_future_owner; at_least=inclusive_lower_bound; every=temporal_frequency_or_named_cache_block; existing=historical_predating_artifact; complete=component_complete.

Intelligible-ticket review: complete.

**2026-11-28T05:40:31Z**

Assessment starts from published a774a54. The governing architecture row is Runtime numerical or status invariant. Preserve the public Engine run/status path for host and compiled execution, observe the actual action via the perception DATA port, and leave categorical checkpoint and action-RNG ownership unchanged. Ordinary/BFE, Colony selection and general vmapped ownership are unaffected by this corridor-only probe. The frozen Hallway environment and Phase A build are in the sibling long-horizon-pomdp repository at ec51aea. Its archived Phase A uses L=15, max_steps=30, open-loop GFE rollout for known dynamics, and MAP (historical first-index tie law). Current default proof-backed GFE does not run the historical damped sweeps.

What can shift today? The fixed categorical GFE composite stores backward state and predictive observation messages in State.messages at scatter_z and scatter_x. The mixed solver re-primes ordinary BP, rebuilds exact support and chart gauges, and has no persistent inner Newton vector. Use numeric member identities and compiler addressing. Keep fresh runtime parameters, prior, terminal, and external rows. The feasibility check runs before priming, so copying old sparse BP cavities can cause artificial infeasibility. Sources are specialized/composites.py, execution/synchronous.py, rules.py and outer_strategy.py. Confidence is high from static ownership and exact-support tracing. A production interface for other layouts requires separate proof of ownership.

What drives budgets and selection? Schedule.rounds measures inward forest path length, not general cyclic or iterative depth. The trust controller gates on reconstructed chart root, local solve, unchanged support and ledger agreement with the max(requested_tol, 64*dtype_eps) threshold. Source is schedule.py, outer.py and run_results.py. engine/agent public gating remains authoritative. The unmerged model-lint e6ad56d tie law samples uniformly over exact maxima with the checkpointed action RNG and no draw for a unique maximum. Probe records exact ties and applies only the current published action to the shared feedback history. A budget change needs care with n_sweeps checkpoint compatibility. Static findings by two independent reviewers are high confidence.

**2026-11-28T07:24:44Z**

Bounded resource assessment on a774a54 with the ec51aea Hallway Phase A builder, M2 Max CPU, JAX 0.10.2, one frozen start (state 47, seed 0), and the published tol=1e-10. All calls retain the normal convergence gate and apply the proposed command through the real simulated transition only after certification. Full status, FE/delta traces, native readout timings, hashes and RSS are in /Users/ravd/agent-share/cffg-5edo-envelope/resource-envelope.json and baseline-a774a54. No continuation study has started.

H5 float64 certified three actions and used 0.39 seconds at each of the two follow-up decisions. H20 float64 certified three actions in 168, 166 and 172 iterations. First action took 51.48 seconds and follow-up decisions took 32.24 and 22.84 seconds. Peak RSS was 1.50 GiB. An independent paired probe's cold follow-ups spent 31.95 and 20.69 seconds in the solver with zero solver compilations. An arithmetic extrapolation from the public follow-ups is about 63–90 hours of planning for 10,000 decisions on this CPU. This is a resource warning from two observations, not a steady-state estimate.

H25 float64 at caps 120 and 400 repeatedly failed with local_budget at obs21 on soft iteration 4. Reported local residual was 1.7053025658242404e-13 against 512*eps=1.1368683772161603e-13. H25 float32 at cap 400 failed at obs14, soft iteration 73, with 6.151199340820312e-5 against 6.103515625e-5. H50 float64 at cap 400 failed local_budget at obs12 on soft iteration 4. All three returned budget_exhausted without an action and leave the exact tolerance unchanged. The stop is inside one GFE composite before any certified outer root, independently of the H25 outer cap.

The H20 shifted prototype copied only the 1,463 overlapping GFE scatter values and verified each copy element for element. The two matched cold trials certified and each shifted trial failed the local solver. A third shifted rejection was recorded before its cold mate reached the 115-second process deadline. No timing result can establish the 3x success criterion. The 100-seed behavior and 10,000-decision storage studies remain open. A bounded separate replay extracts the actual H25 failing cavity, holds the local threshold fixed, and checks whether solving the algebraically equivalent correction from the cavity removes roundoff independently of the outer solver.

**2026-11-28T13:52:20Z**

Bounded actual-cavity replay is complete. Both 57/60 support cases hit the same above-threshold residual at local caps 64 and 256. In the diagnostic, solving for the correction relative to the incoming cavity gave 6.22e-15 float64 and 1.67e-6 float32 in one update at unchanged 512*eps. Decimal recomputation of the centered active KKT residual on the returned float64 posterior gave 1.43e-14. An independent Decimal log-space root matched the materialized float64 log probabilities within 2.85e-14. Float32 materialized only one of 57 supported probabilities at this cavity, so preserve finite log support in future checks and report underflow independently. Source and evidence: /Users/ravd/agent-share/cffg-5edo-envelope/{cavity.py,cavity-f64.json,cavity-f32.json,cavity-oracle.py,independent-cavity-oracle.json}. This is an isolated local diagnostic. The full H25 planner and warm speed remain unqualified.

Legend: prerequisite --> dependent
[ ] clr-xtni local GFE residual (ready) --> [ ] cff-5edo public rollout budget and shifted start --> [ ] cff-t2g3 control benchmark

**2026-11-28T15:18:58Z**

Local-fix candidate 561ba37 now certifies the bounded H25 startup witness, where published a774a54 withheld action. The same frozen ec51aea corridor, seed 0 and start 47 were used. At cap 400 and requested tol=1e-10, float64 took 57.85 seconds for the first decision and certified the outer root in 194 iterations (outer 3.55e-11, local 4.55e-14). Float32 took 27.94 seconds and certified in 166 iterations (outer 2.72e-6, local 5.77e-5). Both chose and actually applied action 1, observed 2, and left the MAP RNG untouched. The separate raw-precision cavity witness has 57/60 semantic support in both dtypes; 56 supported probabilities underflow in float32. Raw records: /Users/ravd/agent-share/cffg-5edo-envelope/production-correction. The branch has a separate local-solver review. One startup per dtype establishes neither warm equivalence nor a per-step latency distribution. Keep the horizon-20/25 behavioral comparison and 10,000-decision continuation pending a resource decision and a stable matched pilot.
