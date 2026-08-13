---
id: cff-5edo
status: open
deps: [cff-wbhu]
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
2. `RollingAgent` gates actions on planning status and exposes planning diagnostics.
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
