---
id: cff-t2g3
status: open
deps: [cff-5edo, cff-x0sv]
links: [cff-cz6v]
created: 2026-07-05T23:57:38Z
type: task
priority: 3
assignee: coopwilliams
parent: cff-21ta
---
# Optimize glucose benchmark with CFFG agent

## Background in Plain English

This task owns the CFFG-controlled simglucose experiment tracked by parent epic `cff-21ta`. The reference benchmark in `src/cffg/benchmarks/glucose.py` uses ten adults, three scheduled meals, and 24-hour episodes. Its ReactiveController recorded 87.6% time in range and 1.7% time below range. That result is useful for the reference protocol, but it does not measure Cooper's requested experiment.

Cooper's experiment uses adults and adolescents, eight-hour episodes, and meals that are not announced to the controller. The evaluated controller adjusts basal insulin and does not administer bolus insulin. Cooper's earlier hand-tuned work recorded 67% time in range with zero critical episodes on this protocol. This task must beat that time-in-range result without introducing a critical episode. The numeric critical-episode threshold and required duration remain undefined, so Cooper must confirm them before the final evaluation protocol is encoded.

The controller must demonstrate continual learning through an inspectable CFFG model and a modular physiological model. The reference benchmark lacks the adolescent cohort, eight-hour evaluation protocol, and critical-episode counter, so this task owns those benchmark extensions as well as the controller.

The task depends on `cff-5edo`, which owns convergence and cost for planning beyond about 20 slices, and closed ticket `cff-x0sv`, which supplied declared learnables and checkpointable parameter updates. Closed tickets `cff-pksr` and `cff-dfpi` provide the repository's discrete-regime and continuous-state route for hybrid epistemic planning if the implementer uses it.

## Outcomes / Acceptance Criteria

1. The benchmark implements Cooper's comparison protocol.
   1.1. The scored cohort includes the ten named adult patients and ten named adolescent patients supplied by simglucose.
   1.2. Each scored episode lasts eight virtual hours, and the recorded evaluation set states its start times, meal timing and amounts, and simulator seeds.
   1.3. The controller receives glucose-sensor observations without meal events, meal amounts, or future meal schedules.
   1.4. The controller's permitted insulin actions adjust basal delivery; every scored bolus action is zero.
   1.5. Cooper's confirmed critical-episode threshold and duration are recorded in this ticket before final evaluation, then encoded in the benchmark counter and tests.
   1.6. The benchmark records simulator blood glucose from 70 through 180 mg/dL as time in range and separately records time below range, time above range, and critical episodes.
2. The evaluated controller satisfies the declared white-box continual-learning contract.
   2.1. The controller is a continual-learning CFFG agent with separately inspectable insulin-response and meal-response model components.
   2.2. A neural-network action policy does not choose insulin actions.
3. The serialized template and referenced parameter data are the source of truth for model structure.
   3.1. The compiled agent is reconstructible from those artifacts.
   3.2. A per-patient checkpoint round trip returns equal learned-parameter values, episode position, and next action when given an identical observation and random key.
4. Stochastic controller behavior satisfies the reproducibility contract.
   4.1. Every stochastic controller operation uses an explicit JAX random key.
   4.2. Repeating an evaluation from identical model artifacts, checkpoints, simulator seeds, and keys reproduces its actions and metrics.
5. Forbidden actions have exact zero probability after the safety mask. Across focused and scored runs, the agent emits no bolus action and no basal action outside the documented action set, including when preferences or model factors assign exact zero support to other actions.
6. A float32 run with the full planned state representation and an eight-hour horizon satisfies the numerical survival contract.
   6.1. The run produces finite messages, beliefs, free-energy values, and actions when its inputs are valid.
   6.2. The run reports converged, budget-exhausted, diverged, or infeasible status truthfully.
   6.3. A non-finite value or missing or non-finite sensor sample cannot be scored as a successful episode.
7. The preregistered evaluation set produces zero critical episodes and combined time in range greater than 67%. Results are also reported by cohort, patient, and episode so an aggregate cannot hide a patient or cohort collapse.
8. The evaluation record distinguishes observations used for continual updates from episodes used for the reported comparison. It identifies the model artifact, checkpoint policy, controller configuration, permitted actions, patient set, episode set, random seeds and keys, runtime statuses, time-in-range values, below-range values, above-range values, and critical-episode counts.
9. For identical patient state, model state, observation, and random key, single-patient execution and cohort-batched execution produce matching actions and learned-state updates within the repository's documented numerical tolerances.
10. Focused automated tests cover the benchmark protocol, prohibited bolus actions, exact-zero action support, float32 execution, deterministic replay, checkpoint replay, runtime failure reporting, criterion 7's scored result, and the safety criteria. Long multi-episode evaluations are marked `slow`.
11. The implementation removes `docs/todo-notes/glucose-cffg-agent.md` in the commit that makes the todo note obsolete.
   11.1. `README.md` documents the runnable controller workflow.
   11.2. `docs/ARCHITECTURE.md` documents the model or agent integration.

## Methodological Suggestions

The implementer may choose another sound method. The following approach is a starting point:

- Confirm the critical-episode definition with Cooper before changing the metric. Freeze and record the scored patient, episode, meal, seed, and random-key set before the final tuning round.
- Extend `src/cffg/benchmarks/glucose.py` and `tests/test_glucose.py` from the reference protocol instead of silently changing the historical 24-hour result. Keep reference and comparison protocols separately named.
- Represent a compact physiological state with factored blood-glucose level and trend, insulin on board, and carbohydrates on board. Keep insulin-response and meal-response parameters separate and update each component from its relevant observations.
- Consider a receding-horizon `RollingAgent` at the sensor cadence, one checkpointed agent state per patient, and `Colony` for cohort execution. Use the declared-learnables API from `cff-x0sv` for patient-specific updates.
- Consider the discrete-regime hybrid path from `cff-pksr` and `cff-dfpi` when a purely discrete glucose model loses too much physiological information. Keep epistemic value on declared discrete regimes and retain the documented free-energy bound when using that path.
- Encode the permitted basal doses as a finite action set documented before evaluation. Apply a hard action mask after policy inference so bolus and unsafe basal entries retain exact zero probability. Shape preferences asymmetrically so low glucose carries the stronger penalty, but do not treat preferences as a substitute for the action mask.
- Use `uv run pytest -q tests/test_glucose.py` for focused checks. Mark randomized, multi-patient, or repeated eight-hour evaluations `slow` and run them explicitly with `uv run pytest -m slow` when validating this behavior. After the final code change, run the repository's one required short-suite dashboard command.
- The `docs/ARCHITECTURE.md` change-path index does not include a row for a benchmark/controller addition. If implementation changes the agent's learnable or checkpoint contract, follow the **Learnable or checkpoint feature** row. If it changes convergence or runtime statuses, follow the **Runtime numerical or status invariant** row. If it adds a factor or message rule, split that core change into a prerequisite ticket and follow the **New factor or rule family** row.
- Review the single-instance/`vmap` pair because this task owns patient and cohort execution. Review the new/legacy-checkpoint pair when checkpoint encoding changes. Host/JIT and flooding/scheduled pairs are not affected when the task consumes unchanged engine behavior; revisit them in a prerequisite ticket if engine behavior changes.

## Future work

Open ticket `cff-o7hs` owns nonlinear Gaussian perception factors. If this task demonstrates that the repository's discrete and hybrid model families cannot satisfy the outcomes without nonlinear Gaussian maps, add `cff-o7hs` as a dependency instead of implementing those factors here.

Parent epic `cff-21ta` owns the project-level closure record after this task closes. No other future work is identified.

## Definitions

- **Basal insulin:** Insulin delivered continuously at an adjustable background rate.
- **Bolus insulin:** An additional insulin dose delivered for a meal or glucose correction.
- **Blood glucose (BG):** The simulator's modeled blood-glucose value, measured in milligrams per deciliter.
- **Carbohydrates on board:** A modeled quantity for meal carbohydrates that are still affecting blood glucose.
- **CFFG:** Constrained Forney-style Factor Graph, the model and inference system implemented by this repository.
- **Checkpoint:** Saved per-patient agent state used to resume inference and learning.
- **Colony:** The repository component that coordinates multiple agent instances with compatible structures.
- **Continual learning:** Updating model parameters from new observations collected as the controller operates.
- **Critical episode:** A serious low- or high-glucose event. Cooper must define its numeric threshold and required duration for this experiment.
- **Epistemic planning:** Choosing actions partly for the information they are expected to reveal.
- **Exact zero support:** A prohibited state or action represented with literal zero probability rather than a positive numeric floor.
- **Free energy:** The CFFG objective reported by inference and used to assess convergence and policies.
- **Hybrid model:** A model that combines discrete regimes with continuous states.
- **Insulin on board (IOB):** A modeled quantity for delivered insulin that is still active in the body.
- **JAX random key:** An explicit value that controls a reproducible stochastic operation in JAX.
- **JIT:** Just-in-time compiled execution.
- **Receding horizon:** Replanning after each new observation over a forward window of future states.
- **ReactiveController:** The repository's hand-written reference controller based on recent glucose level and trend.
- **Safety mask:** A final action filter that assigns zero probability to prohibited actions.
- **Time in range (TIR):** The fraction of recorded simulator blood-glucose samples from 70 through 180 mg/dL, inclusive.
- **`vmap`:** JAX execution that applies one function across a batch of patient states.
- **VERSES-era:** Cooper's earlier hand-tuned glucose-controller experiments conducted during his time at VERSES.
- **White-box agent:** An agent whose model structure, beliefs, updates, and action scores are inspectable.


## Notes

**2026-07-06T00:06:03Z**

THE BAR (from Cooper, 2026-07-05): his VERSES-era hand-tuned experiments = 67% TIR on unannounced ADULTS + ADOLESCENTS, 0 critical episodes, 8-virtual-hour episodes. Goal: beat that with a continual-learning white-box CFFG agent + modular physiological world model. Current reactive controller (87.6%) is NOT comparable (adults-only, 24h, no critical-episode metric). Do not spend effort optimizing until CFFG core is done. When starting: extend benchmark with adolescent cohort, 8h episodes, critical-episode counter (BG<54 or <40 thresholds - confirm definition with Cooper).

**2026-08-08T04:19:57Z**

Bounded scan classifications: integrate=internal_component_wiring; implement=external_harness_or_planned_work_or_component_integration; provide=internal_mechanism; support=literal_probability_support; deliver=physical_insulin_administration; done=historical_implementation_complete; remaining=unfinished_required_work; close=work_item_completion; check=automated_check; converge=runtime_status_criterion; exact=literal_zero_or_identifier_precision; match=tolerance_agreement; reproduce=exact_result_and_rerunnable_artifact; sound=optional_method_choice; test=automated_execution_or_test_artifact; validate=scientific_validation; infer=state_or_policy_posterior; learn=new_evidence_accumulation_or_integrated_learning_workflow; learnable=integrated_fitting_surface; expected=mathematical_expectation; may=optional_requirement; explicit_owner=current_or_named_ticket_owner; named_ticket_handoff=concrete_current_or_future_owner; every=named_class_distributive; only=historical_hard_exclusion; current=historical_latest_evidence; complete=component_complete.

Intelligible-ticket review: complete.
