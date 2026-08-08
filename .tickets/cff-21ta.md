---
id: cff-21ta
status: open
deps: [cff-t2g3]
links: []
created: 2026-07-05T23:57:38Z
type: epic
priority: 2
assignee: coopwilliams
---
# SG: simglucose TIR benchmark (side project)

## Background in Plain English

This epic tracks a side project that evaluates CFFG-based blood-glucose control in the simglucose simulator. The repository already contains a reference benchmark in `src/cffg/benchmarks/glucose.py` and focused automated checks in `tests/test_glucose.py`. The controller receives Dexcom continuous glucose monitor readings but no meal notification. The 24-hour reference scenario gives ten adults meals of 45 grams at 07:00, 70 grams at 12:00, and 80 grams at 18:00 through an Insulet pump simulation.

The reference ReactiveController detects rising glucose, gives patient-adjusted insulin corrections, and suspends insulin when glucose is low or falling quickly. Its recorded result was 87.6% cohort time in range and 1.7% time below range; adult #006 recorded 5.8% time below range. Results recorded for that reference scenario were 95.4% for an announced-meal basal-bolus controller, 33.5% for basal insulin without meal response, and 74.7% for a prior `pymdp` attempt. The repository uses blood glucose from the simulator, not the sensor measurement, to calculate the fraction of samples from 70 through 180 mg/dL and cites 70% as the FDA target.

These reference results do not establish the final project goal. Cooper's comparison protocol covers adults and adolescents, unannounced meals, eight-hour episodes, and critical-episode accounting. Its historical result was 67% time in range with zero critical episodes. The 87.6% result uses adults, 24-hour episodes, and no critical-episode metric, so the two percentages do not measure improvement under a common protocol.

Child ticket `cff-t2g3` owns the CFFG controller and the comparison protocol. It depends on open ticket `cff-5edo` for planning convergence and cost beyond about 20 slices. Its declared-learnables prerequisite, `cff-x0sv`, is closed.

## Outcomes / Acceptance Criteria

1. The repository provides an executable, deterministic reference benchmark for the ten-adult, 24-hour, three-meal protocol described above.
2. Focused tests demonstrate the reference benchmark contract.
   2.1. The ReactiveController reaches at least 70% cohort time in range.
   2.2. Every adult reaches at least 50% time in range and spends at most 10% of samples below 70 mg/dL.
   2.3. The basal-insulin reference for adult #001 stays below 50% time in range.
   2.4. Repeated runs for adult #002 produce identical blood-glucose samples.
   2.5. The cohort contains `adult#001` through `adult#010`.
3. Evidence used to close this epic identifies each result's controller, patient cohorts, episode duration, meal-notification policy, insulin-action policy, simulator seed or evaluation-sampling procedure, time-in-range value, below-range value, critical-episode value when defined, and producing repository artifact.
4. The epic's final comparison identifies results that use identical protocol settings. It does not use the 87.6% adult 24-hour result as evidence of improvement over Cooper's 67% adult-and-adolescent eight-hour result.
5. This epic closes only after `cff-t2g3` closes.
6. The evidence that closes `cff-t2g3` is linked or copied into this ticket's historical notes.

## Methodological Suggestions

The implementer may choose another sound method, but the following approach should make the outcomes easy to review:

- Treat `src/cffg/benchmarks/glucose.py` as the reference benchmark implementation and `tests/test_glucose.py` as its focused executable evidence.
- Run `uv run pytest -q tests/test_glucose.py` after benchmark behavior changes. Record the controller configuration and result summary in the epic's closure record.
- Present benchmark results in a table with separate columns for patient cohorts, episode duration, meal notification, permitted insulin actions, simulator seed or sampling procedure, time in range, time below range, and critical episodes. Compare percentages directly when those protocol columns agree.
- Keep meal notifications outside the environment observations presented to an evaluated controller. Use simulator blood glucose for the recorded time-in-range metric and label sensor-derived diagnostics separately.
- The `docs/ARCHITECTURE.md` change-path index has no row for an isolated side-project benchmark. For this epic's tracking work, the host/JIT, flooding/scheduled, single-instance/`vmap`, and new/legacy-checkpoint review pairs are not applicable. `cff-t2g3` should name a change-path row and revisit each pair if its implementation changes the inference runtime, learning API, agent state, or checkpoint behavior.
- A code change used to close the epic should follow the repository's focused-test and final short-suite policy. This ticket rewrite itself requires no pytest run.

## Future work

Child ticket `cff-t2g3` owns the remaining implementation and evaluation work for this epic. It will:

- build a continual-learning, white-box CFFG controller with a modular physiological model;
- extend the benchmark to adults and adolescents over eight-hour episodes with unannounced meals;
- confirm the critical-episode definition with Cooper before encoding that metric;
- evaluate safety and time in range under the common comparison protocol; and
- address patient-specific low-glucose behavior, including the historical adult #006 finding, through evidence from the common protocol rather than tuning against the 24-hour reference alone.

No additional future work is identified by this epic.

## Definitions

- **Basal insulin:** A continuous background rate of insulin.
- **Basal-bolus controller:** A controller that combines background insulin with additional insulin doses for meals or high glucose.
- **Blood glucose (BG):** The simulator's modeled blood-glucose value, measured here in milligrams per deciliter (mg/dL).
- **CFFG:** Constrained Forney-style Factor Graph, the model and inference system implemented by this repository.
- **Cohort:** The named set of simulated patients included in one aggregate result.
- **Continuous glucose monitor (CGM):** A sensor that supplies glucose readings to the controller.
- **Continual learning:** Updating a model from observations collected while the controller operates.
- **Critical episode:** A serious low- or high-glucose event. `cff-t2g3` must confirm its specific threshold and duration with Cooper before implementation.
- **FDA:** U.S. Food and Drug Administration.
- **Hypoglycemia:** Blood glucose below the accepted range; this ticket's reference below-range threshold is 70 mg/dL.
- **Insulet:** The simulated insulin pump selected for the reference scenario.
- **`pymdp`:** A Python library for active-inference models, used by the prior attempt cited as a reference result.
- **ReactiveController:** The repository's hand-written reference controller based on recent sensor level and trend rather than CFFG inference.
- **SG:** The simglucose side project named in this epic's title.
- **Time in range (TIR):** The fraction of recorded simulator blood-glucose samples from 70 through 180 mg/dL, inclusive.
- **White-box controller:** A controller whose model structure and inference steps are inspectable rather than hidden behind opaque numerical mappings.


## Notes

**2026-07-05T23:57:38Z**

Harness + reactive controller landed and green; epic stays open for the optimization stage (cff-t2g3).

**2026-08-08T04:10:49Z**

Bounded wiggle scan retained these candidates with the following intended meanings: implement = the external benchmark harness, the integrated repository system, or planned cff-t2g3 work as specified by each sentence; provide = the executable benchmark workflow; land = the historical statement that the named harness and controller components entered the repository, whose paths are now identified in Background; remaining = unfinished work assigned to cff-t2g3; close = administrative ticket completion or acceptance completion; check = an automated check; converge = the runtime stopping criterion owned by cff-5edo; evidence = automated test output, an empirical result, or a provenance record as specified by each sentence; sound = reasonable judgment; test = a named automated artifact or its execution; infer = probabilistic state or policy computation; learn = accumulation from operating observations or the integrated learning workflow; learnable = the integrated declared-parameter workflow delivered by cff-x0sv; may = freedom to choose another method; should = a methodological recommendation or expectation; will = the stated cff-t2g3 commitment; explicit_owner = named-ticket ownership by cff-t2g3; named_ticket_handoff = cff-t2g3 as the concrete future owner; at_least = an inclusive numeric lower bound; every = each member of the named ten-adult cohort; integrate = internal_component_wiring; complete = ticket_scope_complete; deliver = integration_delivered; appropriate = evidence_based_judgment; accumulate = distinct_evidence; intended = semantic_purpose.

Intelligible-ticket review: complete.
