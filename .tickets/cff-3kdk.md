---
id: cff-3kdk
status: open
deps: []
links: [cff-d56d]
created: 2026-07-27T05:44:29Z
type: task
priority: 2
assignee: coopwilliams
tags: [performance]
---
# Vectorize categorical and Gaussian engine readouts

## Background in Plain English

`Engine.marginals` and `Engine.gaussian_marginals` turn a runtime state into posterior summaries keyed by edge name. At repository commit `b09a541`, both methods perform the numerical calculation once per edge in Python. The benchmark completed under cff-d56d recorded Gaussian readout latency rising from 46 milliseconds at 16 time steps to 773 milliseconds at 256 time steps on its original two-dimensional linear Gaussian state-space model. Once cff-d56d made hidden-state posterior calculation scale linearly with chain length, that readout latency became the main cost of obtaining the posterior summaries. This ticket batches calculations that share a shape without changing the reported keys, shapes, mathematical meaning, or invalid-belief behavior of those two methods.

## Outcomes / Acceptance Criteria

1. For a given `Arena` and `State`, the edge-name key set and array shape under each key from `Engine.marginals` equal those from the commit `b09a541` scalar calculation. Each returned array still represents the normalized product of that edge's two categorical messages.

2. For a given `Arena` and `State`, the edge-name key set and `(mean, covariance)` shape under each key from `Engine.gaussian_marginals` equal those from the commit `b09a541` scalar calculation. Each returned pair still contains the moments of the product of that edge's two Gaussian information-form messages.

3. An automated scalar oracle that evaluates one edge at a time confirms the batched readouts on models with mixed categorical cardinalities and mixed Gaussian dimensions.

   3.1. Categorical probabilities agree with the scalar oracle using relative and absolute tolerances of `128 * finfo(dtype).eps`. Literal log-zero entries remain literal probability zeros when the combined belief has one or more supported categories.

   3.2. Gaussian means and covariances agree with the scalar oracle using relative and absolute tolerances of `128 * finfo(dtype).eps`. A proper combined precision produces finite moments.

4. Hostile-input checks make the public behavior explicit.

   4.1. Two vacuous categorical messages produce a uniform posterior. One vacuous categorical message acts as the product identity. A combined categorical belief with no supported category returns non-finite values and cannot resemble a valid probability vector.

   4.2. A Gaussian edge with a proper combined precision reports finite moments even when one incoming message is vacuous. A Gaussian edge whose combined precision is vacuous, singular, indefinite, asymmetric beyond the dtype-scaled tolerance, or non-finite reports shape-preserving `NaN` mean and covariance arrays.

   4.3. Float32 inputs stay in float32, preserve exact categorical zeros, and use dtype-scaled numerical checks. The readout implementation introduces no fixed absolute floor, ridge, or clamp.

   4.4. The categorical and Gaussian readouts each pass a 256-step chain case, which has roughly 50 times as many time steps as the five-step focused Gaussian fixture.

5. The numerical work in each public readout is batched by categorical cardinality or Gaussian dimension. No Python loop in the numerical readout path performs one normalization, matrix solve, or matrix inverse per edge.

6. On an Apple M2 Max CPU with JAX x64 enabled, the median warmed wall time of the complete public `Engine.gaussian_marginals` call for the cff-d56d scheduling benchmark's 256-step linear Gaussian state-space model is one tenth or less of the unvectorized call measured with the same inputs, process setup, and repetition protocol. The timed boundary starts with the converged `State` and includes creation of the returned edge-keyed dictionary; it excludes inference.

7. The focused categorical, Gaussian, schedule, and benchmark checks pass without weakening their numerical assertions.

8. The benchmark evidence records the unvectorized and vectorized medians, hardware, dtype mode, model dimensions, number of repetitions, and speed ratio.

## Methodological Suggestions

The implementer may choose a different method that satisfies the outcomes. One likely approach is to extend the `Arena` with deterministically ordered edge groups keyed by categorical cardinality and Gaussian dimension, following `Arena.interior_slot_groups`. Gather both message slots for each edge in a group, form the group's combined natural parameters in one operation, and evaluate categorical or Gaussian moments across the leading group axis. Assemble the `Engine.marginals` and `Engine.gaussian_marginals` dictionaries after the grouped numerical work finishes.

The change-path index in `docs/ARCHITECTURE.md` has no exact row for a host readout optimization. Treat `src/cffg/engine.py`'s public readout methods as the authority. If compilation supplies new grouped indices, keep `src/cffg/compiler.py:Arena` and the public readouts in agreement. No new intermediate representation field, factor family, runtime status rule, learnable, serialization field, or checkpoint behavior is intended.

Review the duplicated execution surfaces as follows:

- Host/JIT: compare any new JAX or JIT-compiled grouped calculation with the scalar host oracle. Keep dtype and invalid-belief behavior equal at the public boundary.
- Flooding/scheduled: both schedules use the same readout implementation. Check representative states from each schedule; no second implementation should be added.
- Single-instance/`vmap`: the public methods read one `State`, and `Colony` delegates to that same API per instance. Instance-level `vmap` readout behavior is not part of this ticket.
- Current/legacy checkpoint decoding: not applicable because the ticket changes neither stored state nor decoding.

Focused checks may extend `tests/test_engine_hmm.py` and `tests/test_kalman.py`, plus a narrowly named readout test file if mixed-shape and hostile cases are clearer there. `tests/test_schedule.py` can confirm schedule-independent readout parity. Run the parity-asserting benchmark with `uv run python -m cffg.benchmarks.scheduling --reps 40`; preserve a pre-change run under the same conditions for the speed ratio. After the final code change, run the repository's required short suite once through `uv run python tools/dashboard.py --run-suite`.

## Future work

No future work is delegated by this ticket. Ticket cff-d56d has status `closed`, and its schedule and benchmark code is present at repository commit `b09a541`; the link supplies context rather than an unresolved prerequisite.

## Definitions

- **Cardinality:** The number of possible values of a categorical variable.
- **Combined belief:** The product of the two incoming edge messages, represented by addition in natural-parameter or log space.
- **Dtype:** A numeric storage type such as float32 or float64.
- **Gaussian dimension:** The number of coordinates in one Gaussian variable.
- **JAX:** The array and compiled numerical computing library used by the runtime.
- **JIT:** Just-in-time compilation of numerical work before repeated execution.
- **Linear Gaussian state-space model:** A time-series model with linear transitions and observations and Gaussian noise.
- **Natural parameters:** The log-space parameters in which exponential-family message products become addition. Gaussian natural parameters are an information vector and precision matrix.
- **Proper combined precision:** A finite, positive-definite precision matrix whose asymmetry does not exceed the dtype-scaled predicate implemented by `efrules.GAUSSIAN.is_proper_precision`.
- **Readout:** A derived posterior summary returned from a runtime `State`; it does not mutate that state.
- **Scalar oracle:** A simple reference calculation that processes one edge at a time and is used to check the batched result.
- **Vacuous message:** The multiplicative identity message. It contributes no information when combined with another message.
- **Warmed wall time:** Elapsed time measured after compilation and one-time setup have completed.

## Notes

**2026-08-08T04:12:04Z**

Retained wiggle candidates after the bounded scan (candidate -> intended meaning): fixed -> constant_parameter; preserve -> byte_or_value_identity, compatibility, or pinned_artifact as selected by the named object; public -> documented_contract for the two Engine methods; same -> equivalent_configuration or shared implementation; implement -> planned_work or primitive_exists; support -> probability_support; complete -> exhaustive timed call or execution_finished setup; remaining -> preserved_state; close -> work_item_completion for cff-d56d status; check -> automated_check, input_guard, or manual_inspection as named; converge -> runtime_status_criterion; evidence -> empirical_support and provenance_artifact; exact -> literal_zero_or_support or absence of an applicable indexed row; oracle -> test_artifact implementing an independent numeric reference; parity -> tolerance_parity or behavioral_parity; pass -> tests_succeed or survive_without_failure for the scale case; test -> test_artifact or automated_execution; infer -> state_or_regime_posterior; learnable -> declared_parameter_artifact; performance -> steady_latency_or_throughput; scale -> asymptotic_behavior, domain_size, or dtype-relative numeric magnitude; speed -> runtime_latency; can -> demonstrated_capability of a named executable check; intended -> declared_scope; may -> optional_requirement; should -> obligation; any -> unrestricted_choice; current -> the named current/legacy review pair; read -> machine_access. Each occurrence is bounded by its sentence, named method, artifact, metric, or repository reference.

Intelligible-ticket review: complete.
