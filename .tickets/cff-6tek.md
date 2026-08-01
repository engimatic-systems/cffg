---
id: cff-6tek
status: open
deps: [cff-x0sv]
links: [cff-l6i5]
created: 2026-07-27T20:52:13Z
type: feature
priority: 2
assignee: coopwilliams
---
# Learnables across non-discrete families: member_of beyond Cat/Dir, NIW/MNIW registration

## Background in Plain English

Models can declare categorical parameters for learning, but they cannot yet use the declared-learning API for Gaussian parameters. The compiler cannot address one named factor inside several non-discrete parameter groups, even though the Gaussian statistics and posterior formulas already exist. This ticket connects those parts so callers can learn and install Gaussian and Gaussian-linear parameters by factor name instead of editing stacked runtime arrays by row number.

`cff-x0sv` supplies the declared-learning contract, keyed hyperparameters and statistics, multi-blob targets, readout selection, cadence, and checkpoint format, while `cff-7c5o` supplies `Engine.gaussian_node_stats` and the posterior updates in `cffg.conj`. This ticket owns the non-discrete address map, named parameter-write API, NIW and MNIW family adapters, and an automated recovery test that crosses declaration, compilation, accumulation, and parameter installation.

## Outcomes / Acceptance Criteria

1. `Arena.member_of` identifies the runtime parameter destination for each named factor instance that reads parameters through the discrete, Gaussian, hybrid Gaussian-mixture, recurrence, precision, EF, or plate-expanded compiler paths.
   1.1. Addressing one member of a stacked group leaves every sibling member unchanged.
   1.2. The address representation identifies each parameter role for groups that store one factor's parameters under several runtime keys.
   1.3. Scoped names produced by plate expansion resolve to their own member addresses.
2. Named parameter installation has the following `set_params` contract:
   2.1. The package exports `set_params`.
   2.2. `README.md` documents `set_params`.
   2.3. `set_params` accepts updates keyed by factor name and parameter role.
   2.4. `set_params` rejects an unknown factor name.
   2.5. `set_params` rejects an unknown parameter role.
   2.6. `set_params` rejects a non-numeric value.
   2.7. `set_params` rejects a value with an incompatible shape.
   2.8. `set_params` casts each accepted value to its destination array's dtype.
   2.9. `set_params` leaves every runtime parameter member not named by the update unchanged.
3. `Template.learnables` accepts `niw` for `gaussian` factors and `mniw` for `gaussian_linear` factors. Other factor-family pairings fail at compile time with a message that names the declaration and the unsupported pairing.
4. The NIW and MNIW adapters satisfy the following `cffg.learn.Family` requirements:
   4.1. Each adapter reads its sufficient statistics from `Engine.gaussian_node_stats`.
   4.2. Each adapter adds one statistic record for each declared source during one `Learnables.accumulate` call.
   4.3. Each adapter routes each posterior output to its declared parameter role.
   4.4. Each adapter exposes posterior means for reporting.
   4.5. Each adapter validates caller-supplied priors.
5. Tied sources and empty source sets have the following behavior:
   5.1. Several source factors that share one learnable add their statistics into one posterior.
   5.2. A declaration with an explicit empty source set adds no statistics.
   5.3. A declaration with an explicit empty source set writes its posterior to each target that reads the declared blobs.
6. `Learnables.write` installs the parameters consumed by Gaussian message rules. In particular, NIW and MNIW covariance roles receive `psi / nu`, the inverse of the VMP expected precision defined by `cffg.conj`, rather than the posterior mean covariance.
7. An end-to-end Gaussian-linear trial loop uses `Learnables.accumulate` and `Learnables.write`, without direct calls to `Engine.gaussian_node_stats` or manual edits to group rows, and recovers the seeded scalar LGSSM target within the tolerances in `tests/test_conj.py::test_trial_loop_recovers_lgssm_parameters`: matrix error below `0.05`, offset magnitude below `0.05`, and covariance error below `0.03`.
8. Hostile inputs have the following outcomes:
   8.1. A masked Gaussian-linear observation whose readout mask is `0` adds no evidence.
   8.2. A literal zero in a valid moment array remains zero without a probability floor.
   8.3. An improper prior fails before the posterior or a runtime parameter array is changed.
   8.4. A non-finite Gaussian readout fails before the posterior or a runtime parameter array is changed.
   8.5. A Gaussian readout marked `proper=False` fails before the posterior or a runtime parameter array is changed.
   8.6. A statistic with an incompatible shape fails before the posterior or a runtime parameter array is changed.
   8.7. A non-positive-definite posterior parameter fails before the posterior or a runtime parameter array is changed.
9. Float32 execution and numerical guards have the following behavior:
   9.1. Float32 accumulation produces finite values.
   9.2. A write to a float32 destination leaves that destination in float32.
   9.3. Each newly introduced floor, ridge, clamp, or tolerance derives from `finfo` for the numerical dtype.
   9.4. The code states the invariant protected by each newly introduced floor, ridge, clamp, or tolerance.
10. Uncomfortable-scale coverage has the following requirements:
   10.1. The scale fixture has at least 16 continuous dimensions and 64 members.
   10.2. The scale fixture constructs member addresses successfully.
   10.3. The scale fixture completes NIW or MNIW accumulation successfully.
   10.4. The scale fixture completes named parameter writes successfully.
   10.5. The numerical path has no Python loop that processes one group member at a time.
   10.6. The numerical path allocates no array that couples unrelated group members.
11. Checkpoint and template persistence have the following requirements:
   11.1. The versioned `v2:` learnable checkpoint format round-trips NIW hyperparameters with their keyed shapes intact.
   11.2. The versioned `v2:` learnable checkpoint format round-trips MNIW hyperparameters with their keyed shapes intact.
   11.3. Template serialization encodes the learning-family declaration.
   11.4. Template serialization encodes each parameter blob role.
   11.5. The structure hash reflects the serialized learning-family declaration.
   11.6. The structure hash reflects each serialized parameter blob role.
   11.7. Template serialization does not embed a runtime parameter array.
   11.8. The structure hash excludes derived runtime parameter arrays.
12. `README.md`, `docs/ARCHITECTURE.md`, and `docs/LIMITATIONS.md` describe the named-write API, Gaussian families available through runtime learning dispatch, inverse-expected-precision rollover, and each approximation or unsupported non-discrete learning family left after this ticket.

## Methodological Suggestions

These are implementation suggestions. The implementer may choose another method that satisfies the outcomes.

- Use the **Learnable or checkpoint feature** row in `docs/ARCHITECTURE.md` as the change-path checklist. Treat `src/cffg/learn.py` (`Family`, `FAMILIES`, and `Learnables`) as the learning authority and `src/cffg/compiler.py` (`Arena.member_of`, group construction, and `_resolve_learnables`) as the compiled-address authority.
- Reuse the NIW and MNIW algebra in `src/cffg/conj.py` and the statistics from `Engine.gaussian_node_stats`. Keep the family adapter responsible for translating keyed `LearnPlan` statistics and roles rather than copying the posterior formulas.
- Evolve the member address only as far as needed to describe multi-key parameter groups. Derive the named bulk writer from that address so learnable writes and direct caller writes cannot disagree about group rows, role layout, or plate-scoped names.
- Check validation, JSON and structure hashing, plate scoping, `_resolve_learnables`, `Arena.learn_plans`, engine readouts, parameter writes, checkpoint encoding and decoding, bundles, and package exports. Agent cadence remains caller-owned and needs no new policy. Message updates, free-energy evaluation, schedule classification, estimator behavior, and visualization are N/A because this ticket changes between-run parameter addressing and conjugate updates, not inference execution.
- Check the duplicated execution surfaces explicitly: singleton and stacked/`vmap` groups are affected; plate-scoped and unscoped names are affected; versioned checkpoints are affected while legacy checkpoint decoding retains its pre-ticket behavior. Compare flooding and scheduled results separately with the factor-family statistic formulas after each schedule reports convergence. Host/JIT message-update pairs are N/A because this ticket writes parameters between runs.
- Sparse categorical log-support is N/A because NIW and MNIW operate on continuous moments, not categorical log probabilities. Exercise a masked Gaussian-linear observation and a source excluded by `LearnableSpec.sources` as zero-evidence cases. Exercise vacuous input with an empty source set and improper input with `proper=False`, non-finite moments, invalid shapes, and non-positive-definite priors. Use float32 destination arrays for dtype coverage. Use 16 dimensions and 64 same-group members for the uncomfortable-scale case. NIW and MNIW updates are closed-form between-trial operations, so iterative convergence and globalization requirements do not apply.
- Extend `tests/test_learnables.py` for compilation, family validation, tying, named writes, plate scoping, hostile inputs, float32, scale, and checkpoint behavior. Adapt the seeded recovery scenario in `tests/test_conj.py` as the independent behavioral target, and use `tests/test_bundle.py` if bundle persistence changes. Run focused tests during implementation and the repository short suite once after the final code change.

## Future work

`cff-y4nx` owns the consumer workflow for a recurrent switching linear dynamical system that combines Dirichlet transition learning, MNIW dynamics learning, and Pólya–Gamma recurrence-weight learning. Its dependency list reported by `tk show cff-y4nx` on 2026-08-07 does not name `cff-6tek`, so that ticket must either use this declared MNIW path or state that its manual between-trial MNIW loop is intentional.

`cff-ahps` owns the broader universal-factor-library roadmap. Learning-family adapters beyond NIW and MNIW remain with the roadmap ticket or a child ticket it names; this ticket does not silently promise adapters for every non-discrete posterior family.

## Definitions

- **Arena:** The compiled collection of grouped message and parameter arrays used by the runtime.
- **`Arena.member_of`:** The compiler-produced map from a factor instance name to that instance's location in grouped runtime parameters.
- **API (application programming interface):** A documented operation that callers may rely on.
- **Bundle:** A saved model package that can include serialized structure and learning state.
- **Cat/Dir:** Shorthand in the title for categorical factors with Dirichlet parameter posteriors.
- **Checkpoint:** Serialized learning state that can be loaded in a later process or session.
- **Conjugate posterior:** A posterior distribution in the same family as its prior, updated through sufficient statistics.
- **Dtype:** A numeric storage type such as float32 or float64.
- **EF (exponential family):** A probability-distribution family represented by sufficient statistics and natural parameters.
- **`finfo`:** Numeric limits, including machine epsilon, for a floating-point dtype.
- **Flooding and scheduled inference:** Two execution orders for message updates. Flooding updates every eligible message together; a schedule updates selected groups in a planned order.
- **Globalization:** Damping or line-search behavior that keeps an iterative numerical solver from taking an unstable full step.
- **Host:** Python-side execution outside JIT-compiled numerical code.
- **JAX:** The array and compiled numerical computing library used by the runtime.
- **JIT (just-in-time compilation):** Compilation of numerical operations before repeated execution.
- **JSON (JavaScript Object Notation):** The text format used to serialize template structure.
- **LGSSM (linear Gaussian state-space model):** A time-series model with linear transitions and Gaussian noise.
- **NIW (Normal-Inverse-Wishart):** A conjugate posterior family for a Gaussian mean and covariance.
- **MNIW (Matrix-Normal-Inverse-Wishart):** A conjugate posterior family for a linear map, offset, and covariance.
- **Parameter blob:** A named serialized array that stores model parameters and may be shared by several factors.
- **Parameter role:** The meaning of one parameter blob within a factor, such as matrix, offset, or covariance.
- **Plate expansion:** Compilation of a repeated model fragment into scoped factor instances.
- **Precision:** The inverse of a covariance matrix.
- **Pólya–Gamma:** A probability distribution used to express logistic-regression likelihoods as conditionally Gaussian updates.
- **Readout:** A derived statistic or belief summary obtained from runtime state without changing that state.
- **Rollover:** Conversion of posterior hyperparameters into the parameters consumed during a later inference run. For NIW and MNIW covariance roles, this is the inverse expected precision.
- **Scoped name:** A factor or blob name prefixed with its composed fragment or plate-copy path.
- **Stacked group:** One runtime array whose leading rows hold several factor instances with compatible shapes.
- **Structure hash:** A digest of serialized model structure that excludes derived runtime arrays.
- **Sufficient statistics:** Values from a factor belief that contain the information needed for a conjugate posterior update.
- **VMP (variational message passing):** Message passing that uses expectations under variational beliefs; its Gaussian rules consume a covariance equal to the inverse expected precision rather than the posterior mean covariance.
- **`vmap`:** JAX's transformation for applying one array function across a batch axis.

## Notes

**2026-07-27T21:57:56Z**

Scope narrowed by review of cff-x0sv (2026-07-27).

The structural half of this ticket landed in cff-x0sv instead, because a
reviewer correctly pointed out that deferring it meant the contract did not
actually admit NIW/MNIW, whatever the docstring claimed. Now done:

- hyperparameters, statistics and rollover outputs are keyed dicts, not bare
  arrays, so a posterior with four hyperparameters and three statistics fits
- a Family names which engine readout supplies its statistics
  (cffg.learn.READOUTS), so a family reading gaussian_node_stats rather than
  node_beliefs needs no change to the runtime
- LearnTarget carries the blob it writes and the parameter role that blob
  fills, so one MNIW rollover routes into a gaussian_linear's separate
  matrix, offset and cov blobs
- LearnPlan carries per-blob shapes, which differ across a joint family

What remains here is genuinely registration plus the map it needs:

1. extend Arena.member_of past the discrete families, so every parameterized
   factor has an addressable per-member slot (Gaussian, EF, hybrid and
   plate-expanded group builders all skip it today). A Gaussian learnable
   currently raises a compile error naming exactly this.
2. register NIW and MNIW against the Family protocol, wrapping the updates
   already in cffg.conj.
3. the bulk set_params(factor_name -> arrays) helper from the 2026-07-10
   evaluation note, which falls out of the same map.

**2026-08-08T04:19:36Z**

Dependency map used for this rewrite. Arrow direction is prerequisite --> dependent.

```text
[x] cff-x0sv declared-learning contract
          |
          +--> [ ] cff-6tek non-discrete learning (ready)
```

`tk dep tree cff-6tek --full`, `tk ready`, and `tk blocked` show no recorded dependent below `cff-6tek`. The Future work section names `cff-y4nx` as the consumer-workflow owner and `cff-ahps` as the roadmap owner without presenting either contextual relationship as a dependency record.

Retained wiggle candidates after the bounded current scan (candidate -> intended meaning): end_to_end -> consumer_workflow in criterion 7; n_a -> a change-path surface that is structurally irrelevant for the stated reason; preserve -> compatibility or preserved_state for historical text and the legacy decoder; register and available -> runtime_family_coverage through `cffg.learn.FAMILIES`; reusable -> code_reuse of `cffg.conj`; same -> object_identity, value_equality, or shared_category as bounded by the named map, state, or group; expose -> public_api for posterior reporting; implement -> registration_exists, component_integration_exists, or planned_work as selected by the sentence; support -> probability_support; complete -> exhaustive coverage of the named protocol or execution_finished for a named scale-fixture operation; done -> implementation_complete in the preserved historical note; land -> component_landed in the preserved historical note; ready -> dependency_ready according to `tk ready` on 2026-08-07; remaining -> unfinished_required_work or residual_set; approximate -> a named mathematical approximation; close -> mathematical_form in “closed-form”; conditional_discretion -> the method permits another address representation; sufficient -> statistical_sufficiency; check -> manual_inspection of named change-path surfaces or automated_check of named behavior; converge -> runtime_status_criterion; correct -> contract_conformance in the preserved review note; estimate -> the named resource estimator; evidence -> automated test evidence; exact -> literal_zero_or_support or identifier_precision as selected by the sentence; pass -> message_passing or tests_succeed; test -> test_artifact or automated_execution; validate -> input_validation; accumulate -> raw_sum of named sufficient-statistic records; fit -> structural_compatibility in the preserved note; infer -> state_or_regime_posterior; learn -> integrated_learning_workflow or the named `cffg.learn` module; learnable -> declared_parameter_artifact or registered_parameter_family as selected by the sentence; recover -> parameter_estimation against the seeded target; rollover -> point_parameter_install in the new contract and posterior_to_prior in the preserved historical note; trial -> independent_dataset; scale -> domain_size or dtype-relative numerical magnitude; can -> demonstrated_capability; expected -> mathematical_expectation; intended -> semantic_purpose recorded by this review; may -> optional_requirement or possible sharing; explicit_owner and named_ticket_handoff -> current-ticket ownership or the concrete `cff-y4nx` and `cff-ahps` future owners named by the sentence; all -> named_class_exhaustive in the preserved note; at_least -> inclusive_lower_bound; every -> named_class_distributive; only -> hard_exclusion, current_limitation, or ticket_scope_boundary as bounded by the sentence; current -> versioned runtime checkpoint state in the new text and unpinned present time in the preserved historical note; deferred -> delegated_ticket in the preserved historical note; later -> ordering_constraint; read -> machine_access.

Intelligible-ticket review: complete.

**2026-08-01T05:06:38Z**

NASCAR consumer finding, 2026-07-31: the independent cff-y4nx harness at /Users/ravd/projects/experiments/cffg-nascar-validation commit 0543683 could run fixed-emission oracle inference but could not express the canonical observed-only fit through packaged learning. Shared gaussian_linear emission MNIW registration is now a concrete downstream requirement of cff-l7y4, not only an API-completeness argument. Source: results.json canonical_fit_readiness and primary-agent code audit. Confidence: high.
