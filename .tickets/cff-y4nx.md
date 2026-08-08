---
id: cff-y4nx
status: open
deps: [cff-k9ws, cff-4apw, cff-twzc]
links: []
created: 2026-07-15T01:46:10Z
type: feature
priority: 2
assignee: coopwilliams
parent: cff-ahps
tags: [hybrid, validation, rslds]
---
# rSLDS tier: model-specific continual learning and fixed-parameter validation

## Background in Plain English

A recurrent switching linear dynamical system tracks a continuous state and a discrete regime whose transition depends on the previous regime and continuous state. Tickets `cff-k9ws`, `cff-lquu`, and `cff-4apw` supply the recurrence rule, soft hybrid statistics, and recurrence-weight update, while this ticket assembles them with switching Gaussian dynamics and shared Gaussian emissions. The model-specific `RSLDSLearning` boundary adds initial-regime, per-regime dynamics, and recurrence statistics from each converged sequence and installs their point parameters for a later trial without rebuilding the graph. Draft PR #56 also records a canonical-scale run in which the harness supplied the generating parameters, so that run validates fixed-parameter inference but does not validate parameter fitting from observations. Observed-only fitting, shared-emission learning, one-sequence replacement updates, initialization, and comparison with a learned ordinary SLDS belong to `cff-l7y4` after `cff-6tek`.

## Outcomes / Acceptance Criteria

1. The public rSLDS model has the following construction contract:
   1.1. The package exports `rslds`.
   1.2. `rslds` returns a serializable `Template`.
   1.3. `rslds` returns parameter arrays separately from the template.
   1.4. The template represents each state-dependent regime transition with one `recurrence_link(z[t+1], z[t], x[t])` factor.
   1.5. The template does not multiply a `matrix_cat` factor into the recurrent transition.
   1.6. Each continuous transition uses `gaussian_mixture_linear(x[t+1], x[t], z[t+1])`.
   1.7. Each observation uses a `gaussian_linear` emission.
   1.8. Emission parameter blobs are shared across time steps.
   1.9. Dynamics parameter blobs are shared across time steps.
   1.10. Recurrence parameter blobs are shared across time steps.
   1.11. Each parameter array remains external to the serialized template.
   1.12. Each serialized parameter reference is a blob path.
2. The model-specific learner has the following update contract:
   2.1. The package exports `RSLDSLearning`.
   2.2. The package exports `vague_rslds_learning`.
   2.3. `vague_rslds_learning` constructs proper weak initial posteriors.
   2.4. One `RSLDSLearning.update` call adds the soft `z0` marginal to the initial-regime Dirichlet posterior.
   2.5. One `RSLDSLearning.update` call adds responsibility-weighted Gaussian statistics to one MNIW posterior per dynamics regime.
   2.6. One `RSLDSLearning.update` call adds Pólya–Gamma natural statistics to the recurrence-weight posterior.
   2.7. Repeating `update` on one sequence counts that sequence as new evidence again.
   2.8. `RSLDSLearning` leaves the shared emission parameters fixed.
3. The model-specific learner has the following installation contract:
   3.1. `RSLDSLearning.install` accepts a fresh `State` from the compiled rSLDS engine.
   3.2. `install` rejects a state that already contains inference messages.
   3.3. `install` writes the normalized initial-regime probability to the `z0_prior` member.
   3.4. `install` writes each regime's rolled MNIW matrix to every tied dynamics row.
   3.5. `install` writes each regime's rolled MNIW offset to every tied dynamics row.
   3.6. `install` writes each regime's rolled MNIW covariance to every tied dynamics row.
   3.7. `install` writes the rolled Pólya–Gamma weights to every tied recurrence row.
   3.8. `install` writes the rolled Pólya–Gamma offsets to every tied recurrence row.
   3.9. `install` leaves the shared emission parameters unchanged.
   3.10. `install` preserves the compiled parameter-tree keys.
   3.11. `install` preserves the compiled parameter-array shapes.
   3.12. `install` casts each learned value to its destination array's dtype.
   3.13. A trial loop reuses one compiled `Engine` while installing parameters into fresh states.
4. The compact fixed-parameter oracle check has the following evidence:
   4.1. A two-regime, one-dimensional float32 rSLDS returns `RunStatus.CONVERGED`.
   4.2. The converged state has finite Bethe free energy.
   4.3. The reference calculation enumerates both regime variables and uses 160-point Gauss–Hermite quadrature for the continuous state.
   4.4. Each regime posterior differs from the reference by no more than `0.1` in absolute probability.
   4.5. Each reported state mean differs from the reference by no more than `0.12`.
   4.6. Each reported state covariance differs from the reference by no more than `0.05`.
   4.7. The measured pre-threshold regime-probability gap remains recorded as `0.058`.
   4.8. The measured pre-threshold state-mean gap remains recorded as `0.067`.
   4.9. The measured pre-threshold covariance gap remains recorded as `0.019`.
   4.10. The test folds initial-regime statistics into proper posterior parameters.
   4.11. The test folds dynamics statistics into proper posterior parameters.
   4.12. The test folds recurrence statistics into proper posterior parameters.
   4.13. The test installs the learned initial-regime parameter into a fresh state.
   4.14. The test installs the learned dynamics parameters into a fresh state.
   4.15. The test installs the learned recurrence parameters into a fresh state.
   4.16. The existing D3 and D8 rule tests remain the authority for the documented free-energy bound directions.
5. The independent canonical-scale check has the following provenance and results:
   5.1. The sibling uv project `../experiments/cffg-nascar-validation` is pinned at commit `0543683`.
   5.2. The experiment pins CFFG commit `6c018125e9a42bbac234de03d54f7004b4d1aab0`.
   5.3. The experiment pins the stick-breaking NASCAR generator at upstream commit `383ddac2472cdf3248c6e871baede9ecda6f14cb`.
   5.4. The run uses `T=10,000`, `K=4`, latent dimension `d=2`, and observation dimension `dy=10`.
   5.5. The harness supplies the generating emission, dynamics, and recurrence parameters during inference.
   5.6. Inference reports convergence after 248 sweeps.
   5.7. The recorded latent-state root mean square error is `0.001264`.
   5.8. The recorded maximum-a-posteriori regime accuracy is `99.95%`.
   5.9. The recorded mean probability of the generating regime is `0.999106`.
   5.10. The recorded regime negative log likelihood is `0.001223`.
   5.11. The recorded compile and setup time is `723.14` seconds.
   5.12. The recorded inference time is `30.58` seconds.
   5.13. The recorded whole-process wall time is `844.38` seconds.
   5.14. The recorded peak resident memory is `2.03` GB.
   5.15. The experiment's `results.json` and `external-timing.json` record the command and artifact provenance.
   5.16. The result is described as fixed-parameter oracle inference rather than observed-only fitting or learned-model recovery.
   5.17. The result makes no claim that a learned rSLDS outperforms a matched learned SLDS.
6. Invalid and hostile inputs have the following outcomes:
   6.1. `rslds` rejects fewer than two time steps.
   6.2. `rslds` rejects fewer than two regimes.
   6.3. `rslds` rejects an inconsistent parameter shape before compilation.
   6.4. `RSLDSLearning` rejects a non-finite initial Dirichlet parameter.
   6.5. `RSLDSLearning` rejects a non-positive initial Dirichlet parameter.
   6.6. `RSLDSLearning.update` rejects a missing initial-regime belief before changing a posterior.
   6.7. `RSLDSLearning.update` rejects an improper initial-regime belief before changing a posterior.
   6.8. `RSLDSLearning.update` rejects a dynamics record marked `proper=False` before changing a posterior.
   6.9. `RSLDSLearning.update` rejects a recurrence record marked `proper=False` before changing a posterior.
   6.10. A categorical state with exact-zero support remains unsupported during one inference run.
   6.11. A proper positive Dirichlet rollover does not create exact-zero support between trials.
   6.12. The compact integration check preserves float32 parameter arrays.
   6.13. The `T=10,000` run reuses one compiled engine.
   6.14. Runtime state array shapes remain independent of the number of completed trials.
   6.15. Runtime parameter array shapes remain independent of the number of completed trials.
7. Documentation has the following content:
   7.1. `README.md` documents rSLDS construction.
   7.2. `README.md` documents the between-trial update.
   7.3. `README.md` documents learned-parameter installation.
   7.4. `docs/ARCHITECTURE.md` identifies the model-specific learner.
   7.5. `docs/ARCHITECTURE.md` identifies the learner's statistical authorities.
   7.6. `docs/LIMITATIONS.md` states that `RSLDSLearning` accumulates distinct-trial evidence rather than replacing one-sequence fitting coordinates.
   7.7. `docs/LIMITATIONS.md` states that shared-emission learning is outside `RSLDSLearning`.
   7.8. `docs/LIMITATIONS.md` describes the canonical run as fixed-parameter inference.

## Methodological Suggestions

These suggestions are optional. The implementer may choose another sound method that satisfies the outcomes.

- Use the **Learnable or checkpoint feature** change-path row in `docs/ARCHITECTURE.md`, restricted to a model-specific host-side learning boundary. Treat `src/cffg/models/rslds.py` as the owner of model assembly, posterior coordination, and parameter installation. Keep `src/cffg/conj.py`, `Engine.marginals`, `Engine.gaussian_node_stats`, and `Engine.recurrence_weight_stats` as the mathematical and statistic authorities.
- Reuse the existing `recurrence_link`, `gaussian_mixture_linear`, and `gaussian_linear` compiler groups. Validation, JSON, structure hashing, and blob paths are affected through the public builder. A new IR field, new factor registration, new message rule, new free-energy term, generic `Arena.member_of` expansion, and `Arena.learn_plans` are N/A because this ticket composes existing factors and uses a model-specific writer; `cff-6tek` owns generic non-discrete learning registration.
- Keep the parameter update outside JIT. The host/JIT pair crosses the host learner and JIT inference boundary in the compact integration test. Mixed scheduling is the primary rSLDS inference path; existing flooding-versus-mixed recurrence tests cover schedule parity. Single-instance/`vmap`, agent cadence, bundle persistence, and current or legacy learning checkpoints are N/A because this ticket introduces none of those contracts.
- Use `tests/test_recurrence_link.py::test_rslds_tier_matches_tiny_exact_oracle_and_rolls_all_parameters` as the compact integration check. Keep direct regime enumeration and Gauss–Hermite quadrature independent of engine beliefs. Retain the measured gaps and thresholds recorded in the test instead of treating the variational posterior as exact.
- Exercise sparse support through the existing recurrence and mixture rule tests. Exercise improper or vacuous inputs through the learner's `proper` checks and conjugate-posterior validation. Run the compact integration check in float32. Use the external `T=10,000` NASCAR run as the uncomfortable-scale evidence; do not add that experiment to pytest.
- Keep the external experiment in its sibling uv project. Pin the CFFG commit, upstream generator commit, seed, command, environment, metrics, timings, and result artifacts. Audit those records from the primary repository before using them as ticket evidence.
- Preserve the distinction between the two scientific workflows. The fixed-parameter oracle may use generating parameters to test inference against known latent states and regimes. Observed-only fitting may use observations and documented initialization, but not generating parameters, states, or regimes; `cff-l7y4` owns that workflow.

## Future work

`cff-l7y4`, introduced on the draft PR #56 branch, owns observed-only NASCAR fitting and comparison with a matched learned SLDS. It depends on this rSLDS assembly and on `cff-6tek`, because generic Gaussian/MNIW learnables are required for shared-emission fitting and replacement-coordinate updates. It must estimate parameters without access to the generating states, regimes, dynamics, recurrence weights, or emission parameters.

`cff-r6bl`, introduced on the draft PR #56 branch, owns the measured cold compile and setup cost. Its target is the `723.14`-second setup at `T=10,000`; it must preserve the fixed-parameter inference semantics and certification established here while reducing that cost.

No dependency from `cff-y4nx` to `cff-6tek` is needed. This ticket intentionally uses `RSLDSLearning` as a model-specific manual between-trial MNIW path, while `cff-l7y4` owns the consumer workflow that requires generic declared MNIW learning.

## Definitions

- **ARHMM (autoregressive hidden Markov model):** A switching time-series model often used to initialize regime assignments before fitting an rSLDS.
- **Bethe free energy (BFE):** The factor-graph objective reported by the engine and used as its convergence signal.
- **Blob:** A named external parameter array referenced by the serialized template.
- **Conjugate posterior:** A posterior distribution in the same family as its prior, updated through sufficient statistics.
- **D3 and D8:** Sections of `docs/DERIVATIONS.md` that define the hybrid-mixture and recurrence-link approximations and free-energy terms.
- **Dirichlet posterior:** A probability-distribution posterior over a categorical probability vector.
- **Distinct trial:** A new sequence or episode whose statistics add new evidence to a posterior.
- **Fixed-parameter oracle inference:** Inference that receives the generating model parameters and compares inferred hidden variables with generating hidden variables.
- **Gauss–Hermite quadrature:** A weighted numerical sum used to approximate an integral under a Gaussian distribution.
- **JIT (just-in-time compilation):** Compilation of numerical operations before repeated execution.
- **MAP (maximum a posteriori):** Selection of the regime with the largest posterior probability.
- **MNIW (Matrix-Normal-Inverse-Wishart):** A conjugate posterior for a linear dynamics matrix, offset, and covariance.
- **NASCAR:** The four-regime synthetic oval-motion example distributed with the recurrent-SLDS research code.
- **NLL (negative log likelihood):** The negative logarithm of the probability assigned to the generating regime.
- **Observed-only fitting:** Parameter estimation that receives observations but not generating parameters, states, or regimes.
- **PCA (principal component analysis):** A linear dimensionality-reduction method used here only as a possible initializer for latent states.
- **Pólya–Gamma posterior:** The Gaussian-form posterior over stick-breaking logistic weights obtained through Pólya–Gamma augmentation.
- **Responsibility:** A soft posterior probability that one regime generated a transition.
- **Rollover:** Conversion of posterior hyperparameters into point parameters used during a later trial.
- **rSLDS (recurrent switching linear dynamical system):** A state-space model whose discrete regime transition depends on its continuous latent state.
- **SLDS (switching linear dynamical system):** A state-space model whose discrete regime transition does not depend on its continuous latent state.
- **Soft statistic:** A sufficient statistic weighted by posterior probabilities rather than a hard regime assignment.
- **State:** The engine's immutable runtime arrays for messages, observations, and parameters.
- **Variational EM (variational expectation-maximization):** Alternating approximate hidden-variable inference and parameter updates. `RSLDSLearning` adds evidence across distinct trials; it is not the replacement-coordinate loop for fitting one sequence.

## Notes

**2026-08-01T04:23:00Z**

Research — Question: which canonical experiment demonstrates the rSLDS advantage, and what should it prove? Answer: use Linderman et al. (2017) synthetic NASCAR (four regimes, two-dimensional latent oval, ten-dimensional linear observations) and compare with a matched ordinary SLDS; evaluate state-dependent partition recovery, held-out prediction, and periodic generative rollouts rather than exact paper pixels. Source: https://proceedings.mlr.press/v54/linderman17a.html and the authors’ reproduction entry point https://github.com/slinderman/recurrent-slds (examples/nascar.py). Confidence: high for benchmark selection and qualitative claim; pilot measurements are still required before quantitative thresholds are fixed.

**2026-08-01T05:06:38Z**

Independent validation and audit, 2026-07-31.

Question: can a fresh consumer reproduce canonical NASCAR with the public CFFG surface at commit 6c018125e9a42bbac234de03d54f7004b4d1aab0? Answer: fixed-parameter inference yes; canonical observed-only learned fitting no. Source: sibling experiment /Users/ravd/projects/experiments/cffg-nascar-validation at commit 0543683, pinned to the current stick-breaking upstream generator 383ddac2472cdf3248c6e871baede9ecda6f14cb. Confidence: high; the primary agent audited the installed dependency path, generator, metrics, plot, schema provenance, clean tree, and independently reran all 3 lightweight experiment tests.

The T=10,000 oracle run converged in 248 sweeps. State RMSE 0.001264; MAP regime accuracy 99.95%; mean true-regime probability 0.999106; regime NLL 0.001223. CFFG compile/setup was 723.14 s, inference 30.58 s, parent-process wall 844.38 s, and peak RSS 2.03 GB. Artifact provenance and the exact command are recorded in results.json and external-timing.json.

This proves canonical-scale inference for the recurrence link plus switching dynamics, but does not satisfy the learned NASCAR claim: the emission parameters were fixed to their generating values, and there is no matched learned SLDS or generative-rollout comparison. The consumer used its permitted not-ready exit because RSLDSLearning does not learn shared emissions, exposes accumulation rather than replacement coordinates for one long sequence, and has no PCA/ARHMM-equivalent initializer. cff-l7y4 owns that learned-model continuation after cff-6tek. cff-r6bl owns the measured cold-compile blocker. This ticket stays in progress and this PR must describe the result as oracle inference, never full NASCAR reproduction.

Legend: prerequisite --> dependent

    [x] cff-k9ws recurrence rule
    [x] cff-4apw PG learning
              |
              +--> [~] cff-y4nx rSLDS assembly + oracle validation
                          |
                          +--> [ ] cff-r6bl cold compile reduction (blocked)
                          |
                          +--> [ ] cff-l7y4 learned NASCAR fit (blocked)
                                      ^
                                      |
                    [ ] cff-6tek Gaussian learnables (ready)

cff-ahps is the parent roadmap epic. The ticket dependency records above are the source of truth.

**2026-08-08T04:30:05Z**

Dependency review for the rewrite. Arrow direction is prerequisite --> dependent.

```text
[x] cff-k9ws -----------------------------+
       |                                  |
       +--> [x] cff-4apw -----------------+
                 ^                        |
                 |                        +--> [ ] cff-y4nx (blocked)
[x] cff-lquu ----+                        |
                                          |
[~] cff-twzc -----------------------------+
```

The current `tk` records do not make `cff-y4nx` depend on `cff-6tek`, and no edge is missing. Draft PR #56 implements `RSLDSLearning` as this ticket's intentional model-specific MNIW path. The PR-head ticket `cff-l7y4`, not `cff-y4nx`, depends on `cff-6tek` for observed-only fitting through generic declared Gaussian learning.

Retained wiggle candidates after the bounded current scan (candidate -> intended meaning): canonical -> literature_standard for the pinned Linderman NASCAR variant; fixed -> constant_parameter for oracle-supplied or unchanged parameters and pinned_artifact for named commits, dimensions, and thresholds; generic -> protocol_general for declared-learning registration; integrate -> integration_test for the compact cross-layer check; n_a -> structurally_irrelevant or delegated_elsewhere with the reason stated in the change-path handoff; preserve -> byte_or_value_identity, semantic_invariant, or compatibility for the named dtype, shape, key, inference, or certification property; public -> exported_symbol, documented_contract, or consumer_workflow as selected by the named surface; register -> runtime_family_coverage delegated to `cff-6tek`; reusable -> code_reuse or artifact_reuse for existing factors and compiled engines; same -> shared_category in the conjugate-posterior definition; expose -> internal_access in the preserved PR note; implement -> component_integration_exists for the PR's model-specific learner; support -> probability_support; complete -> execution_finished for completed trials; ready -> dependency_ready in the dated dependency map or the permitted not-ready experiment exit; remaining -> preserved_state; approximate -> mathematical_approximation; sufficient -> statistical_sufficiency; certify -> formal_runtime_certificate; check -> automated_check, manual_inspection, or experiment audit as named; converge -> runtime_status_criterion; equivalent -> structural_compatibility for an initializer comparable to PCA/ARHMM; estimate -> parameter_estimation from observations; evidence -> distinct_evidence in learning semantics or empirical_support/provenance_artifact in validation; exact -> exhaustive_computation, literal_zero_or_support, or exact_target_checked_to_tolerance as bounded by the sentence; match -> equivalent_configuration for the SLDS baseline or tolerance_parity for numeric comparisons; oracle -> generating_parameter_oracle; parity -> behavioral_parity between schedules; reproduce -> scientific_reproduction of the named benchmark; sound -> reasonable_judgment for an optional method; test -> test_artifact, automated_execution, or scientific_experiment as named; validate -> component_validation or scientific_validation; accumulate -> distinct_evidence across sequences and repeated_coordinate when the preserved note warns about reusing one sequence; fit -> parameter_estimation or numerical_agreement as bounded by the sentence; infer -> state_or_regime_posterior with parameters held fixed; learn -> parameter_update_primitive, integrated_learning_workflow, or observed parameter estimation as named; learnable -> registered_parameter_family for generic Gaussian declarations; recover -> state_or_regime_recovery for oracle metrics or parameter_estimation in delegated fitting; rollover -> point_parameter_install; trial -> independent_dataset; scale -> horizon/domain_size or measured memory and latency load; small -> domain_size for the compact oracle fixture; can -> demonstrated_capability in the answered consumer question; intended -> semantic_purpose recorded by this review; may -> permission for a method or for oracle-only access to generating values; should -> design_intent in the preserved research question; explicit_owner and named_ticket_handoff -> current-ticket ownership or the concrete `cff-l7y4`, `cff-r6bl`, `cff-6tek`, and `cff-ahps` owners named by the sentence; all -> enumerated_subset_exhaustive; every -> named_class_distributive; only -> hard_exclusion, current_limitation, or ticket_scope_boundary as bounded by the sentence; current -> runtime_active_value or repository_at_reference when pinned by a commit; existing -> predates_change for reused factors and tests; later -> ordering_constraint.

Intelligible-ticket review: complete.
