---
id: cff-y4nx
status: in_progress
deps: [cff-k9ws, cff-4apw, cff-twzc]
links: [cff-l7y4]
created: 2026-07-15T01:46:10Z
type: feature
priority: 2
assignee: coopwilliams
parent: cff-ahps
tags: [hybrid, validation, rslds]
---
# rSLDS tier: model-specific continual learning and fixed-parameter validation

## Background in Plain English

A recurrent switching linear dynamical system tracks a continuous state and a
discrete regime. Its regime transition depends on both the previous regime and
the continuous state. Closed tickets `cff-k9ws`, `cff-lquu`, and `cff-4apw`
supply the recurrence rule, soft hybrid statistics, and recurrence-weight
updates that this ticket assembles with switching Gaussian dynamics and shared
Gaussian emissions.

This ticket also owns `RSLDSLearning`, a model-specific host-side learner. It
adds evidence from each distinct sequence to posteriors for the initial regime,
per-regime dynamics, and recurrence weights, then installs point parameters for
a later trial without rebuilding the graph. It does not fit one observed
sequence by repeatedly replacing variational coordinates, and it keeps the
shared emission fixed. Ticket `cff-l7y4` owns that observed-only fitting
workflow after `cff-6tek` supplies generic Gaussian learnables.

The external NASCAR run supplies the generating model parameters. It therefore
validates fixed-parameter inference at the canonical scale, not parameter
learning or superiority over a learned ordinary switching model. Ticket
`cff-r6bl` owns the cold setup cost exposed by that run.

## Outcomes / Acceptance Criteria

1. The package exports `rslds`, `RSLDSLearning`, and `vague_rslds_learning`.
2. `rslds` returns a serializable `Template` and a separate map of parameter
   arrays.
3. Each state-dependent regime transition is represented by one
   `recurrence_link(z[t+1], z[t], x[t])` factor and no additional `matrix_cat`
   transition factor.
4. Each continuous transition uses
   `gaussian_mixture_linear(x[t+1], x[t], z[t+1])`.
5. Each observation uses one `gaussian_linear` emission whose parameter blobs
   are shared across time.
6. Dynamics and recurrence parameter blobs are shared across time, remain
   external to the template, and are referenced by blob paths.
7. `rslds` rejects fewer than two time steps, fewer than two regimes, and
   inconsistent parameter shapes before compilation.
8. `vague_rslds_learning` creates proper weak Dirichlet, MNIW, and Pólya–Gamma
   posteriors for the model dimensions supplied by the caller.
9. One `RSLDSLearning.update` call adds the soft `z0` marginal to the
   initial-regime Dirichlet posterior.
10. One `RSLDSLearning.update` call adds responsibility-weighted factor-local
    Gaussian statistics to one MNIW posterior per dynamics regime.
11. One `RSLDSLearning.update` call adds factor-local Pólya–Gamma natural
    statistics to the recurrence-weight posterior.
12. Repeating `update` on one sequence counts that sequence as new evidence
    again; the API does not present this operation as one-sequence coordinate
    replacement.
13. `RSLDSLearning` leaves the shared emission parameters unchanged.
14. `RSLDSLearning.install` accepts a fresh state from the compiled rSLDS
    engine and rejects a state that already contains inference messages.
15. `install` writes the normalized initial-regime probability, every regime's
    rolled dynamics matrix, offset, and covariance, and the rolled recurrence
    weights and offsets to every tied runtime row.
16. `install` preserves parameter-tree keys and array shapes and casts each
    learned value to its destination dtype.
17. A trial loop reuses one compiled `Engine` while installing parameters into
    fresh states.
18. A two-regime, one-dimensional float32 integration case returns
    `RunStatus.CONVERGED` with finite Bethe free energy.
19. The compact reference independently enumerates both regime variables and
    uses 160-point Gauss–Hermite quadrature for the continuous state.
20. In the compact case, each regime probability differs from the reference by
    at most `0.1`, each state mean differs by at most `0.12`, and each state
    covariance differs by at most `0.05`.
21. The compact case records the pre-threshold gaps as `0.058` for regime
    probability, `0.067` for state mean, and `0.019` for covariance.
22. The compact case updates and installs the initial-regime, dynamics, and
    recurrence parameters while preserving float32 parameter arrays.
23. Existing D3 and D8 rule tests remain the authority for the documented
    free-energy bound directions.
24. The sibling uv project `../experiments/cffg-nascar-validation` records the
    canonical-scale check at experiment commit `0543683`, CFFG commit
    `6c018125e9a42bbac234de03d54f7004b4d1aab0`, and upstream generator commit
    `383ddac2472cdf3248c6e871baede9ecda6f14cb`.
25. The external check uses `T=10,000`, `K=4`, latent dimension `2`, and
    observation dimension `10`, with generating emission, dynamics, and
    recurrence parameters supplied to inference.
26. The external check records convergence after 248 sweeps, latent-state RMSE
    `0.001264`, MAP regime accuracy `99.95%`, mean generating-regime probability
    `0.999106`, and regime negative log likelihood `0.001223`.
27. The external check records `723.14` seconds of compile and setup, `30.58`
    seconds of inference, `844.38` seconds of whole-process wall time, and
    `2.03` GB peak resident memory.
28. The external result is described as fixed-parameter oracle inference and
    makes no claim about observed-only fitting or learned rSLDS performance
    against a learned SLDS.
29. `RSLDSLearning` rejects non-finite or non-positive Dirichlet parameters,
    missing or improper initial-regime beliefs, and factor-local dynamics or
    recurrence records marked `proper=False` before changing a posterior.
30. Exact-zero categorical support remains zero within one inference run; a
    positive Dirichlet rollover does not create exact-zero support between
    trials.
31. The `T=10,000` check reuses one compiled engine, and its runtime state and
    parameter array shapes do not grow with the number of completed trials.
32. `README.md` documents construction, between-trial updating, and point
    parameter installation.
33. `docs/ARCHITECTURE.md` identifies the model-specific learner and its
    statistical authorities.
34. `docs/LIMITATIONS.md` states that the learner accumulates distinct-trial
    evidence, keeps shared emissions fixed, and does not establish a learned
    NASCAR result.

## Methodological Suggestions

These suggestions are optional. The implementer may choose another sound
method that satisfies the outcomes.

- Use the **Learnable or checkpoint feature** row in
  `docs/ARCHITECTURE.md`, restricted to the model-specific host-side learning
  boundary. Treat `src/cffg/models/rslds.py` as the owner of model assembly,
  posterior coordination, and parameter installation. Keep `cffg.conj.MNIW`,
  `cffg.conj.PGWeights`, `Engine.marginals`,
  `Engine.gaussian_node_stats`, and `Engine.recurrence_weight_stats` as the
  statistical authorities.
- Reuse the registered `recurrence_link`, `gaussian_mixture_linear`, and
  `gaussian_linear` factors. This ticket adds no IR field, factor family,
  message rule, free-energy term, generic `Arena.member_of` registration, or
  `Arena.learn_plans` entry. Ticket `cff-6tek` owns generic non-discrete
  learning registration.
- Keep learning outside JIT. Exercise the host-learner/JIT-inference boundary
  in the compact integration case. Existing recurrence tests own
  flooding-versus-mixed schedule parity. Batched `vmap`, agent cadence,
  bundles, and learning checkpoints are not part of this ticket.
- Use `tests/test_recurrence_link.py` for the compact float32 integration and
  independent oracle. Keep the `T=10,000` run in the sibling experiment so it
  does not enter the default test suite.
- Preserve the external experiment command, environment, commits, seed,
  timings, memory, metrics, and artifact hashes before citing its result.

## Future work

`cff-l7y4` owns fitting the NASCAR observations without access to generating
parameters, states, or regimes. It also owns shared-emission learning,
one-sequence replacement updates, initialization, and comparison with a matched
learned SLDS. It depends on this ticket and on `cff-6tek`.

`cff-r6bl` owns reducing the measured `723.14`-second cold compile and setup
cost while preserving the inference semantics and certification established
here.

`cff-ahps` remains the downstream roadmap epic. This ticket does not depend on
`cff-6tek` because its manual `RSLDSLearning` path is deliberately
model-specific.

## Definitions

- **Bethe free energy:** The factor-graph objective reported by the engine and
  used as its convergence signal.
- **Blob:** A parameter array stored outside a serialized template and
  referenced by path.
- **Distinct trial:** A new sequence whose statistics add new evidence to a
  posterior.
- **Fixed-parameter oracle inference:** Hidden-state and regime inference that
  receives the generating model parameters for validation.
- **Gauss–Hermite quadrature:** A weighted numerical sum used to approximate an
  integral under a Gaussian distribution.
- **JIT:** Just-in-time compilation of numerical operations before repeated
  execution.
- **MAP:** Maximum a posteriori; the regime with the largest posterior
  probability.
- **MNIW:** Matrix-Normal-Inverse-Wishart, a conjugate posterior for a linear
  matrix, offset, and covariance.
- **NASCAR:** The four-regime synthetic oval-motion example distributed with
  the recurrent-SLDS research code.
- **Observed-only fitting:** Parameter estimation that receives observations
  but not generating parameters, states, or regimes.
- **Pólya–Gamma posterior:** The Gaussian-form posterior over stick-breaking
  logistic weights obtained through Pólya–Gamma augmentation.
- **Responsibility:** A soft posterior probability that one regime generated a
  transition.
- **Rollover:** Conversion of posterior hyperparameters into point parameters
  for a later trial.
- **rSLDS:** Recurrent switching linear dynamical system; its discrete regime
  transition depends on its continuous latent state.
- **SLDS:** Switching linear dynamical system; its discrete regime transition
  does not depend on its continuous latent state.

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

**2026-08-08T16:00:51Z**

Rewrite handoff, 2026-08-08. The ticket now makes the implemented boundary reviewable: cff-y4nx owns rSLDS assembly, distinct-trial posterior accumulation, point-parameter installation, compact float32/oracle validation, and the pinned fixed-parameter NASCAR result. cff-l7y4 owns observed-only fitting and the learned-model comparison after cff-6tek; cff-r6bl owns cold setup cost. Historical research and experiment evidence remain unchanged.

Legend: prerequisite --> dependent

    [x] cff-k9ws --> [x] cff-4apw --+
    [~] cff-twzc --------------------+--> [~] cff-y4nx (blocked)
                                              |
                                              +--> [ ] cff-r6bl
                                              |
    [ ] cff-6tek (ready) ---------------------+--> [ ] cff-l7y4
                                                       |
                                                       +--> [ ] cff-ahps

The tk dependency records are authoritative. Validation used the bounded historical slice, two bounded current scans, tk ready, tk blocked, tk dep cycle, and git diff --check. Retained current-scan candidates and meanings: canonical=literature_standard; fixed=constant_parameter or pinned_artifact; generic=protocol_general; integrate=integration_test; preserve=byte_or_value_identity or semantic_invariant; public=public_api in the historical note; register=runtime_family_coverage; reusable=code_reuse or artifact_reuse; expose=reveal_defect or historical internal_access; support=probability_support; complete=execution_finished; ready=dependency_ready; remaining=preserved_state or deliberately_excluded_work; approximate=numerical_approximation; close=work_item_completion; certify=runtime_certificate; check=automated or scientific check; converge=runtime_status_criterion; equivalent=structural_compatibility; estimate=parameter_estimation; evidence=distinct_evidence, empirical_support, or provenance_artifact; exact=literal value, support, or command identity; match=equivalent_configuration or numeric agreement; oracle=generating_parameter_oracle; parity=behavioral_parity; reproduce=scientific_reproduction; sound=reasonable_judgment; test=test_artifact or automated_execution; validate=component or scientific validation; accumulate=new_evidence_accumulation; fit=parameter_estimation; infer=state_or_regime_posterior; learn=new_evidence_accumulation, parameter_update_primitive, fixed-parameter inference, or delegated observed-only fitting as selected by the sentence; learnable=registered_parameter_family; recover=state_or_parameter recovery in the historical note; rollover=point_parameter_install; trial=independent_dataset; performance=named learned-model comparison; scale=domain_size; small=compact historical fixture; can=demonstrated consumer capability; may=implementation permission; should=historical design intent; explicit_owner and named_ticket_handoff=the ticket, component, or concrete future ticket named in the sentence; all=the historical locally named set; every=named_class_distributive; only=current_limitation or ticket_scope_boundary; current=repository_at_reference in the historical note; existing=predates_change; later=ordering_constraint. Capability ownership was checked separately. Intelligible-ticket review: complete.

**2026-08-08T16:01:10Z**

Post-note scan addition: implement=component_integration_exists for the present rSLDS boundary; the occurrence inside the may mapping denotes permission to choose a method. Both readings are selected by their sentences.

**2026-08-08T16:16:48Z**

Intelligible-ticket review: complete.
