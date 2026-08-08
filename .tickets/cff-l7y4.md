---
id: cff-l7y4
status: open
deps: [cff-y4nx, cff-6tek]
links: [cff-y4nx]
created: 2026-08-01T05:05:47Z
type: feature
priority: 2
assignee: coopwilliams
parent: cff-ahps
tags: [rslds, validation, learning]
---
# Fit and compare the canonical NASCAR rSLDS from observations

## Background in Plain English

Ticket `cff-y4nx` assembled the rSLDS model and validated inference at
`T=10,000` by supplying the generating parameters. That run did not show that
CFFG can estimate an rSLDS from observations. The model-specific
`RSLDSLearning` helper keeps the shared emission fixed, counts a sequence again
when `update` is repeated, and has no initializer that separates latent
coordinates and regime labels without access to the generating variables.

This ticket owns the missing observed-only fitting workflow and the external
comparison between a learned rSLDS and a matched learned SLDS. Ticket
`cff-6tek` owns the generic Gaussian and MNIW learnable registration that this
workflow requires. Ticket `cff-r6bl` owns cold setup optimization; this ticket
may proceed with the measured setup cost because each fitted model compiles
once and reuses that executable throughout fitting.

## Outcomes / Acceptance Criteria

1. The public fitting API accepts observations, a documented initializer,
   fixed prior hyperparameters, an outer-iteration budget, and explicit random
   keys without accepting generating states, regimes, or model parameters.
2. The fitting API estimates the initial-regime probabilities, per-regime
   dynamics matrix, dynamics offset, dynamics covariance, recurrence weights,
   recurrence offsets, shared emission matrix, shared emission offset, and
   shared emission covariance.
3. Shared `gaussian_linear` emission parameters update from soft factor-local
   MNIW statistics through the generic learning surface supplied by
   `cff-6tek`.
4. Each outer iteration recomputes every one-sequence variational coordinate
   from the fixed priors and the current sequence statistics rather than adding
   the same sequence to the previous iteration as new evidence.
5. Each outer iteration runs hidden-state and regime inference with the public
   `Engine` and installs the next point parameters into a fresh state without
   rebuilding the model or recompiling the executable.
6. The outer loop reports `converged`, `budget_exhausted`, or `diverged` and
   preserves the inner `Engine` status that caused a non-converged result.
7. A non-finite Bethe free energy, parameter-prior term, outer objective, or
   coordinate change produces `diverged` and prevents a parameter write.
8. The outer stopping rule names the finite objective and dtype-scaled
   tolerance that make a convergence result falsifiable.
9. The documented initializer uses PCA plus an ARHMM-style method, or another
   method with recorded justification, to choose latent coordinates and break
   regime-label symmetry without generating states or regimes.
10. The initializer is reproducible from observations, configuration, and an
    explicit random key.
11. A compact synthetic case estimates all parameter families from
    observations and compares them after affine and label alignment with the
    generating values withheld from fitting.
12. Repeating one outer coordinate update with unchanged sequence statistics
    does not increase a posterior evidence count as though another sequence had
    been observed.
13. Exact-zero regime support remains zero during inference and is not revived
    by a numeric probability floor.
14. Improper Gaussian, recurrence, or emission statistics terminate fitting
    before any new parameters are installed.
15. The compact fitting case runs in float32 with dtype-derived guards and
    finite parameters, objective values, and posterior statistics.
16. The compact suite includes a deliberately uncomfortable domain or horizon
    at least ten times larger than its smallest recovery fixture.
17. The external experiment uses the pinned stick-breaking NASCAR generator
    with `T=10,000`, `K=4`, latent dimension `2`, and observation dimension
    `10`.
18. The learned rSLDS and learned SLDS receive identical training observations,
    held-out observations, initialization information, parameter budgets,
    outer budgets, random-seed sets, and convergence rules except where their
    model structures require a documented difference.
19. Neither fitting arm receives generating states, regimes, transition
    parameters, dynamics parameters, or emission parameters before fitting is
    finished.
20. The final experiment reports affine- and label-aligned latent and regime
    recovery, held-out predictive log likelihood, and long generative-rollout
    periodicity for both model families.
21. Exact numeric thresholds and the required rSLDS-versus-SLDS margin for each
    claimed advantage are fixed from a named pilot artifact before the final
    seed set is run.
22. The final conclusion is limited to the metrics and seed set whose
    preregistered thresholds succeed; a missed threshold cannot be replaced by
    an unregistered qualitative claim.
23. One compiled engine per fitted model is reused across every outer
    iteration, and runtime state and parameter shapes do not grow with the
    number of iterations or horizon observations.
24. The result records the upstream generator commit, CFFG commit, experiment
    commit, data split, random seeds, commands, environment, statuses, timing
    phases, peak resident memory, and artifact hashes.
25. A not-ready result names a concrete capability blocker and creates or links
    the ticket that owns it instead of claiming learned-model validation.
26. `README.md` documents the fitting API and the distinction between
    distinct-trial accumulation and one-sequence coordinate replacement.
27. `docs/ARCHITECTURE.md` records fitting cadence, statistical authorities,
    initialization, parameter installation, and status propagation.
28. `docs/LIMITATIONS.md` records the recurrence mean-field and
    empirical-Bayes point-parameter approximations, affine-identifiability,
    label-identifiability, and demonstrated experimental boundaries.

## Methodological Suggestions

These suggestions are optional. The implementer may choose another sound
method that satisfies the outcomes.

- Use the **Learnable or checkpoint feature** change-path row. Treat
  `cffg.learn` after `cff-6tek` as the generic declaration and cadence owner,
  `cffg.conj` as the posterior-algebra owner, and `cffg.models.rslds` as the
  model-specific assembly and parameter-mapping owner.
- Inspect validation, serialized blob references, compiler `member_of` maps,
  `Arena.learn_plans`, engine statistic readouts, fitting cadence, parameter
  writes, and documentation. Add no checkpoint or bundle format unless the
  acceptance outcomes require resumable fitting; current-versus-legacy
  checkpoint handling is otherwise outside this ticket.
- Keep the outer fitter on the host around JIT inference. Exercise that
  host/JIT boundary and mixed-schedule inference in compact tests. Batched
  `vmap` fitting is outside this ticket because no public batched fitting
  contract is required.
- Keep the `T=10,000` fit, matched SLDS baseline, and long rollouts in the
  sibling uv experiment. Keep compact coordinate, hostile-input, and float32
  cases in the repository test suite.
- Use generating states and parameters only after fitting, for aligned scoring
  and diagnostics. Store the pre-final pilot thresholds in a provenance-bearing
  artifact that the final-run command cannot overwrite.

## Future work

`cff-r6bl` owns reducing the measured cold setup cost. Its optimization is not
an acceptance dependency for this ticket because each fitting arm can compile
once and reuse its executable. Lower setup cost shortens future model
development and repeated experiments.

`cff-ahps` is the downstream roadmap epic that receives the demonstrated
observed-only rSLDS fitting capability. No other future work is identified.

## Definitions

- **Affine alignment:** A fitted linear transformation used only during
  evaluation to compare latent coordinate systems that are mathematically
  equivalent.
- **ARHMM:** Autoregressive hidden Markov model, used here as an initialization
  method for regime-specific linear dynamics.
- **Bethe free energy:** The factor-graph objective reported by the inference
  engine.
- **Distinct-trial accumulation:** Adding statistics from a new independent
  sequence to posterior evidence retained across trials.
- **Held-out predictive log likelihood:** The log probability assigned to
  observations excluded from parameter fitting.
- **Label alignment:** A permutation applied only during evaluation to compare
  equivalent regime labels.
- **MNIW:** Matrix-Normal-Inverse-Wishart, a conjugate posterior for a linear
  matrix, offset, and covariance.
- **Observed-only fitting:** Parameter estimation that receives observations
  but not generating parameters, states, or regimes.
- **Outer coordinate:** One parameter-posterior update in a variational fitting
  loop around hidden-variable inference.
- **Parameter budget:** The model dimensions and free parameter counts used to
  keep two comparison arms matched.
- **PCA:** Principal component analysis, used here as a possible observation-
  based latent-state initializer.
- **Preregistered threshold:** A numeric success boundary recorded before the
  final evaluation data or seed set is run.
- **rSLDS:** Recurrent switching linear dynamical system, whose regime
  transition depends on the continuous latent state.
- **SLDS:** Switching linear dynamical system, whose regime transition does not
  depend on the continuous latent state.

## Notes

**2026-08-08T16:06:55Z**

Rewrite handoff, 2026-08-08. cff-l7y4 owns observed-only one-sequence fitting, shared-emission coordinates, truth-free initialization, matched learned-model controls, and the final external comparison. cff-6tek owns generic Gaussian/MNIW registration; cff-y4nx owns model assembly and fixed-parameter inference; cff-r6bl owns cold setup optimization and is deliberately not a prerequisite.

Legend: prerequisite --> dependent

    [x] cff-x0sv --> [ ] cff-6tek (ready) ----------------+
                                                               |
    [x] cff-k9ws --> [x] cff-4apw --+                         |
    [~] cff-twzc --------------------+--> [~] cff-y4nx --------+--> [ ] cff-l7y4
                                                                           |
                                                                           +--> [ ] cff-ahps

The tk dependency records are authoritative. Validation used the bounded historical slice, two bounded current scans, tk dep tree, tk ready, tk blocked, tk dep cycle, and git diff --check. Retained current-scan candidates and meanings: canonical=literature_standard; fixed=constant_parameter or pinned_artifact; generic=protocol_general or registered_coverage; preserve=semantic_invariant; public=documented_contract or consumer_workflow; register=runtime_family_coverage; reusable=artifact_reuse; same=object_identity; support=probability_support; ready=operational_ready in the not-ready result; remaining=preserved_state; approximate=named_mathematical_method; complete=ticket-review completion marker; check=automated_check; converge=runtime_status_criterion; equivalent=equivalent latent or label representation; estimate=parameter_estimation; evidence=new independent data or posterior evidence as named; exact=literal_zero_or_support or numeric threshold; match=equivalent_configuration; reproduce=seeded initialization; sound=reasonable_judgment; test=test_artifact or automated_execution; validate=component_validation or scientific_validation; accumulate=distinct_evidence; fit=fixed_dataset_fit or parameter estimation; infer=state_or_regime_posterior; learn=integrated_learning_workflow or recovery_demonstrated; learnable=registered_parameter_family; recover=parameter_estimation against withheld values; trial=independent_dataset; scale=domain_size; small=the named smallest fixture; can=demonstrated capability or one permitted compile; may=permission; explicit_owner and named_ticket_handoff=the ticket or component named in each sentence; all=the enumerated parameter-family set; any=negative_scope; at_least=inclusive_lower_bound; every=temporal_frequency or named-class distribution; only=hard phase separation or ticket boundary; current=runtime_active_value or the named current-versus-legacy pair. Capability ownership was checked separately. Intelligible-ticket review: complete.

**2026-08-08T16:16:48Z**

Intelligible-ticket review: complete.
