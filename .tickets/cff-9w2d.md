---
id: cff-9w2d
status: in_progress
deps: []
links: []
created: 2026-08-24T07:21:35Z
type: bug
priority: 1
assignee: coopwilliams
parent: cff-45ia
tags: [rslds, experiment, d10, validation, numerics]
---
# Use authoritative D10 terminal beliefs before classic scoring

## Background in Plain English — WHY

The first cff-45ia funnel fit passed the runner's D10 identity, convergence, inner-status, and terminal-coordinate checks. Metric evaluation then failed because affine alignment received NaN latent means and NumPy's least-squares solver could not converge.

The fit did not supply those NaNs. The experiment helper read the legacy D3/D9 Gaussian and categorical marginal arenas after a D10 solve. D10 owns separate structured-hybrid singleton, pair, Gaussian, and lag beliefs and does not copy them into the old arenas. The old Gaussian arena is therefore vacuous and correctly reports improper NaN moments. The experiment must use the public D10 belief authority when D10 is active.

The runner also wrote its fit checkpoint only after alignment, metrics, and rollouts. The valid fit status, work, parameters, and authoritative beliefs were lost when alignment raised. This ticket fixes both experiment boundaries before a newly registered funnel may repeat the same development coordinate.

## Outcomes / Acceptance Criteria — WHAT

1. One shared experiment helper returns terminal beliefs from exactly one proper, converged D10 component through `Engine.structured_hybrid_beliefs` when a structured-hybrid plan exists. It returns Gaussian means and covariances, categorical singleton probabilities, consecutive-regime pairs, lag moments, support, and semantic identities in their documented source/destination and current/previous orientations.
2. The helper uses existing generic Gaussian and categorical marginal readouts only when no D10 component exists. Existing D9 and matched-SLDS consumers retain their current values and shapes.
3. A D10 belief result is rejected before alignment when its status is not converged, `proper` is false, shapes differ from the declared trajectory, values are non-finite, pair and singleton marginals disagree beyond dtype-scaled tolerance, or its component identity differs from the compiled plan.
4. Immediately after every fit returns and before truth-bound alignment, recovery metrics, prediction, or rollouts, the paired classic runner writes an append-only authenticated fit-complete checkpoint. It binds the full arm coordinate, observations, initialization, stochastic streams, experiment/CFFG/runtime identities, D10 topology and four statistic owners, fit status and phase work, point parameters, authoritative terminal beliefs, fit wall time, process high-water RSS, and finite shape/scale summaries.
5. A failure after the fit-complete checkpoint preserves that checkpoint and records a named post-fit phase failure. A same-code resume authenticates and reuses the completed fit without calling `fit_rslds` again. A failure before fit completion remains a fit-stage failure with no reusable scientific checkpoint.
6. Alignment validates finite inputs and declared shapes before linear algebra. It records centered design column scales and singular values, uses an equivalent column-scaled least-squares solve, and distinguishes `terminal_posterior_invalid`, `alignment_rank_deficient`, and `alignment_numerical_failure` without exposing an untyped LAPACK exception as a fit failure.
7. Compact D10 evidence proves that the legacy Gaussian readout is improper while the structured-hybrid readout is finite and proper, and that the metric path uses only the structured result. A compact non-D10 fixture proves the legacy path is unchanged.
8. Hostile tests cover D10 budget exhaustion and divergence, improper support, NaN and infinity, wrong shapes, inconsistent pair/singleton values, finite rank loss, extreme unequal column scales, and an unrelated huge row that cannot widen another local tolerance.
9. The authenticated cff-45ia attempt-0001 failure remains byte-for-byte immutable. It cannot resume, promote, or masquerade as a fit-complete checkpoint. The repair requires a new immutable registration and output that links the superseded infrastructure artifact before repeating the same declared coordinate and streams.
10. The fix is applied to every experiment consumer that reads CFFG rSLDS terminal latent/regime beliefs, including classic and future confirmatory metrics. Duplicated callers are audited and either migrated to the shared helper or marked structurally inapplicable with a test.
11. No CFFG engine behavior, inference objective, fit algorithm, prior strength, metric threshold, adapter, calibration seed, final seed, or scientific conclusion changes in this ticket.

## Methodological Suggestions — HOW

These suggestions are optional. The implementer may choose another sound method.

- Factor the D10 extraction already used by `structured_hybrid_checkpoint.py` into a shared experiment module instead of copying its validation.
- Keep fit-complete evidence separate from derived metrics so replay can resume at the first unfinished phase.
- Compute affine alignment from centered, locally scaled columns, then transform coefficients back to the original fitted coordinate.
- Use explicit exception or status types for posterior, rank, and numerical alignment failures so aggregate validation can distinguish scientific and infrastructure outcomes.

## Future work

- After this ticket passes review, cff-45ia must issue a new registration and repeat only its four-fit funnel. The new artifact must link and supersede the immutable failed attempt.
- cff-45ia owns the paired prior cohort and any expansion decision. This ticket does not authorize a fit.
- The later classic calibration and freeze must consume the same authoritative D10 belief helper and fit-complete checkpoint contract.

## Definitions

- **Authoritative D10 belief:** A singleton, consecutive-pair, Gaussian, or lag moment returned from the active structured-hybrid D10 component rather than a superseded generic message arena.
- **Fit-complete checkpoint:** An immutable record written as soon as a fit returns, before truth-dependent metrics or rollouts run.
- **Affine alignment:** A fitted linear transformation that maps inferred latent coordinates to development truth coordinates for scoring.
- **Column scaling:** Dividing design-matrix columns by local finite scales before a numerical solve, then transforming the result back to the original units.
- **Process high-water RSS:** The largest resident memory used by the process up to the recorded point.

## Notes

Architecture handoff: this changes the experiment consumer boundary for observed-only rSLDS result construction, D10 public readout selection, checkpointing, secondary metrics, confirmatory metrics, and artifact resume. Core compiler, engine, structured-hybrid inference, D10 learning, and persistence are consumed without behavior change. Adapter, generator, calibration, and final execution are deliberately excluded.

Hostile-input decisions: exact categorical zeros remain exact; improper or non-finite D10 beliefs fail before NumPy; all consistency and rank tolerances scale with dtype and local operands; 10,000-step extraction is linear in time and does not materialize switching paths.

Legend: prerequisite --> dependent

```text
[x] cff-45ia authenticated post-fit readout failure
      |
      +--> [ ] cff-9w2d authoritative D10 readout/checkpoint (this ticket)
                  |
                  +--> [ ] cff-45ia newly registered four-fit funnel
                              |
                              +--> [ ] paired cohort expansion
                                          |
                                          +--> [ ] cff-3v7q periodicity decision
```
