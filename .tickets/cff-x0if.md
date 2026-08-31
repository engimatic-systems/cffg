---
id: cff-x0if
status: open
deps: [cff-ocxl]
links: []
created: 2026-08-31T03:47:35Z
type: task
priority: 2
assignee: coopwilliams
parent: cff-zov5
tags: [rslds, learning, validation]
---
# Qualify observation-only source-specific transition recovery

## Background in Plain English — WHY

cff-iur0 and cff-ocxl change the inference approximation intended to supply
source-specific recurrence learning. This ticket asks one bounded development
question: with regimes, continuous states, and generating parameters withheld,
does the public learner now recover distinct current-regime transition
functions on adequately informative synthetic observations?

This is not a Linderman-code comparison and not a confirmatory experiment.
The historical source-heterogeneous program already established useful pinned
choices: at `6,400` total steps with `5,120` training steps, 300 of 300 declared
development datasets passed the source/gate coverage rules, with an exact
one-sided 95% lower bound of `0.990064`. The `0.150` capability ceiling remains
unchanged for both worst-source transition error and worst source-contrast
error. Reusing those choices avoids a new calibration arc.

## Outcomes / Acceptance Criteria — WHAT

1. Before the first fit starts, one committed development manifest fixes:
   heterogeneous generator/schema identity; CFFG and runner commits; `K=4`,
   continuous dimension `2`, observation dimension `10`; `6,400` total and
   `5,120` training steps; dataset seeds `5101`, `5102`, and `5103`;
   initializer keys `7101`, `7102`, and `7103`; the public initializer and
   priors; schedule, damping, inner budget, outer budget, stopping rules;
   common-state grid; alignment; metrics; failure policy; and artifact schema.
2. Each of the three fits receives only its training observations,
   configuration, priors, and explicit random key. Generating regimes,
   continuous states, transition parameters, dynamics, and emissions remain
   unavailable until the fitted artifact is sealed.
3. A prefit check requires every source/gate cell to have at least five
   successes, five failures, twenty exposures, augmented-design rank three,
   and finite declared conditioning. A failing dataset receives a not-ready
   record and is not replaced with another seed.
4. All three fits return `converged` under the fixed budgets, with finite inner
   Bethe free energies, finite installed parameters, coherent terminal state,
   and the approximation identity delivered by cff-ocxl.
5. After label and affine alignment used only for scoring, every dataset has
   worst-source transition root-mean-square error at most `0.150` and worst
   source-contrast root-mean-square error at most `0.150` on the frozen
   common-state grid.
6. The report gives on-support transition errors beside common-grid errors,
   plus per-source/per-gate exposure and error. Strong on-support performance
   cannot substitute for a missed common-grid ceiling, and sparse
   counterfactual cells cannot be hidden in an aggregate mean.
7. Held-out predictive log likelihood, aligned regime accuracy, aligned latent
   error, outer residuals, iteration counts, status, wall time, and peak memory
   are reported as diagnostics. They do not replace criteria 4 or 5.
8. One source-homogeneous classic fixture retains the checked-in D9 baseline's transition
   probabilities, D9 reduction, and observation-only status behavior. It is a
   regression check, not a second empirical claim.
9. Artifact validation recomputes every gate-bearing metric and conclusion
   from sealed per-dataset records and records repository, runner, generator,
   configuration, data, and result digests.
10. The run executes exactly the three declared fits. It runs no Gibbs adapter,
    learned SLDS control, extra initialization tier, replacement seed,
    confirmatory cohort, or long-rollout certification.
11. If any gate fails, this ticket records the limitation and performs at most
    one diagnostic pass over the same sealed artifacts. It creates one ticket
    for the evidence-backed owner and does not change thresholds, add data, or
    implement the repair inside this ticket.
12. A passing result updates README.md and docs/LIMITATIONS.md with the exact
    demonstrated boundary. A failure updates docs/LIMITATIONS.md without
    weakening the gate or implying that representation or local learning
    algebra is absent.

## Methodological Suggestions — HOW

These suggestions are optional. The implementer may choose another sound
method that satisfies the outcomes.

- Reuse the source-heterogeneous NASCAR generator, coverage audit, common-grid
  scorer, and artifact validator preserved in the local sibling validation
  repository; do not move benchmark runners into the production package.
- Reuse the public `fit_rslds` cadence with cff-ocxl's declared approximation
  rather than adding a second model-specific learner.
- Inspect per-source conditional-stick error and source-conditioned continuous
  moments only after a gate fails. The sealed artifacts should make that
  diagnosis possible without refitting.

## Future work

A successful development result may unblock a separately authorized
confirmatory comparison or real-data study. cff-nkbj remains the owner of the
old paper-equation-adapter comparison; it is outside this ticket. A failed
result returns one named mathematical, component, learning, initialization, or
data-identifiability owner and stops.

## Definitions

- **Common-state grid:** The fixed continuous-state points at which every
  source row is compared, including source/state combinations not visited by
  one trajectory.
- **Development manifest:** A committed record that fixes inputs, budgets,
  metrics, and failure handling before the declared development fits run.
- **On-support error:** Transition-function error measured where the observed
  trajectory provides source-conditioned evidence.
- **Source contrast:** The difference between two current-regime transition
  functions evaluated at the same continuous state.
- **Worst-source transition error:** The largest transition-probability
  root-mean-square error over the four current regimes.

## Notes

Change-path handoff: this is an evidence ticket, so compiler, engine, IR,
learnable, checkpoint, and bundle changes are N/A unless a failing result
creates a separate owner. The runner and artifact work stays in the sibling
validation repository. In CFFG, only canonical capability/limitation text may
change after the result.

Duplicated-surface handoff: production host/JIT, flooding/scheduled,
single-instance/`vmap`, and current/legacy checkpoint pairs are N/A because
cff-ocxl owns them. This ticket checks the three fixed dataset/key pairs, all
four sources and three decision-list gates, on-support/common-grid views,
fitting/scoring information boundary, and pass/not-ready artifact paths.

Intelligible-ticket review: complete. This ticket owns exactly the frozen
three-fit development execution, artifact validation, and resulting canonical
claim/limitation text. cff-ocxl owns production behavior; cff-nkbj owns any
external-adapter comparison; a newly created ticket owns one diagnosed failure.

Wiggle-audit review retained clear contextual uses. `canonical` means the
named README/LIMITATIONS claim surface; `fixed` and `pinned` mean the explicitly
listed manifest values, budgets, grid, and three dataset/key pairs; `N/A` marks
the production surface groups excluded from this evidence ticket; `preserved`
means the existing sibling-repository artifacts remain available; `public`
means the exported initializer and `fit_rslds` cadence; `reuse` means using
those unchanged code/evidence surfaces; `same` means the identical sealed
artifacts or one continuous-state value; `implement` is the repair explicitly
forbidden here; `provide` means the measured historical coverage or
source-conditioned data; `support` means posterior/data support in the named
on-support metric; `delivered` means cff-ocxl's approximation identity;
`ready` names the explicit not-ready artifact status; `remaining` names a
preserved information boundary, unchanged threshold, or separately owned
comparison; `approximation` means cff-ocxl's versioned mathematical family;
`certification` is excluded long-rollout work; `check` means the stated prefit,
regression, or surface inspection; `converged` is the public outer status;
`evidence` is a source-conditioned observation or failure diagnostic; `exact`
means the three-fit count, the confidence-bound calculation, or exact
documented result boundary; `pass` means a diagnostic pass, successful result,
or named artifact branch as stated; `sound` means another method satisfying
acceptance; `validation` means artifact recomputation or the named sibling
repository; `exposure` is a binary-gate trial count; `fit` means parameter
estimation on one fixed dataset; `infer` means the posterior approximation;
`learn` and `recover` mean the integrated fitting workflow or estimation of
withheld transition functions; `learnable` names the excluded IR family;
`performance` refers only to the on-support metric phrase, not runtime;
`intended` states the approximation's role; `may` marks optional downstream
authorization or a file-scope allowance; `should` states the diagnostic
artifact requirement; `all`, `any`, `at least`, `every`, and `only` are bounded
by the three fits, four sources, three gates, named thresholds, and explicit
scope exclusions; and `current` means mathematical current regime or the
current/legacy checkpoint pair, not an unpinned repository claim.
