---
id: cff-0l31
status: open
deps: [cff-64ri, cff-garu, cff-r6bl, cff-l7y4, cff-11jj]
links: []
created: 2026-08-10T02:58:08Z
type: task
priority: 1
assignee: coopwilliams
parent: cff-nkbj
tags: [rslds, validation, statistics]
---
# Freeze the rSLDS parity harness and preregistration

## Background in Plain English

The final comparison must not choose favorable seeds, thresholds, or stopping
rules after seeing confirmatory results. cff-akmp and cff-64ri provide the two
reference boundaries: the narrower unmodified learner and the independent
full source-specific paper-equation adapter. Closed tickets cff-l7y4,
cff-garu, and cff-r6bl provide the CFFG fitting workflow, stabilized repeated
coordinates, and practical canonical-scale setup.
Ticket cff-11jj moves that production fitting path onto declared learnables
before this protocol freezes, so the confirmatory result does not validate a
model-specific learning path that the roadmap has already scheduled for
replacement.

This ticket makes the comparison executable and freezes its statistical rules
before final seeds run. It includes calibration because calibration determines
budgets and meaningful non-inferiority margins. It excludes every final seed.

## Outcomes / Acceptance Criteria

1. A deterministic mapping test relates CFFG and adapter coefficients, source
   and destination labels, decision-list gates, initial distributions,
   dynamics, emissions, and priors without using generating truth.
2. Small exact enumeration shows that mapped point parameters produce the same
   transition probabilities for multiple source states and continuous inputs.
3. The harness has two primary comparison tiers.
   3.1. A shared-initialization tier maps one truth-free starting state into
   both full source-specific implementations.
   3.2. An end-to-end tier uses each implementation's documented truth-free
   initializer.
4. The pinned unmodified recurrence-only upstream learner is registered as a
   secondary nested-model comparator and not as the same-model primary
   reference.
5. The same-`K` and closest-free-scalar-budget learned SLDS controls use the
   identical observations, splits, and seeds as the rSLDS arms.
6. Calibration seeds are named, committed, and disjoint from cff-l7y4 seeds
   `0` and `101`, every cff-garu repair seed, and every final seed.
7. Calibration fixes the CFFG and reference iteration budgets, stopping rules,
   particle or chain counts, initialization settings, and failure handling.
8. The final sample contains at least 20 named data seeds and uses one
   canonical-scale dataset per seed with `T=10,000`, `K=4`, latent dimension
   `2`, and observation dimension `10` unless the preregistration gives a
   specific reference-driven reason for a separate scale tier.
9. Primary metrics are held-out mean predictive log likelihood,
   affine-aligned latent RMSE, and label-aligned regime accuracy.
10. Exact non-inferiority margins are justified from calibration repeatability
    or a named practical tolerance and frozen before final execution.
11. The primary conclusion requires every metric to pass in both primary
    comparison tiers under paired simultaneous 95% confidence intervals or
    another preregistered family-wise error control.
12. The failure policy counts every non-converged, budget-exhausted, diverged,
    invalid, or missing seed against the registered conclusion or declares the
    result not ready. It never drops a failed seed silently.
13. The unmodified upstream secondary claim is limited to outcome comparison
    on recurrence-only data and does not claim same-model update parity.
14. Compact tests verify mapping, alignment, paired aggregation, interval
    construction, failure accounting, manifest validation, and truth-boundary
    enforcement without running a canonical dataset.
15. The protocol inherits exact-zero support and loud runtime statuses from
    CFFG. It names the float64 legacy-reference boundary, exercises compact
    float32 CFFG fitting, and treats the final `T=10,000` sample as the
    uncomfortable-scale case.
16. A preregistration artifact records all arms, metrics, thresholds, seeds,
    commands, environments, commits, data hashes, calibration outputs, and a
    content hash before cff-oyie begins.
17. The CFFG arm invokes the production declared-learning path delivered by
    cff-11jj. A compatibility-only manual `RSLDSLearning` path cannot satisfy
    the production arm.
18. No final seed is executed by this ticket.

## Methodological Suggestions

These suggestions are optional. The implementer may choose another sound
method that satisfies the outcomes.

- Keep the driver in `../experiments/cffg-nascar-validation` and reuse the
  artifact schema and truth-information boundary from cff-l7y4.
- Use paired resampling over data seeds and register how Monte Carlo noise is
  controlled within each seed.
- Run seeds sequentially unless direct memory measurements justify bounded
  parallelism.

## Future work

cff-oyie is the only ticket authorized to run the frozen final seed list and
publish the confirmatory conclusion. Any defect found before the
preregistration hash is frozen becomes a separate prerequisite ticket rather
than an undocumented protocol change.

## Definitions

- **Calibration seed:** A dataset seed used to choose protocol settings and
  therefore excluded from the final evidence.
- **End-to-end tier:** A comparison in which each implementation uses its own
  documented observation-only initializer and fitter.
- **Family-wise error control:** A rule that limits false claims across the
  complete set of primary metrics.
- **Nested-model comparator:** A baseline whose parameter family is a
  restricted subset of the primary model family.
- **Non-inferiority margin:** The largest registered loss that still counts as
  not meaningfully worse than the reference.
- **Preregistration:** A committed protocol that fixes scientific decisions
  before confirmatory results are observed.

## Notes

**2026-08-10T03:20:00Z**

Wiggle-audit retained candidates and intended meanings: canonical=the named
`T=10,000`, `K=4`, `d=2`, observation-dimension-10 scale; end_to_end=the defined
native truth-free workflow tier; register=freeze in the preregistration;
reusable=reuse the named harness schema; same=equal probabilities or the
explicit same-model/same-`K` controls; implement and provide=the two named
software workflows and prerequisite capabilities; support=literal zero
support; complete=the enumerated primary metric family; ready=the explicit
not-ready scientific outcome; close=historically closed prerequisites;
material=the calibrated practical-loss boundary; converge=the named runtime
statuses; evidence=the final confirmatory set excluded here; exact=finite
enumeration, fixed margins, or literal zero; parity and pass=the registered
same-model non-inferiority decision; sound=another method satisfying outcomes;
stable=cff-garu's repeated-coordinate behavior; test, validate, and verify=the
specified mapping, statistics, manifest, and truth-boundary checks; fit and
learn=observation-only parameter estimation; scale and small=the named
canonical versus compact cases; may=optional method; explicit_owner=cff-oyie
owns final execution; all, any, at_least, every, and only=the locally listed
arms, seeds, metrics, phase boundary, and one authorized final-run owner.
Capability owner: cff-0l31 owns calibration, harness correctness, and the
immutable protocol, but no final evidence. Intelligible-ticket review:
complete.

Classification-note additions: fixed=the preregistered budgets and thresholds;
check=the mapping, aggregation, or artifact check named locally;
correct=harness results that satisfy those executable checks; estimate=the
named parameter-fitting workflow; intended=the contextual assignments in this
note; named_ticket_handoff=cff-oyie as the final-run owner.
