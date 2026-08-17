---
id: cff-0l31
status: closed
deps: [cff-64ri, cff-garu, cff-r6bl, cff-l7y4, cff-11jj, cff-v8a1]
links: []
created: 2026-08-10T02:58:08Z
type: task
priority: 1
assignee: coopwilliams
parent: cff-nkbj
tags: [rslds, validation, statistics]
---
# Freeze the rSLDS comparison harness and preregistration

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
   2.1. Compact CFFG cases compare each recurrence-link outgoing message before
   end-to-end normalization and its matching factor-local energy or normalizer
   with an independent analytic, enumeration, or quadrature oracle.
   2.2. The local oracle imports no production recurrence message or energy
   implementation and checks the declared approximation or bound direction.
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
   7.1. For each Gibbs arm, it also fixes warmup, retained draws, independent
   chain count, stochastic-key streams, posterior-sample retention, a minimum
   effective sample size, and a maximum split R-hat.
   7.2. A chain that misses a registered mixing diagnostic follows the failure
   policy instead of contributing an apparently valid outcome.
8. The final sample contains at least 20 named data seeds and uses one
   canonical-scale dataset per seed with `T=10,000`, `K=4`, latent dimension
   `2`, and observation dimension `10` unless the preregistration gives a
   specific reference-driven reason for a separate scale tier.
9. Primary metrics are held-out mean predictive log likelihood,
   affine-aligned latent RMSE, and label-aligned regime accuracy.
10. Each exact non-inferiority margin is justified by calibration
    repeatability and is no wider than a separately named domain-meaningful
    maximum loss, with both rules frozen before final execution.
11. The protocol states each paired estimand, effect direction, confidence
    interval construction, and multiplicity procedure. The primary conclusion
    requires every metric to pass in both tiers under simultaneous 95%
    confidence intervals or another preregistered family-wise error control.
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
    stochastic-key streams, commands, environments, commits, data hashes,
    calibration outputs, retained posterior-sample locations, and a content
    hash before cff-oyie begins.
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
- **Effective sample size:** The approximate number of independent posterior
  draws represented by an autocorrelated Gibbs chain.
- **Family-wise error control:** A rule that limits false claims across the
  complete set of primary metrics.
- **Nested-model comparator:** A baseline whose parameter family is a
  restricted subset of the primary model family.
- **Non-inferiority margin:** The largest registered loss that still counts as
  not meaningfully worse than the reference.
- **Preregistration:** A committed protocol that fixes scientific decisions
  before confirmatory results are observed.
- **Split R-hat:** A multi-chain diagnostic that detects chains whose sampled
  distributions have not mixed to the same stationary target.

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

**2026-08-13T05:19:12Z**

PR merge-gate correction (2026-08-12): the frozen protocol now includes a direct production recurrence-message and factor-energy audit, Gibbs key streams and retained samples, ESS and split-R-hat gates, calibration-derived margins capped by domain-meaningful loss, and the exact paired estimand, direction, and multiplicity procedure.

**2026-08-17T17:24:25Z**

Pre-implementation handoff: docs/ARCHITECTURE.md change-path index row is N/A because implementation is confined to the sibling experiment harness and consumes, but does not alter, the CFFG production API. Duplicated execution surfaces audited: primary CFFG native initializer/fitter; primary CFFG mapped shared initializer; primary paper-adapter native Gibbs initializer/fitter; primary paper-adapter mapped shared initializer; secondary pinned unmodified recurrence-only learner; same-K SLDS; closest-budget SLDS; mapping/oracle tests; calibration; paired aggregation; manifest/preregistration validation. CFFG compiler, VM, schedule, kernel, ledger, serialization, canonical docs, and production learning implementation are N/A. Legend: prerequisite --> dependent. [x] cff-11jj + [x] cff-64ri + [x] cff-v8a1 + [x] cff-l7y4/cff-garu/cff-r6bl --> [ ] cff-0l31 (ready/in progress) --> [ ] cff-oyie --> [ ] cff-nkbj epic. Parallel controls and oracle work live inside cff-0l31 and converge before the immutable preregistration artifact is hashed.

**2026-08-17T18:21:27Z**

Completed in the sibling experiment repository on branch cff-0l31-rslds-preregistration at artifact head 3af4709. The immutable preregistration is artifacts/rslds-preregistration/preregistration.json with content SHA-256 0ca693cc7eb7821da161a763e0262e1495ff19d6fe92cf6445b7f01a3cf97e4c. Its manifest hashes 118 calibration and registration payloads. The registration names experiment code 0b93627 and CFFG declared-learning commit cd4dbd3; calibration ran from experiment commit 13c70f4.

Implementation evidence: the isolated paper adapter now exposes generic shared and native observation-only Gibbs chains, full-covariance shared emissions, exact CFFG prior mapping, retained per-draw parameters and latents, and loud per-chain statuses. The modern harness implements both primary tiers, both registered SLDS controls, independent NumPy recurrence messages/energy/quadrature, mapping and transition enumeration, split R-hat/ESS, paired Bonferroni lower intervals, strict failure accounting, exact stream registration, calibration derivation, and tamper-evident freeze validation. The native legacy initializer uses only observation SVD, clustering, and regularized regression. The primary CFFG arm calls fit_rslds through declared LearnableSpec plans; the manual compatibility path is forbidden by validation.

Calibration used only seeds 3101 and 3102 with two replicates, 32/32 successful isolated chains, eight ready primary rows, ready same-K and closest-budget controls, unique stochastic streams, and no metric errors. Wall time was 500.095 seconds with peak RSS 3,460,907,008 bytes. A pilot defect was caught before freeze: cross-chain parameter averaging was invalid under label switching and produced pathological predictions. The fixed score evaluates one within-chain posterior point summary per chain and combines the likelihoods as an equal mixture. Diagnostics now use label-order-invariant summaries. Measured concurrent chain execution doubled wall time, so the frozen protocol uses sequential isolated subprocesses.

Frozen final protocol: seeds 2001 through 2024; T=10,000 with 8,000 training observations; K=4, d=2, p=10; CFFG maximum 10 outer iterations, 1,000 inner iterations, tolerance 1e-4, damping 0.9, 512 particles; four reference chains, 100 warmup, 336 retained draws, thin 1, ESS at least 50 and split R-hat at most 1.05. Non-inferiority margins are 0.10 held-out mean predictive log likelihood, 0.0025 affine-aligned latent RMSE, and 0.02 label-aligned regime accuracy in both tiers. The simultaneous family is all six tier/metric effects. Any failed primary or control arm makes the complete result not ready. The registered secondary comparator remains outcome-only and makes no same-model claim. Phase boundary final_seed_executed=false; no final dataset or artifact exists.

Validation at the proposed experiment head: 24 modern tests passed in 21.10 seconds; 35 isolated Python 3.11 legacy tests passed; the freeze reproduced the identical content hash; manifest and preregistration validators passed; git diff --check passed. The two superseded generated calibration attempts were removed after their findings were distilled here.

Next: cff-oyie alone may run uv run cffg-rslds-confirmatory --phase final --preregistration artifacts/rslds-preregistration/preregistration.json --output artifacts/rslds-confirmatory. Do not change code, protocol, lockfiles, hashes, seeds, budgets, or thresholds after this freeze; a defect requires a new prerequisite ticket and new preregistration.
