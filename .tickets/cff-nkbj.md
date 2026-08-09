---
id: cff-nkbj
status: open
deps: [cff-l7y4, cff-garu]
links: [cff-r6bl]
created: 2026-08-09T13:09:26Z
type: feature
priority: 1
assignee: coopwilliams
parent: cff-ahps
tags: [rslds, validation, learning, statistics]
---
# Confirm multi-seed learned rSLDS parity with the pinned literature reference

## Background in Plain English

Ticket cff-l7y4 proved that CFFG can fit the synthetic NASCAR rSLDS from
observations at `T=10,000`. Its preregistered final used one seed and two
parameter-replacement coordinates. The learned rSLDS predicted held-out
observations better than the same-`K` learned SLDS, but it did not recover the
latent trajectory or regime sequence better. All three model fits also
exhausted the two-coordinate outer budget instead of reaching an outer fixed
point.

That result does not yet show that CFFG is as good as the rSLDS implementation
used in the literature. The cff-l7y4 experiment pinned the current
stick-breaking NASCAR generator at upstream commit `383ddac`, but it did not
run the upstream observed-only learner as a comparison arm. The repository
also records that the publication-era generator and later `ssm` demo use
different offsets, noise, seeds, or transition parameterizations. Calling any
one of them “canonical” without naming the exact model would make the result
impossible to review.

This ticket repeats learned fitting over a preregistered multi-seed sample
after cff-garu stabilizes inference beyond the third parameter replacement. It
compares CFFG directly with a pinned, executable literature reference and
retains the same learned-SLDS controls. The purpose is a non-inferiority test:
CFFG may claim literature parity only if it is not meaningfully worse on every
registered primary metric. A larger mean on one metric cannot hide a failed
convergence status or a material loss on another.

cff-r6bl may reduce the cost of these runs, but it is parallel work rather
than a prerequisite. The experiment may run slowly if cff-r6bl has not landed.

## Outcomes / Acceptance Criteria

1. Before any final seed is run, the preregistration names the exact literature
   target: paper or demo, upstream repository and commit, environment,
   transition parameterization, generator constants, initialization method,
   priors, fitting objective, and stopping rule.
   1.1. It states the differences found by comparing the publication-era
   NASCAR model, upstream commit `383ddac`, and the selected later `ssm`
   variant.
   1.2. If the executable reference is not parameter-equivalent to CFFG's
   `recurrence_link`, the result uses a new factor/derivation ticket or limits
   its claim to the common model. It does not call an approximate translation
   exact literature parity.
2. The external harness includes an observed-only reference-rSLDS arm pinned
   to the selected upstream implementation. Neither the CFFG arm nor the
   reference arm receives generating states, regimes, or parameters before
   fitting finishes.
3. The comparison has two registered tiers.
   3.1. A shared-initialization tier starts both implementations from mapped
   point parameters and priors derived without truth. It isolates fitting and
   inference behavior from initializer differences.
   3.2. An end-to-end tier uses each implementation's documented truth-free
   initializer. It compares the workflows a consumer would actually run.
4. The CFFG arm consumes the cff-garu correction and runs a longer outer loop.
   Its iteration budget and stopping rule are selected on calibration seeds
   and committed before final seeds run.
   4.1. `converged`, `budget_exhausted`, and `diverged` retain their existing
   meanings. The harness does not accept a small Bethe free-energy change in
   place of the complete engine certificate.
   4.2. No final seed is silently dropped. A preregistered failure policy
   counts non-converged fits against the parity conclusion or declares the
   result not ready.
5. The final sample contains at least 20 data seeds that are disjoint from
   cff-l7y4 seeds `0` and `101`, all calibration seeds, and every seed inspected
   while repairing cff-garu. The exact seed list and stopping rule are committed
   before the first final run.
6. The primary non-inferiority metrics are held-out mean predictive log
   likelihood, affine-aligned latent RMSE, and label-aligned regime accuracy.
   Exact numeric non-inferiority margins are justified from reference
   repeatability or a named practical tolerance on calibration data and are
   committed before the final seed set runs.
7. Literature parity is claimed only if all three primary metrics pass their
   registered non-inferiority tests in both comparison tiers. The analysis uses
   paired per-seed differences and simultaneous 95% confidence intervals or
   another preregistered family-wise error control.
8. The report includes every per-seed value, aggregate effect, uncertainty
   interval, convergence status, and failed or not-ready evaluation. It does
   not report only the mean or only successful seeds.
9. Secondary outcomes report every fitted parameter-family error, bounded
   long-rollout periodicity and bounded-rollout rate, outer iterations,
   inner-solver work, wall-clock phases, and peak resident memory. Secondary
   outcomes are not substituted for a missed primary threshold.
10. The same-`K` and closest-free-scalar-budget learned SLDS controls from
    cff-l7y4 run on the identical final observations, splits, and seed set.
    SLDS superiority or inferiority remains a separate conclusion from parity
    with the literature rSLDS.
11. The multi-seed harness uses one canonical-scale dataset per seed with
    `T=10,000`, `K=4`, latent dimension `2`, observation dimension `10`, and a
    preregistered train/test split unless criterion 1 proves that the selected
    literature target requires a different named scale. Any additional scale
    is reported as a separate tier.
12. The numerical robustness contract remains intact.
    12.1. Exact-zero regime support is not floored, and improper or non-finite
    beliefs remain loud non-converged results through inherited compact tests.
    12.2. Compact float32 fitting remains covered with dtype-derived guards.
    The canonical literature run may use x64 when required for like-for-like
    reference execution, and must state that boundary.
    12.3. The at-least-20-seed `T=10,000` final is the uncomfortable-scale
    case; it remains outside the default short suite.
13. A compact deterministic test verifies the reference/CFFG parameter mapping,
    affine and label alignment, paired aggregation, confidence-interval logic,
    and failure accounting without running the canonical-scale experiment.
14. Final artifacts record package, experiment, generator, and reference
    commits; environments; exact commands; seeds; data hashes; configurations;
    statuses; raw per-seed metrics; timing and memory; preregistration and pilot
    hashes; and a manifest hash.
15. README.md states the supported scientific conclusion in plain language.
    docs/ARCHITECTURE.md records the reference boundary and stabilized fitting
    cadence. docs/LIMITATIONS.md records every missed metric, non-converged
    seed, model mismatch, and single-benchmark limitation.
16. If CFFG misses literature non-inferiority, the result names the failing
    layer when evidence permits and creates or links a separate ticket for the
    initializer, variational approximation, message schedule, parameter
    coordinate, or factor parameterization that owns the gap.

## Methodological Suggestions

These suggestions are optional. The implementer may choose another sound
method that satisfies the outcomes.

- Use the Runtime numerical or status invariant change-path row for any fitter
  or engine change and keep the multi-seed driver in
  `../experiments/cffg-nascar-validation`.
- Reserve calibration seeds for choosing the post-cff-garu outer budget and
  non-inferiority margins. Never move a calibration seed into the final set.
- Use a paired bootstrap or paired permutation procedure over seeds. Register
  how Monte Carlo particle-filter noise is controlled before measuring final
  effects.
- Run seeds sequentially unless direct memory measurements justify bounded
  parallelism. Reuse compiled structure across seeds if a public, shape-safe
  interface supports it; do not capture observations or learned parameters as
  constants to gain speed.
- Keep the shared-initialization tier as the diagnostic authority when native
  initializers differ. Use the end-to-end tier for the consumer-facing claim.
- Reuse cff-l7y4's parameter counts, truth-information boundary, artifact
  schema, and bounded-rollout gate instead of inventing incompatible reports.

## Future work

cff-r6bl owns canonical rSLDS cold setup, repeated-slice lowering, and memory
optimization. It may make this experiment cheaper, but this ticket does not
wait for it or absorb its runtime implementation.

If the selected literature reference uses a softmax transition rather than
CFFG's source-specific stick-breaking recurrence, the required new factor and
free-energy derivation become a prerequisite ticket. This ticket must not
silently broaden `recurrence_link` semantics.

Validation on real observations, other switching-system families, and other
literature datasets remains future work after synthetic NASCAR parity. No
unnamed dataset is part of this ticket.

## Definitions

- **Calibration seed:** A dataset seed used to choose protocol settings or
  margins and therefore excluded from the final confirmatory sample.
- **Canonical NASCAR model:** Only the exact literature implementation and
  parameterization named by criterion 1; the phrase has no unqualified meaning
  in this ticket.
- **End-to-end tier:** A comparison in which each implementation uses its own
  documented observation-only initializer and fitter.
- **Family-wise error control:** A statistical rule that limits false claims
  across several primary metrics considered together.
- **Literature parity:** Passing every preregistered primary non-inferiority
  test against the pinned reference implementation.
- **Non-inferiority margin:** The largest registered loss that is small enough
  to count as not meaningfully worse than the reference.
- **Paired difference:** The metric difference between two implementations on
  the same generated dataset seed.
- **Shared-initialization tier:** A comparison that maps one truth-free starting
  state into both implementations to isolate fitting and inference behavior.

## Notes

**2026-08-09T13:11:56Z**

Ticket authoring handoff, 2026-08-09.

Legend: prerequisite --> dependent

    [x] cff-l7y4 -------------------+
                                     +--> [ ] cff-nkbj (blocked)
    [ ] cff-garu (ready) -----------+

    Parallel linked optimization: [ ] cff-r6bl (ready)
    Roadmap parent, not a dependency edge: cff-ahps

The tk dependency records are authoritative. `tk dep tree --full`, `tk ready`, and `tk blocked` show cff-garu as the only unresolved prerequisite; cff-l7y4 is complete. cff-r6bl is deliberately linked rather than required, so performance work may land in parallel. `tk dep cycle` and `git diff --check` pass.

The post-refinement wiggle scan retained 48 candidates. Intended meanings are: canonical=the exact named literature standard or its T=10,000 scale; end_to_end=the truth-free consumer workflow; fixed=the named outer fixed point; public=the documented shape-safe interface; register=the preregistered scientific protocol or claim; reusable=code, compiled-artifact, or prior-schema reuse; same=the named matched K, control configuration, or dataset seed; implement=the concrete CFFG or pinned reference software; support=probability support, a documented conclusion, or interface capability as stated; complete=the full engine certificate; land=closure of named cff-r6bl work; ready=the explicit not-ready evaluation state; remaining=preserved semantics or excluded future work; approximate=the named model translation or variational approximation; material and sufficient=the numeric non-inferiority margin; certify and converge=the public runtime status certificate; equivalent=parameter-equivalence or aligned representations; evidence=the named scientific or provenance artifact; exact=pinned identity, numeric protocol, or literal support; parity=the defined joint non-inferiority gate; pass=registered threshold or automated test success; robust=the AGENTS.md numerical contract; sound=bounded implementer judgment; stable=post-cff-garu repeated-coordinate execution; test, validate, and verify=the named automated, statistical, experiment, mapping, or artifact checks; fit and learn=observed-only repeated-dataset parameter estimation; infer=latent or regime posterior calculation; recover=aligned latent or regime reconstruction; scale=T, dimension, seed count, or runtime load as named; small=the named FE delta or registered practical margin; speed=runtime latency; can=the capability demonstrated by cff-l7y4; may=permission, optional method, or stated possibility; explicit_owner and named_ticket_handoff=the named ticket or component in each sentence; all, any, at_least, every, and only=the locally enumerated arms, seeds, metrics, or hard phase boundary; current=upstream commit 383ddac; existing=the status semantics that predate this ticket; later=the historical ssm variant or future ordering. Capability ownership was checked separately: cff-nkbj owns multi-seed reference parity, cff-garu owns repeated-coordinate stabilization, cff-r6bl owns cold setup, and a newly created prerequisite will own any missing transition factor or derivation. Intelligible-ticket review: complete.
