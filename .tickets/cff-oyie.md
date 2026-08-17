---
id: cff-oyie
status: open
deps: [cff-0l31, cff-ah4t]
links: []
created: 2026-08-10T02:58:08Z
type: task
priority: 1
assignee: coopwilliams
parent: cff-nkbj
tags: [rslds, validation, statistics]
---
# Run the confirmatory multi-seed rSLDS comparison

## Background in Plain English

cff-0l31 freezes the complete comparison before any final seed runs. This
ticket executes that frozen protocol and reports the result. Separating final
execution prevents implementation repair, threshold selection, and evidence
collection from becoming one unreviewable activity.

The primary question is whether CFFG is non-inferior to an independent
paper-equation Pólya–Gamma Gibbs reference on the same full source-specific
rSLDS model. The unmodified legacy learner remains a useful but narrower
secondary baseline.

## Outcomes / Acceptance Criteria

1. The run starts from the committed cff-0l31 preregistration hash and changes
   no registered seed, arm, threshold, budget, stopping rule, metric, alignment
   rule, or failure policy after the first final seed begins.
2. Every registered final dataset and comparison arm runs or receives an
   explicit failed or not-ready record.
3. Neither CFFG nor either reference arm receives generating states, regimes,
   or parameters before fitting finishes.
4. Both primary tiers report paired per-seed values for held-out predictive log
   likelihood, aligned latent RMSE, aligned regime accuracy, runtime status,
   and every registered secondary metric.
5. The primary conclusion applies the registered simultaneous uncertainty
   procedure and passes only if all three primary metrics pass in both tiers.
6. The unmodified upstream learner is reported separately as a narrower
   recurrence-only workflow baseline and is not used to claim same-model
   parameter or algorithm parity.
   6.1. Passing the primary comparison is described as source-specific
   empirical non-inferiority for the registered outcomes, not as authors-code
   parity or validation of CFFG-paper message and free-energy equations.
7. The same-`K` and closest-free-scalar-budget learned SLDS controls run on the
   identical final observations, splits, and seeds.
8. The report includes fitted parameter-family errors, bounded long-rollout
   periodicity and bounded-rollout rate, outer iterations, inner work, phase
   timings, and peak resident memory as secondary outcomes.
9. Aggregate tables retain every failed seed and include paired effects,
   uncertainty intervals, and the registered non-inferiority decisions rather
   than only means or successful fits.
10. Artifacts record repository, experiment, generator, upstream, adapter, and
    environment commits; exact commands; hardware; seeds; configurations;
    data hashes; raw metrics; statuses; timing; memory; preregistration hash;
    and a complete manifest hash.
11. Artifact validation recomputes all registered aggregates and conclusions
    from raw per-seed records.
12. README.md states the supported conclusion in plain language.
    docs/ARCHITECTURE.md records the reference and fitting boundaries.
    docs/LIMITATIONS.md records every missed metric, non-converged seed,
    model mismatch, and single-benchmark limitation.
13. If the primary comparison misses a threshold, the report says so and
    creates or links one ticket for each evidence-backed failing owner. It does
    not change the threshold or silently extend this ticket into repair work.
14. Exact-zero support, float32 CFFG coverage, loud non-convergence, and the
    canonical-scale case retain the numerical contracts frozen by cff-0l31.

## Methodological Suggestions

These suggestions are optional. The implementer may choose another sound
method that satisfies the outcomes.

- Validate each seed artifact immediately, but do not inspect aggregate final
  effects until the registered seed set is complete unless the failure policy
  requires stopping.
- Keep raw outputs append-only and derive every table from them.
- Reuse the dashboard-backed short-suite result when repository code changes;
  the marked slow final experiment remains outside the default pytest gate.

## Future work

The cff-nkbj epic closes after this ticket produces a complete, reviewable
conclusion. Real observations, other switching-system families, balanced-tree
transitions, and destination-softmax transitions remain separate roadmap
work. Failed layers receive their own tickets under criterion 13.

## Definitions

- **Confirmatory run:** Evidence collected after the protocol and thresholds
  have been fixed.
- **Manifest hash:** A digest covering the complete declared artifact set.
- **Not-ready record:** An explicit result stating that the registered
  comparison could not support a conclusion.
- **Primary tier:** One of the two same-model comparisons against the
  full source-specific paper-equation adapter.
- **Secondary outcome:** A registered measurement that informs diagnosis but
  cannot replace a failed primary threshold.

## Notes

**2026-08-10T03:20:00Z**

Wiggle-audit retained candidates and intended meanings: canonical=the scale
frozen by cff-0l31; fixed=scientific choices frozen before execution;
register=each preregistered arm, seed, metric, rule, or aggregate;
reusable=reuse the unchanged dashboard result; same=the full model, same-`K`
control, or named same-model tier; implement=implementation repair excluded
from this phase; support=literal probability support or a conclusion justified
by results; complete=the full comparison, manifest, seed set, or declared
artifact set; ready=the explicit not-ready result; remaining=the useful legacy
baseline or deferred work; close=epic closure after acceptance; converge=the
engine's named non-converged statuses; evidence=confirmatory artifacts or a
failure-linked diagnosis; exact=commands or literal zero; parity and pass=the
registered threshold decision; sound=another method satisfying outcomes;
test and validate=the slow experiment, artifact integrity, or aggregate
recomputation as stated; fit and learn=observed-only fitted parameters;
scale=the frozen canonical case; could=the not-ready possibility; may=optional
method; explicit_owner=new repair tickets own any diagnosed failure; all, any,
every, and only=the registered metric/seed/artifact universe and hard scope
boundaries. Capability owner: cff-oyie owns immutable final execution and
reporting only. Intelligible-ticket review: complete.

Classification-note additions: intended=the contextual meaning assignments in
this note; named_ticket_handoff=the new repair ticket required by criterion 13;
deferred=the explicitly excluded model families and datasets.

**2026-08-13T05:19:12Z**

PR merge-gate correction (2026-08-12): the final report may claim only source-specific empirical non-inferiority for the registered outcomes. It must explicitly deny authors-code parity and CFFG-paper message/free-energy validation.
