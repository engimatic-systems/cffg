---
id: cff-fely
status: closed
deps: [cff-45ia]
links: []
created: 2026-08-24T11:26:41Z
type: task
priority: 1
assignee: coopwilliams
parent: cff-3v7q
tags: [rslds, validation, periodicity, diagnostic]
---
# Attribute classic rollout boundedness failures with retained parameter block swaps

Use the authenticated completed cff-45ia checkpoints to locate whether recurrence parameters, the remaining fitted parameter blocks, or their interaction causes the classic long-rollout boundedness failures. This is a development-only no-fit diagnostic.

## Background in Plain English — WHY

The prospective cff-45ia cohort completed all 24 D10 fits and 96 declared
10,000-step rollouts. Recurrence prior 1.0 preserved fit convergence,
transition recovery, prediction, and regime recovery, but passed only 43 of 48
rollout cells. Five cells were periodic yet exceeded the fixed amplitude or
variance-growth ceiling. Prior 0.001 passed 45 of 48 cells, and its three
failures did not overlap the candidate failures.

The retained fitted parameters show that every fit has at least one very
slightly expansive dynamics mode. That fact alone does not explain the misses:
each failing fit also generated bounded trajectories on other registered
streams, and the prior pairs changed recurrence parameters far more than their
dynamics matrices. The saved artifact does not retain rollout regime paths or
dwell times, so it cannot tell whether recurrence selects long visits to an
expansive mode, another parameter block creates the growth, or both blocks must
interact.

This ticket answers only that ownership question by recombining already fitted
parameter blocks. It cannot rehabilitate either prior or alter cff-45ia's
scientific miss.

## Outcomes / Acceptance Criteria — WHAT

1. Before rollout execution, one development-only registration authenticates
   cff-45ia experiment commit `b9c7461`, results SHA-256
   `08e2ecbac6f00b501fb7d883a59b7ee0f4a82afa7c0d3b730762579cf23841f7`,
   manifest SHA-256
   `857429cff80ede7dc07811e60e32c88fd6c7a1acf15eb46b0de2d093e2396f07`,
   all 24 fit checkpoints, the classic generator, and the original coordinate
   and rollout streams. Missing, moved, duplicated, or modified evidence fails
   before a rollout.
2. For each of the 12 matched seed/fit-replicate/tier pairs and each of its four
   registered rollout streams, the diagnostic evaluates the complete 2-by-2
   parameter design:
   - prior-0.001 recurrence with prior-0.001 non-recurrence parameters;
   - prior-1.0 recurrence with prior-0.001 non-recurrence parameters;
   - prior-0.001 recurrence with prior-1.0 non-recurrence parameters; and
   - prior-1.0 recurrence with prior-1.0 non-recurrence parameters.
   Initial-regime, dynamics, and emission parameters form the non-recurrence
   block. The 192 rollouts run sequentially; no fit, posterior update, or
   parameter interpolation occurs.
3. The two original design corners reproduce all 96 stored cff-45ia rollout
   summaries under the exact original streams before either hybrid is used as
   evidence. Reproduction covers requested/completed steps, boundedness,
   periodicity score, amplitude ratio, variance-growth ratio, and parameter and
   stream identities. Any mismatch stops the diagnostic.
4. Each rollout retains compact mechanism evidence: regime occupancy,
   transition counts, longest dwell by regime, maximum state norm and its time,
   quarter-trajectory variances, amplitude numerator and denominator, and
   recurrence transition-probability quantiles along the realized path. It does
   not persist the full 10,000-step trajectory or use generating truth.
5. Attribution is fixed before reading the hybrids:
   - `recurrence_owned` requires the two recurrence-swapped hybrids to follow
     the recurrence donor's boundedness and mechanism direction on every
     original miss cell and on a registered passing-control set;
   - `non_recurrence_owned` requires the corresponding result for the
     non-recurrence donor;
   - all other outcomes, including effects that require both donor blocks, are
     `interaction_or_ambiguous`.
   The registration names the passing-control cells and quantitative mechanism
   directions; no post-outcome subset may be chosen.
6. A `threshold_only` descriptive flag is allowed only when independent
   numerator/denominator recomputation matches the stored ratios and neither
   swap changes any registered mechanism statistic beyond its dtype- and
   operand-scaled replay tolerance. The fixed boundedness and periodicity
   thresholds never change.
7. The result separately reports each original miss, each passing control, all
   four parameter corners, and aggregate ownership evidence by tier, seed, fit
   replicate, and rollout stream. It remains append-only, resumable at one
   rollout cell, recursively manifested, and replay-valid after artifact
   commit.
8. Hostile tests cover non-finite parameters, exact categorical zeros,
   constant trajectories, an amplitude-only failure, a variance-only failure,
   float32 scoring, and a 10,000-step sequential execution without retaining
   all trajectories in memory.
9. No CFFG engine behavior, fitter, metric threshold, adapter chain,
   heterogeneous dataset, SLDS control, calibration seed, final seed, or
   confirmatory conclusion changes or runs in this ticket.

## Methodological Suggestions — HOW

These suggestions are optional. The implementer may choose another sound
method that satisfies the outcomes.

- Add a development diagnostic wrapper around the existing generative rollout
  so the production score remains unchanged while compact regime/path evidence
  is accumulated.
- Treat recurrence weights and offsets as one atomic donor block. Treat the
  initial-regime law, dynamics matrices/noise, and emission parameters as the
  other atomic donor block.
- Use the full factorial design rather than swapping only the five failing
  cells; the passing controls are what distinguish a causal block from a
  generally disruptive recombination.
- Derive numerical replay tolerances from the stored dtype and local operand
  scales. Exact identities, requested step counts, and Boolean boundedness are
  not tolerance-based.

## Future work

- If one block owns the failures, create a separate prospective stabilization
  ticket for that block before any new fit or freeze.
- If ownership is interactive or ambiguous, cff-3v7q must either derive a joint
  stability intervention or record that robust long-rollout adequacy is not
  currently met.
- cff-3v7q owns the classic periodicity decision. This ticket cannot select a
  prior or authorize calibration/final evidence.

## Definitions

- **Parameter block swap:** Recombining the recurrence parameters from one
  member of a matched prior pair with all remaining point parameters from the
  other member, without learning or interpolation.
- **Mode dwell:** A consecutive run of rollout steps assigned to one discrete
  regime.
- **Passing control:** A rollout cell named in the registration because both
  original prior corners were bounded and passed before any hybrid was run.
- **Mechanism direction:** A predeclared comparison such as increased dwell in
  a mildly expansive regime, increased maximum state norm, or increased
  quarter-to-quarter variance growth.

## Notes

Architecture handoff: the sibling experiment's development rollout,
parameter-checkpoint, compact diagnostic, scoring, and artifact-reporting
surfaces apply. Core compiler, engine inference, D10 learning, persistence,
adapter, calibration, and final-execution rows are N/A because parameters are
loaded read-only and no inference or learning runs.

Duplicated surfaces: both initialization tiers, all three data seeds, both fit
replicates, both registered prior corners, four rollout streams, parameter
serialization, and boundedness/periodicity scoring must remain aligned.

Hostile-input decisions: literal categorical zeros remain exact; non-finite or
wrong-shaped parameters fail before rollout; float32 guards use dtype-scaled
tolerances; constant and unbounded trajectories remain explicit failures; the
10,000-step mechanism accumulator is linear in time and stores only compact
summaries.

Legend: prerequisite --> dependent

```text
[x] cff-45ia complete paired D10 cohort (scientific miss)
      |
      +--> [ ] cff-fely retained-parameter block swaps (this ticket)
                  |
                  +--> [ ] prospective stabilization candidate
                              |
                              +--> [ ] cff-3v7q classic periodicity decision
```

**2026-08-24T12:42:19Z**

Completed acceptance-complete no-fit diagnosis in sibling experiment commit 18e61be24b0b3c8ca1e031b7d165c1a8166463fa. The manifested artifact validates after commit: results SHA-256 6cf8d86c446bcb92f2df72864e6300e749a867cbf5194e182be156f731235f48, manifest SHA-256 a93fa1a599516792d1cfa7a56c995587acdf7745dc5fbe6d6439bf88a0de6d3b, plan SHA-256 3c28ec7289cc85ab4d611fe6b12d4183fc2f443489e1049f0bd2cb3114affa73. All 192 registered 10,000-step rollouts completed; all 96 originals reproduced exactly before 96 hybrids were interpreted. Registered verdict: interaction_or_ambiguous. Of 8 original miss cells, 4 satisfy the full recurrence-donor rule, 0 the non-recurrence rule, and 4 neither; the boundedness/gate Boolean follows recurrence in 5/8. Of 40 passing controls, 39 keep all four corners passing; one hybrid introduces a new periodicity miss. Full five-metric recurrence proximity holds 16/40 controls, non-recurrence 0/40, threshold-only 0/48. Both independent reviewers authenticated the artifact and agreed the verdict must not be upgraded to recurrence ownership. This establishes strong partial recurrence influence plus genuine joint recurrence/non-recurrence behavior, not a universal one-block repair. No fit, data generation, adapter, control, heterogeneous coordinate, calibration seed, or final seed ran.
