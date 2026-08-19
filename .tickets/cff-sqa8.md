---
id: cff-sqa8
status: in_progress
deps: []
links: []
created: 2026-08-19T04:09:10Z
type: bug
priority: 1
assignee: coopwilliams
parent: cff-nkbj
tags: [rslds, validation, numerical-robustness]
---
# Record divergent secondary rollouts as unbounded evidence

## Background in Plain English

The dual-family calibration completed all eight registered coordinates, but one
reference-chain secondary rollout became numerically unbounded. Its latent state
left finite floating-point range at step 1,704. The scorer then passed invalid
transition probabilities to NumPy, which raised `Probabilities contain NaN`.
That exception made the entire tier not ready even though the primary reference
metrics were finite and the scientifically relevant secondary result was simply
that this fitted model did not produce a bounded rollout.

The evaluation must distinguish an observed unbounded model from a broken or
missing calculation. A detected rollout divergence is negative boundedness
evidence. Unexpected scorer errors, invalid inputs, and non-finite primary
metrics must remain loud failures.

## Outcomes / Acceptance Criteria

1. The generative-rollout evaluator detects the first non-finite state or
   invalid transition-probability vector and reports a structured divergence
   with its step and reason instead of leaking a downstream NumPy exception.
2. The secondary scorer converts only that detected rollout divergence into a
   complete finite record with `bounded` false, periodicity score zero, and the
   divergence metadata.
   2.1. Parameter-recovery and functional transition results that completed
        before the rollout remain available.
   2.2. Invalid inputs, malformed posterior parameters, and unexpected scorer
        exceptions still make the arm not ready.
3. Stable finite rollouts retain their existing seeded trajectory and
   periodicity results.
4. Rollout eligibility uses the protocol-declared variance-growth and amplitude
   bounds. The scorer does not keep a second hard-coded copy of those values.
5. Regression coverage reproduces the mechanism with finite fitted parameters
   whose unstable dynamics leave floating-point range.
   5.1. Direct rollout execution reports the first divergence loudly.
   5.2. Secondary evaluation records the same case as complete unbounded
        evidence with no serialized NaN or infinity.
   5.3. Float32 and a 10,000-step finite rollout remain supported.
6. No registered final seed is generated, fit, or inspected by this ticket.

## Methodological Suggestions

These suggestions are optional. An implementer may choose another sound method
that satisfies the numbered outcomes.

- Give rollout divergence a dedicated exception or result type so callers do
  not classify failures by matching NumPy error text.
- Validate each transition-probability vector and sampled state at the point
  where it is produced. Keep dtype-scaled probability tolerances.
- Catch the dedicated divergence only at the secondary-evaluation boundary.
  Do not clamp states, project probabilities, or replace unstable dynamics.
- Thread the registered bounds through `point_secondary`, `engine_secondary`,
  `reference_secondary`, and the control scorer.

## Future work

`cff-s6rs` reruns the calibration under one coherent experiment revision after
this fix and recalibrates the exhausted CFFG inner-solve budget. `cff-huxt` may
freeze only the resulting complete calibration artifact. `cff-oyie` remains the
sole owner of final-seed execution.

```text
Legend: prerequisite --> dependent

[ ] cff-sqa8 (ready) --> [~] cff-s6rs --> [ ] cff-huxt
                                             |
                                             +--> [ ] cff-oyie --> [~] cff-nkbj
```

## Definitions

- **Bounded rollout:** A generated trajectory whose registered variance-growth
  and amplitude ratios stay within their declared limits.
- **Rollout divergence:** A generated latent state or transition-probability
  vector leaves the finite numerical domain before the requested horizon ends.
- **Secondary outcome:** Descriptive evidence that supplements the registered
  primary comparison but does not replace it.
