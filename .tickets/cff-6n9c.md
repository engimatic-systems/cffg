---
id: cff-6n9c
status: open
deps: [cff-ny7f]
links: [cff-ny7f]
created: 2026-09-10T02:07:07Z
type: bug
priority: 2
assignee: coopwilliams
tags: [numerics, mu3]
---
# Reject malformed fixed particles in the mu3 importance rule

## Background in Plain English — WHY

The importance rule can report success while returning a NaN posterior mean
when a fixed particle contains NaN. At source ab17b5a, replace the first of
four positive, column-normalized 2-by-2 particles with NaN and call
rules.mu3_importance_update with uniform beliefs and a finite goal. Both the
direct call and jitted batching return status 0 with a NaN mean. rules.py is
unchanged from the pre-extraction source 1308688.

mu3_log_message tests support with particles > 0, which treats NaN as absent
support. The importance rule checks the resulting weights, not the particle
values or returned moments. This finding concerns the numerical rule boundary;
Engine.run separately checks all initial State leaves for finiteness. No false
public convergence or checkpoint bypass has been demonstrated.

## Outcomes / Acceptance Criteria — WHAT

1. The numerical importance rule rejects non-finite or negative particle values
   with its existing divergence code instead of returning successful estimates.
2. Its contract states whether column normalization is required and where it is
   checked. Valid exact-zero probabilities remain supported without clipping.
3. A successful estimate has finite normalized weights, finite moments and
   finite diagnostics. Collapsed weights retain the existing failure meaning.
4. Tests reproduce the malformed-particle mechanism directly and under jitted
   batching in float32 and float64, including one malformed particle beside
   valid particles and a tenfold sample count. Valid sparse-support and vacuous
   incoming-message cases retain the established rule result.
5. The handoff covers the Runtime numerical or status invariant change path:
   rules.mu3_log_message and mu3_importance_update own the numerical contract;
   execution/functional.py forwards its facts; Engine retains public status and
   fixed-particle lifecycle policy. Verify host-run protection and checkpoint
   rejection where those paths accept particle data; Colony mu3 execution is
   currently unsupported and is N/A here.
6. Existing derivation, rule and lifecycle tests and one final default suite
   pass. No sampler, objective, schema or new public status is introduced.

## Methodological Suggestions — HOW

Optional: distinguish malformed inputs from legitimate zero support before
support masks can hide them. Keep checks and failure facts as JAX arrays.
Any numeric normalization tolerance must follow the input dtype.

## Future work

cff-rqqc owns future Colony mu3 batching. This correction does not add that
feature and does not block the structural extraction in cff-ny7f.

## Definitions

- mu3: The functional parameter message evaluated on a fixed set of sampled
  probability tables.
- Particle: One sampled probability table used to estimate that message.
- Importance weight: A particle's normalized contribution to the estimate.
- NaN: A floating-point value meaning not a number.
- Jitted batching: JAX compilation of the same numerical function over a batch.
- Moment: A summary of a distribution, such as its mean.
