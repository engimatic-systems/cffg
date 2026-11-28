---
id: cff-6n9c
status: in_progress
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

## Notes

**2026-11-28T03:57:33Z**

The numerical rule will validate each fixed table before masking zero support, using the particle dtype and the same unit-column allowance as Dirichlet preparation. One malformed sample fails its entire bank. Checkpoint restore will use the same particle-column bound while retaining its existing weight and moment tolerances. Direct and jitted lane tests will cover exact zeros, uniform incoming evidence, both dtypes, mixed good and bad banks, and tenfold sample count. Engine and checkpoint checks cover the public boundaries. The parent owns the final short-suite gate.

**2026-11-28T04:04:33Z**

Implementation handoff. The fixed-table check runs for each particle before support masking. One NaN, infinity, negative cell, or non-unit column makes its bank diverge. Preparation and checkpoint restore use the same 128-epsilon column allowance in the particle dtype. Exact zeros and independent valid batch lanes retain their weights. The importance calculation subtracts a common log score before normalizing and evaluating KL, preserving the self-normalized objective even when the common score dwarfs the relative evidence. If the two saved scores lose their KL difference to rounding, restore verifies the KL against the saved weights as well as the roundoff bound. Engine keeps its existing divergence and low-ESS policies. Colony mu3 is unsupported.

Focused verification with PYTHONPATH=src and the shared uv environment: uv run --frozen --no-sync pytest -q tests/test_mu3.py tests/test_mu3_derivation.py tests/test_agent_mu3.py tests/test_lifecycle.py tests/test_execution_specialized.py -k 'particle or importance or mu3 or functional' --tb=short. All 39 selected tests passed in 37.71 s. Tests cover direct and jitted float32/float64, mixed score precision, 4 and 40 particles, sparse support, uniform evidence, the quadrature oracle, Engine divergence, atomic rejection of corrupted checkpoint samples, and the centered KL ledger. Dependency cycles and diff whitespace checks are clean. Remains in progress for the parent’s combined short-suite gate, PR and closure.
