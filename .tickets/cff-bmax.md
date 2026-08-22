---
id: cff-bmax
status: in_progress
deps: []
links: []
created: 2026-08-22T20:21:34Z
type: task
priority: 1
assignee: coopwilliams
parent: cff-ngxx
tags: [rslds, inference, derivation, trustworthiness]
---
# Implement D10 structured hybrid coordinate authority

## Background in Plain English — WHY

The source-heterogeneous rSLDS fit misses both recovery caps even though the
local D3 dynamics and D9 recurrence rules reproduce independent calculations.
Replacing either learned dynamics or learned recurrence with its generating
development value restores both caps. This means the next repair must make
the two blocks optimize one approximation to the hidden path.

DERIVATIONS.md D10 selects a structured mean-field family with one categorical
path belief and one Gaussian Markov trajectory. This ticket establishes that
mathematics as independently callable code before any compiler or engine
dispatch changes. The authority must be useful outside the NASCAR experiment
and must not read generating states or parameters.

## Outcomes / Acceptance Criteria — WHAT

1. An exported `cffg.structured_hybrid` module implements the D10 structured
   hybrid coordinate without registering compiler or engine dispatch.
2. The categorical coordinate combines D9 source-row/destination-column
   recurrence potentials with D10.2 expected dynamics unary evidence and
   agrees with an independent NumPy K=2 and K=3 authority within a tolerance
   derived from the active floating dtype.
3. The Gaussian coordinate assembles D10.4 into a proper block-tridiagonal
   trajectory solve and returns means, marginal covariances, and adjacent-time
   cross moments that agree with an independent NumPy authority within a
   tolerance derived from the active floating dtype.
4. One objective authority evaluates every term in D10.6 from the returned
   beliefs. Exact coordinate updates are stationary for that objective, and a
   completed finite coordinate does not increase it beyond a dtype-scaled
   tolerance.
5. D10 reduces to the existing D9 exact categorical smoother when drives are
   fixed and to an ordinary linear Gaussian chain when K=1.
6. The returned singleton, pair, Gaussian, and lag moments independently reproduce
   the D10.5 component-MNIW increments and D9.14 PG increments. This ticket
   returns the values through the module API but does not install learned
   parameters.
7. Exact categorical zeros stay exact. Empty support, improper Gaussian
   precision, a non-finite objective or residual, and inconsistent shapes
   return a loud divergent or invalid status. Float32 guards come from the
   dtype. K=8 and T>=100 run without regime-path enumeration or cubic work in
   T.
8. The authority states the conservative bound in D10.7 and does not reuse
   D3's optimistic collapsed-mixture ledger label.

## Methodological Suggestions — HOW

These suggestions are optional. The implementer may choose another sound
method.

- Put pure categorical, block-Gaussian, objective, and status calculations in
  one focused module rather than extending already large D3 or D9 files.
- Reuse D9's pair-potential algebra and a Gaussian information-form smoother,
  but compare both against independent NumPy calculations that do not call the
  production functions under test.
- Use direct tiny-model enumeration or analytic Gaussian integration to check
  one complete objective, then use polynomial-time authorities for the hostile
  scale tests.
- Return named singleton, pair, Gaussian marginal, lag-moment, objective,
  residual, and status fields so cff-e10u can install that coordinate without
  recomputing it.

## Future work

cff-e10u compiles eligible graph components and dispatches this authority in
the engine. cff-6rav then binds learning and checkpoint identities to the
installed D10 beliefs. The cff-ngxx development pilot and cff-h0hx
requalification remain downstream; this ticket runs no NASCAR coordinate.

## Definitions

- **Structured mean-field:** A variational family that preserves the full
  categorical chain and the full Gaussian Markov trajectory but factors those
  two paths from each other.
- **Block-tridiagonal:** A matrix whose nonzero blocks lie on the main diagonal
  and the two adjacent diagonals. A Gaussian Markov chain has this form.
- **Lag moment:** The adjacent-time expectation
  `E[x_t x_(t-1)^T]`, needed for dynamics learning.

Architecture handoff: use the **New derived component coordinate** row in
`docs/ARCHITECTURE.md`. The focused authority and objective apply. Compiler
plans, engine dispatch, host/JIT pairing, flooding/scheduled pairing,
single/`vmap` pairing, learning installation, persistence, and model consumers
are N/A in this ticket and belong to cff-e10u or cff-6rav.

## Notes

**2026-08-22T20:28:40Z**

Implementation start after cff-ngxx cross-block and local-stat audits. The pure D10 authority must land and pass independent categorical, block-Gaussian, objective, learning-statistic, exact-support, float32, and K=8/T>=100 oracles before any compiler or engine dispatch changes. No NASCAR coordinate is in scope.
