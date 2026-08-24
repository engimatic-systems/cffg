---
id: cff-bmax
status: closed
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
   beliefs, including arbitrary categorical unary evidence. The analytic
   auxiliary, categorical, and fixed-auxiliary Gaussian updates are exact
   coordinates of the expanded PG/JJ bound. A complete finite sweep does not
   increase that bound beyond a dtype-scaled tolerance; only a joint fixed
   point claims stationarity for the auxiliary-eliminated objective.
5. A conditioned drive reduces directly to the existing D9 exact categorical
   smoother without constructing a singular Gaussian belief. K=1 bypasses D9
   and reduces to an ordinary linear Gaussian chain.
6. The returned singleton, pair, Gaussian, and lag moments independently reproduce
   the D10.5 component-MNIW increments and D9.14 PG increments. This ticket
   returns the values through the module API but does not install learned
   parameters.
7. Exact categorical zeros stay exact. Empty support, improper Gaussian
   precision, a non-finite objective or residual, and inconsistent shapes
   return a loud divergent or invalid status. Float32 guards come from the
   dtype. K=8 and T>=100 run without regime-path enumeration or cubic work in
   T.
8. The authority states the conservative bound in D10.8 and does not reuse
   D3's optimistic collapsed-mixture ledger label or claim uncertain-drive
   equality with exact evidence.

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

**2026-08-22T21:09:30Z**

Closure handoff (2026-08-22). Commit c2827d0 exports the pure `cffg.structured_hybrid` authority without compiler or Engine dispatch. Public entry points are `categorical_coordinate`, `conditioned_categorical_coordinate`, `recurrence_auxiliary` (optimized or fixed xi), `gaussian_coordinate`, `expected_dynamics_log_potential`, `structured_hybrid_objective`, `dynamics_statistics`, `recurrence_statistics`, `structured_hybrid_initial`, and `structured_hybrid_sweep`. Pair rows are sources, columns are destinations; lag covariance and second moments are oriented current-by-previous. Numeric shape errors raise ValueError; improper/non-finite algebra returns shape-preserving NaNs/positive infinity and `SWEEP_DIVERGED`.

Independent evidence covers K2/K3 path enumeration, implicit last-stick axes, K3/d2 nonsymmetric dynamics with noncommuting precisions against one dense Gaussian solve, oriented lag moments and D10.2, conditioned D9 and K1 bypasses, explicit categorical-unary objective ownership, fixed-xi expanded sweep monotonicity, auxiliary-reoptimized collapsed boundary values, independent F_JJ >= F_exact(q) >= -log evidence integration, MNIW/PG statistics, literal-zero/empty/improper/nonfinite/float32 inputs, and K8/T100 polynomial storage. Focused authority plus hygiene/dashboard selection: 28 passed. Required unchanged-head dashboard suite: 1491 passed, 39 slow tests deselected, zero failed, four existing dependency warnings, 612.07 seconds; ~/agent-share/cffg-tests.html refreshed.

Reviewer corrections are incorporated in DERIVATIONS D10 and ARCHITECTURE: arbitrary u_t is scored; qX is exact only for fixed PG auxiliary; collapsed stationarity is claimed only at a joint fixed point; initial/emission/unary ledger ownership is explicit; initial dispatch eligibility is tied regime-independent x0/shared emission with equality-closure mapping; conditioned fixed drive is not a singular Gaussian and K1 bypasses D9.

cff-e10u handoff: derive StructuredHybridPlan only for that documented topology, dispatch these coordinates through synchronous/scheduled host/JIT paths, persist the installed singleton/pair/Gaussian/lag state, and give D10 one energy owner that suppresses the replaced initial, emission, D3, D9, and internal separator terms. Retain D3/D9 for every ineligible graph. cff-6rav remains the owner of learning installation and persistence identity. No fit, adapter, control, NASCAR data, calibration seed, or final seed ran.

**2026-08-24T14:05:40Z**

Integrated D10 core is published for review in draft PR #93: https://github.com/engimatic-systems/cffg/pull/93. The PR is chained on #92 and excludes scientific requalification, stability gating, calibration, and final evidence.
