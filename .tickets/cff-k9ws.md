---
id: cff-k9ws
status: closed
deps: []
links: [cff-o7hs, cff-pksr, cff-lquu]
created: 2026-07-15T01:45:43Z
type: feature
priority: 2
assignee: coopwilliams
parent: cff-ahps
tags: [hybrid, factor-family, rslds]
---
# Recurrence-link factor: stick-breaking logistic transition (continuous drives discrete)

## Plain English

The engine will gain the missing fixed-weight rSLDS transition: the previous
continuous state can change the probability of the next discrete regime. A
source-state-specific stick-breaking link keeps the categorical transition
normalized, while a coordinate-optimal Pólya–Gamma/Jaakkola–Jordan bound
produces ordinary categorical messages and a Gaussian message to the drive.
The free-energy ledger will state and test the approximation direction.

## Design

Add `recurrence_link(out, in, drive)` as a hybrid factor with equal K-state
discrete edges, one d-dimensional Gaussian drive, and canonical parameters
`weights[K, K-1, d]` and `offsets[K, K-1]`. For previous regime i and stick l,
`psi_il(x) = weights_il @ x + offsets_il`; sequential Bernoulli sticks define
the normalized K-way transition. The source-state axis is required for a true
recurrent switching transition; the older `[K-1, d]` shorthand omitted it.

The exact Pólya–Gamma augmentation is the derivational basis. Runtime inference
collapses the auxiliary into its coordinate-optimal expectation
`tanh(xi/2)/(2 xi)`, where `xi^2 = E[psi^2]`; this is exactly the deterministic
Jaakkola–Jordan mean-field bound. One mixed-schedule sweep updates all factor
messages from the current edge marginals, and the outer CAVI loop makes the
categorical, Gaussian, and auxiliary coordinates self-consistent. The bounded
expected log transition is below the exact expectation, so the reported node
and global variational free energies are upper bounds at the same beliefs.

Implementation is a dedicated hybrid plan, not the generic nonlinear-Gaussian
path. Failure-prefix sums construct all K destination scores in O(K) per source
row; the Gaussian precision accumulation costs O(K^2 d^2) for K source states
without a destination-by-stick tensor or whole-graph Jacobian.

## Change-path handoff

Applicable rows: **New factor or rule family** and **Runtime numerical or
status invariant**.

- Authority: `docs/DERIVATIONS.md`, `src/cffg/core/ir.py`, the recurrence rule
  module, and the `AGENTS.md` numerical contract.
- Derived surfaces: hybrid edge-family validation; blob-shape inference and
  parameter packing; a recurrence arena plan and fingerprint; synchronous
  flooding and restricted mixed-schedule execution; Gaussian/discrete damping;
  the node/edge free-energy ledger; continuous warm-up classification in host
  and colony runs; visualization glyph; README, ARCHITECTURE, LIMITATIONS, and
  dashboard coverage.
- Duplicated pairs: host/JIT, flood/mixed, and sequential/`vmap` are affected
  and share one rule implementation. Exact two-pass and colored factor firing
  are N/A because the bound is an iterative coordinate structure. Max-product,
  P-substitution, learned recurrence weights, checkpoint migration, and generic
  `FactorKernel` dispatch are N/A; the compiler rejects unsupported modes and
  cff-4apw/cff-209l own the deferred surfaces.
- Focused tests: the new recurrence rule/engine oracle, `tests/test_ir.py`,
  `tests/test_schedule.py`, `tests/test_runtime_status.py`,
  `tests/test_colony.py`, `tests/test_viz_diagram.py`, and serialization tests.

## Hostile inputs

- Exact-zero states retain zero probability in edge marginals and contribute
  exactly zero to responsibilities, at-risk probabilities, and PG weights.
  Empty edge support is infeasible through the existing exact-support check;
  the finite logistic factor never invents a structural zero.
- An exactly vacuous Gaussian drive is a neutral transient during flooding.
  A final vacuous belief or a negative, nonsymmetric, or non-finite precision
  makes the bound non-finite and inference divergent; PSD outgoing likelihood
  precision is valid and is never regularized into a proper prior.
- Float32 evaluates log-sigmoid stably and uses the analytic small-xi series
  selected by a `finfo(dtype)` threshold. Roundoff-scale negative second
  moments may clamp to zero using a dtype/scale tolerance; material negativity
  is divergence. No absolute floor or ridge is introduced.
- Scale coverage uses K=32 and d=16 (at least 10x the scalar binary fixture),
  asserts finite convergence and bounded plan storage, and exercises prefix
  construction rather than a K-cubed destination tensor.

## Acceptance Criteria

1. `DERIVATIONS.md` gives the stick factorization, exact PG identity,
   coordinate-optimal JJ bound, all three messages, node energy, and proof that
   the reported variational free energy upper-bounds the exact value at fixed
   beliefs.
2. `recurrence_link` validates and serializes as a hybrid factor, compiles to
   segmented discrete/Gaussian slots, runs under flood and mixed schedules,
   and is rejected under unsupported beads and max-product.
3. Rule outputs match an independent NumPy implementation, the engine posterior
   stays within documented tolerance of dense scalar quadrature, and the
   ledger's bound direction is pinned against exact expected log likelihood.
4. Sparse support, transient vacuity, improper drive precision, float32, and
   K=32/d=16 satisfy the hostile-input contract; a `vmap` batch matches
   sequential execution without a whole-graph Jacobian.
5. Canonical documentation exposes construction and approximation semantics.
   Fixed-weight inference lands here; readouts and weight learning remain with
   cff-lquu and cff-4apw, and hybrid kernel migration remains with cff-209l.

## Notes

**2026-07-31T21:05:43Z**

Research question -> What PG factorization and deterministic messages make the source-specific stick-breaking recurrence executable, and which way does the ledger bound point? Answer -> at-risk b[j,l]=1[j>=l], success a[j,l]=1[j=l], and kappa=a-b/2 give the exact PG augmented term exp(kappa*psi-omega*psi^2/2) PG(omega|b,0)/2^b. Mean-field q(omega)=PG(b,xi), xi^2=E[psi^2], is exactly the JJ bound; its categorical messages are expected per-destination bound scores and its Gaussian natural parameters are responsibility-weighted kappa/expected-omega sums. The global mixed schedule supplies the CAVI alternation. Because E[log p] is lower-bounded, the reported VFE is an upper bound at fixed beliefs. Sources -> Polson, Scott & Windle 2013 arXiv:1205.0310; Linderman, Johnson & Adams 2015 NeurIPS stick-breaking construction; Linderman et al. 2017 PMLR rSLDS paper and supplement; Durante & Rigon 2019 arXiv:1711.06999 PG-MF/JJ equivalence. Confidence -> high on factorization, messages, and bound direction; medium-high on runtime transient-vacuity policy, which follows this repository numerical contract.

**2026-07-31T21:37:38Z**

Implementation complete in 3290109. Added the source-specific recurrence_link factor, dedicated dual-arena plan, PG/JJ categorical and Gaussian coordinates, mixed/flood execution, VFE ledger term, numerical status handling, serialization, visualization, canonical derivation, public construction docs, limitations, and dashboard ownership. The bounded expected log transition is below the exact expectation, so the recurrence VFE is an upper bound at held beliefs; dense scalar quadrature pins both posterior tolerance and global bound direction. Storage is O(P K^2 d), updates are O(P K^2 d^2), and no K^3 destination tensor is allocated. Validation: 10 recurrence tests passed; 124 cross-surface focused tests passed; 38 slow tests passed in 420.81s; final dashboard suite passed 1271 tests with 0 failures and 38 slow deselections in 336.62s, refreshing the shared dashboard. Deferred by design: weight learning cff-4apw, hybrid readouts cff-lquu, and generic kernel migration cff-209l.
