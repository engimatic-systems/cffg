---
id: cff-ffsg
status: closed
deps: [cff-ocxl]
links: []
created: 2026-09-14T01:11:24Z
type: feature
priority: 2
assignee: coopwilliams
parent: cff-zov5
tags: [rslds, inference, hybrid]
---
# Implement hybrid EC local kernels and auxiliary stationarity

## Background in Plain English — WHY

D12 provides executable NumPy derivation anchors, but production JAX has no declared hybrid tilt, quotient, adapter, or joint auxiliary residual. The source-conditioned statistics must come from the same local belief and objective.

## Outcomes / Acceptance Criteria — WHAT

1. JAX local kernels implement D12.1–8, including labeled entropy, complete Gaussian normalizers, conditional cross-moments, signed hybrid quotients, categorical/Gaussian adapters, and the xi=c² auxiliary residual.
2. Destination-selected affine dynamics uses the alternating scopes or the proven grouped elimination with equal messages, moments, and energy; unsupported channel conventions fail explicitly.
3. Independent tiny-reference tests cover both D11 mean twins, both variance assignments, local objective/gradient, all boundary message blocks, and the D12 temporal approximation falsifier. D9 fixed-drive and drive-independent local reductions remain covered.
4. Literal zeros remain exact; empty support, support changes, inactive auxiliaries, improper/vacuous cavities, invalid shapes, non-finite objectives/residuals, and positive-support underflow follow an explicit contract. Guards derive from dtype; signed messages are never clamped.
5. Host/JIT and single/vmap kernels agree in float32/float64. T=100, K=20, and d=10 inventory cases stay polynomial and use no dense trajectory covariance; slow globalization tests are opt-in when needed.
6. The Derived component coordinate, New factor or rule family, and Runtime numerical or status invariant inventories mark compiler/runtime scheduling/persistence N/A here. Numerical docstrings and D12 implementation anchors are updated; focused tests and one final short-suite gate pass.

## Methodological Suggestions — HOW

These suggestions are optional; another sound method meeting the outcomes is welcome.

Keep local mathematics independent of Engine or model selection. Use the existing D11 recurrence polynomial where the derivation proves identity, and compare every natural constant as well as precision.

## Future work

cff-f3jz binds these local kernels to cff-7sw5 plans and supplies the joint message/auxiliary iterative policy. This ticket does not change public engine selection or learning state.

## Definitions

- D12: the hybrid-regime-ec-v1 mathematical contract in docs/DERIVATIONS.md.
- EC: expectation consistency, matching declared moments between regional beliefs.
- Hybrid separator: one boundary carrying regime probabilities and regime-conditioned state moments.
- Adapter: an explicit equality region connecting a hybrid boundary to an ordinary categorical or Gaussian copy.
- JJ / xi: the Jaakkola–Jordan logistic quadratic bound and its squared auxiliary coordinate.
- FE / ledger: free energy and the single assembly of its owned energy/entropy terms.
- GFE: generalised free energy with P-substitution, excluded from D12.
- IR: serialized intermediate representation of model semantics.
- Arena / member_of: compiled array storage and the map from factors to its members.
- JIT / vmap: JAX compiled and batched execution surfaces.
- CFFGZ: the project bundle format containing template, data, and optional learning state.
- Gauge: redundant numerical coordinate that leaves a belief unchanged.
- Krylov: a matrix-free iterative linear solve using bounded stored vectors.
- D9 / D10: the current factor-local recurrence approximation and the retired whole-model candidate.
- Moment / covariance: an expected feature or product, and its centered second moment.
- Semantic identity: the versioned meaning of a declaration, executable, or persisted statistic.



## Notes

**2026-09-15T13:20:19Z**

Closed as not pursuing D12 on this PR142-based branch at Cooper's direction; see cff-ocxl for the deferral summary. The preserved PR143 branch contains implemented and qualified local JAX kernels, including auxiliary stationarity and numerical repairs. Later branches add the independent joint reference and solver. Those results remain valid for their recorded sources; no implementation is incorporated here and local qualification is not a source-recovery result.
