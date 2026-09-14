---
id: cff-ffsg
status: in_progress
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

**2026-09-14T02:37:52Z**

Preimplementation handoff: applicable ARCHITECTURE change-path rows are Derived component coordinate, New factor or rule family, and Runtime numerical or status invariant. Affected surfaces are local unbound JAX natural/belief normalization, grouped destination-selected channel tilt, moment projection and signed quotient, ordinary identity adapters, labeled objective, tied xi residual, pair-conditioned learning increments, and pure local/batched chain evaluation. Host/JIT and single/vmap pairs apply and must agree. Compiler discovery/grouping/slots/packing/resolution/assembly/Arena/member_of, synchronous/scheduled runtime execution, lifecycle/status consumers, production ledger dispatch, readout/learning state, serialization/checkpoints, estimator, and public exports are checked N/A: downstream tickets own those bindings. Static shape/dtype/channel errors raise before execution; numerical failures use explicit array statuses in host and JIT and a host raising adapter. Hostile acceptance: exact log-zero support produces exact zero weighted moments; empty support, changed quotient faces, positive-support underflow, malformed/non-finite coordinates, improper aggregate/tilted beliefs and non-finite energy/residual are failures, never repaired by a ridge. Signed/vacuous cavities are allowed if their tilt is proper. Inactive xi is fixed to zero with zero learning increments. Float32 and float64 use dimension/dtype-relative spectral and squared-logit guards. Inventory cases T=100, K=20 and d=10 stay polynomial and store only local covariance; no iteration policy or dense Jacobian/trajectory covariance is introduced. Ticket fits a bounded implementation context as two local math modules and focused reference/contract tests. Independent review and the single final full suite are sequenced by the supervisor after implementation.

**2026-09-14T02:56:31Z**

Implementation handoff for independent review: added unbound local JAX hybrid_natural.py and hybrid_separator.py (315 and 283 lines), with exact labeled Gaussian normalizers/entropy, signed quotients and explicit adapter regions, destination-selected grouped transitions, O(K²) stick prefixes, per-source/stick xi=c² stationarity and pair-conditioned learning increments. The pure stacked chain evaluator returns the same beliefs, primal/dual expressions, scaled conditional-moment residual and full quotient residual without an iteration policy. Numerical host/JIT failures are explicit bitmask statuses; raise_for_status is the host exception boundary. Invalid beliefs/objectives are NaN, positive-support underflow retains its finite log probability and fails, and inactive coordinates/learning remain exact zero. Added independent-reference and hostile/transformation/size tests and registered both new test files in dashboard EPIC_FILES. Updated ARCHITECTURE and D12 executable/numerical anchors. Final focused command: uv run pytest -q tests/test_hybrid_separator_kernels.py tests/test_hybrid_separator_numerics.py tests/test_hybrid_separator_derivation.py tests/test_regime_pair_derivation.py — 114 passed in 49.71s. Log: /tmp/cff-ffsg-focused-final.log. git diff --check and tk dep cycle passed. No mathematical contract, supported scope, approximation semantics or thresholds changed. Next: supervisor starts a fresh reviewer, returns feedback to this implementer, and performs the one final short-suite gate after final code changes before closure/PR. Full suite deliberately not launched here. cff-a3wz owns numerical joint-solver feasibility, actual iteration-storage measurement and initialization/globalization policy; these local T=100/K=20/d=10 inventories are not solver qualification.

**2026-09-14T03:04:15Z**

Independent review P1 resolved: large common natural constants previously erased label-dependent Gaussian integrals and the log-sum correction, allowing probabilities [1,1] with OK status. normalize now centers constants before adding Gaussian masses, normalizes centered scores, and restores the full log normalizer separately. Mechanism regressions cover equal and unequal Gaussian normalizers at f32 gauge 1e8 and f64 gauge 1e16, host/JIT parity, unit total probability, labeled entropy invariance and the log-normalizer gradient. Validation: uv run pytest -q tests/test_hybrid_separator_numerics.py tests/test_hybrid_separator_kernels.py — 59 passed in 49.53s; log /tmp/cff-ffsg-gauge.log. P2 centered auxiliary variance fix remains next; no full suite launched.

**2026-09-14T03:08:17Z**

Independent review P2 resolved for the xi residual: contracting raw augmented second moments lost positive conditional variance when a large affine mean canceled its offset (for example f32 mean100,var1e-4 and f64 mean1e8,var1). Auxiliary targets now use reverse weighted covariance merging followed by Var[psi]+E[psi]². The merge keeps within-pair and between-pair covariance and preserves O(K² d²) work/storage even when a stick excludes a distant destination. No K³ tensor or absolute guard was added. Mechanism regressions cover f32/f64, host/JIT, the target derivative with respect to weights and a distant excluded-label suffix. Validation: uv run pytest -q tests/test_hybrid_separator_numerics.py tests/test_hybrid_separator_kernels.py — 65 passed in 54.49s; log /tmp/cff-ffsg-centered-auxiliary.log. A related diagnostic found that reconstructing/differentiating recurrence energy from generic raw canonical coefficients can still cancel the same small variance; assessing a bounded stable energy helper before final handoff.
