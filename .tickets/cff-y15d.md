---
id: cff-y15d
status: open
deps: [cff-j191, cff-ffsg]
links: []
created: 2026-09-14T04:40:24Z
type: task
priority: 2
assignee: coopwilliams
parent: cff-a3wz
tags: [rslds, inference, hybrid, numerical]
---
# Implement a bounded unbound hybrid message and auxiliary solver

## Background in Plain English — WHY

The local hybrid kernels expose equations and maximum error certificates, but they do not solve the complete message and auxiliary problem. This bounded ticket implements the reusable numerical solver after the independent tiny reference. The downstream cff-a3wz ticket owns the fixed moderate qualification runs and actual storage evidence.

## Outcomes / Acceptance Criteria — WHAT

1. An unbound JAX solver jointly solves D12.7 message and D12.8 auxiliary equations using the qualified local kernels. Pack only active floating message constants after removing the common gauge, linear coefficients, independent symmetric precision entries, and active xi. Origins and support inventory are fixed; statuses and retained numerical corrections are metadata or derived values, never extra unknowns.
2. Commit candidate fixtures/initializations, direction strategy, line-search or trust-region budgets, linear iteration/restart limits, memory limits, and reporting specification before candidate checkpoint solves. Inherit every numerical threshold and fixture from cff-a3wz and the independent reference specification. Record diagnosed strategy or budget revisions prospectively, with earlier failures retained; do not silently tune them away.
3. Signed residual blocks and a bounded globalized iteration include natural equations, label mass, unweighted conditional means/covariances, and xi closure so rare labels affect the merit. Linearization and trial scales are controlled consistently. A converged result passes every exact cff-a3wz maximum certificate, proper active beliefs, finite/stable FE, and primal/dual agreement recomputed at the returned coordinate. An averaged merit or a stationary nonzero residual cannot certify convergence.
4. Feasible initialization handles the fixed valid tiny cases, xi=0/1 and fixed message perturbations. Negative xi trials respect the nonnegative boundary; inactive xi stays zero. Improper trials may be rejected, but invalid accepted residuals/objectives are divergence. Converged, budget-exhausted, and diverged outcomes remain distinct and report rejected steps and failure reasons.
5. Tiny solved outputs agree with the independent complete reference under its frozen arithmetic/stopping budgets. Host/JIT and single/vmap comparisons cover the solver and all certificates. Before checkpoint solves, bounded complete-residual JVP and transpose-identity tests cover nonzero origins, the repaired covariance path, sparse support, and xi=0; do not rely only on selected free-energy gradients.
6. Differentiate accepted residual evaluation rather than the entire optimizer. Fixed-size iteration state and explicitly bounded directional vectors avoid retained optimization history, dense global Jacobians, dense trajectory covariances, and runtime path enumeration. Provide inspectable buffer shapes and a compiled-memory measurement interface for cff-a3wz. With full support the unknown count is N=2*(T-1)*(K*(1+d+d*(d+1)/2)-1)+(T-1)*K*(K-1); retained origins and corrections are additionally counted.
7. Before implementation, state literal-zero/empty/support-change behavior, tiny-positive underflow, signed/vacuous cavities, invalid input, float32, and larger shapes. Apply the Derived component coordinate and Runtime numerical or status invariant change paths to local host/JIT and single/vmap packing, evaluation, iteration, and certification. Compiler, schedules, lifecycle, persistence, public learning activation, and source-recovery fits are N/A.
8. Focused tests, independent review, in-place documentation/dashboard registration, and final short-suite validation pass. Hand off the reusable solver API and frozen executable qualification specification. Closing this ticket certifies only this bounded solver delivery and tiny comparisons; cff-a3wz must still pass all fixed moderate cases and actual storage measurements before compiler work.

## Methodological Suggestions — HOW

- Build a signed residual, not derivatives of the evaluator's scalar max certificates. Call auxiliary_statistics at the current xi and freeze linear-subproblem scales while keeping final dynamic certificate scales exact.
- A bounded matrix-free Gauss-Newton/Steihaug trust region with feasibility backtracking is one candidate. src/cffg/outer.py has a linearize/vjp pattern, but its dimension-default linear cap is not a fixed storage/runtime budget. Another sound strategy is welcome.
- Keep host support-inventory construction and raise_for_status outside transformed iteration. Rebuild inventory on support change. Linearize only accepted proper coordinates; do not differentiate invalid-trial rejection masks.

## Future work

cff-a3wz runs the fixed T=32/T=100 float32 cases, assesses initialization sensitivity and learning statistics, measures actual iteration buffers and compiled temporary memory at multiple horizons/caps, and owns the positive/negative qualification decision. Compiler integration and registered recovery stay blocked until that evidence passes.

## Definitions

- D12: the hybrid-regime-ec-v1 approximation in docs/DERIVATIONS.md.
- Unbound: local numerical code that does not select or execute a production compiled model.
- Gauge: a redundant common scalar message constant.
- Xi: squared logistic-bound auxiliary coordinate.
- Merit: a scalar summary used to choose trial directions, distinct from acceptance certificates.
- JVP: a Jacobian applied to a vector without storing the full Jacobian.
- Transpose identity: the equality between inner products using a Jacobian and its transpose.
- Gauss-Newton/Steihaug: a residual-based local quadratic solve with bounded conjugate-gradient trust-region steps.
- JIT/vmap: JAX compiled and batched execution.
- Primal/dual: two stationary expressions of the same free-energy objective.



## Notes

**2026-09-14T04:41:18Z**

Independent strategy research — Question: what residual and bounded solver layout fits the complete checkpoint? Answer: pack only active gauge-reduced floating message coefficients and xi, with fixed support/origins; use signed natural, label-mass, unweighted conditional mean/covariance and xi residuals. Maximum certificates are stopping tests, not differentiable solver residuals. Matrix-free Gauss-Newton with explicitly capped Steihaug/trust-region directions and feasibility backtracking is a candidate; alternative sound strategies remain allowed. Freeze linear/trial scales consistently, then recompute the exact dynamic acceptance scales. Status masks can introduce NaNs, so linearize only accepted proper coordinates and keep host support rebuilding outside transformed iteration. Fixed-size while-loop state avoids differentiating/retaining the optimizer history. Check complete JVP and transpose identity including xi=0, nonzero origins and the forthcoming repaired covariance path before checkpoint execution. The affine custom JVP uses ordinary tangent arithmetic: cancellation-sensitive directions are a test risk, not a demonstrated defect. Sources: solver_checkpoint_research, hybrid_separator evaluate/auxiliary_statistics, hybrid_natural Natural/Belief, hybrid_affine custom JVP, outer.py linearize/vjp/Steihaug pattern, D12.4-9/D7.5. Confidence high for packing/API constraints; medium for strategy before measured qualification. No checkpoint solves were run during this research.

**2026-09-14T06:36:21Z**

Committed local API handoff at ff41a02 (independent review and full short-suite still pending): Belief.covariance is the high word; optional covariance_correction is the retained low word, with None exactly zero. correction exposes a zero-filled array when needed; full_covariance is a once-rounded physical readout. Cancellation-sensitive residuals and contractions must preserve both words instead of reconstructing from full_covariance. These words are derived from the message coordinate, not additional optimization variables. Natural.arrays() still returns only constant/linear/precision, and fixed origins remain metadata. Normalize reconstructed status arrays to int32: a weak int64 Python-default status combined with a batched int32 flag reproduced a JAX backend segfault, now guarded in local paths.

hybrid_covariance.solve uses custom_linear_solve with compensated residual refinement and a componentwise backward-error certificate, without a unit-scale floor; unsupported unresolved solve arithmetic returns NONFINITE. The logdet uses a whitened log1p identity and the exact trace differential. Candidate complete-residual JVP/transpose checks must include this retained covariance path, xi=0 and fixed nonzero origins. Local tests passed201/201 in1828.42s; this is not complete-solver qualification.

Reproducible local compiled inventory is uv run python tests/test_hybrid_projection.py, which lowers dynamic ShapeDtypeStruct arguments and emits four memory analyses. At T100/K20/d10, f32, combined projections and quotient messages have argument72,497,200/output5,490,608/temporary258,775,144 bytes, zero aliases; retained projection corrections total1,600,000 bytes. Whole pytest-process RSS observed12.4GiB during the long focused command is a separate process-level observation, not a per-iteration allocation. The qualification must still measure the complete iteration, differentiation/linear vectors, compilation cost and available process memory at its fixed T32/T100 cases and multiple iteration caps; do not extrapolate or sum unrelated local temporary peaks as if they were that measurement. No moderate checkpoint or recovery solve has run.

**2026-09-14T08:48:12Z**

Qualified local API update: FFSG, B0S3 and EJ67 are now closed in PR #143 (https://github.com/engimatic-systems/cffg/pull/143). Final numerical source is 07b18fd; independent review found no remaining issues, and the combined short suite passed 2489 tests, 41 deselected, in 3182.41s through tools/dashboard.py --run-suite. Earlier notes marked review/full-suite pending and the 201-test focused result are historical, not the current gate.

Natural.arrays still exposes only message coefficients; fixed origins remain metadata and retained covariance low words remain derived. Cancellation-sensitive covariance/quotient arithmetic still requires both representable words. Ordinary physical second moments use a distinct rounded product so an unused subnormal remainder does not reject an otherwise finite belief; both constructors and readouts share overflow behavior. This preserves the mathematical contract and all thresholds.

The reproducible local command uv run python tests/test_hybrid_projection.py now lowers six dynamic-argument programs. At T=100, K=20, d=10, float32 data with x64 enabled, combined projections/quotients retain argument/output/temporary bytes 72,497,200 / 5,490,608 / 258,775,144; grouped normalization is 70,560,400 / 70,720,856 / 131,200,984; grouped physical second moments are 70,560,000 / 64,000,000 / 0, with zero aliases. Zero temporary allocation reflects fusion into outputs, not zero work. The ordinary rounded product has one floating word and integer significand limbs, not the strict product's floating remainder/matrix accumulator. Full six-program details and final diagnostics live in EJ67 and D12.6. These remain local program measurements, not complete iterative buffers or process memory. Complete-residual differentiation, candidate convergence and actual iteration storage still require this sequential checkpoint chain. No complete reference/candidate/moderate solve or registered recovery fit has run at this handoff.

**2026-09-14T08:51:59Z**

Independent storage research — Question: what must complete matrix-free iteration inventory beyond packed unknowns and directional vectors? Answer: distinguish N coordinate-space vectors from M signed-residual vectors. With L=T-1 and s=d*(d+1)/2, full-support N=2*L*(K*(1+d+s)-1)+L*K*(K-1). Retaining all label masses and independent covariance entries gives M=N+2*L*K*(1+d+s), hence N=58*L and M=106*L at K4,d2. Dropping one redundant label-mass equation per incidence subtracts2*L; freeze the chosen layout explicitly. The existing outer.py fixed-point candidate treats its residual as a coordinate step and assumes M=N; do not copy it into a rectangular residual solver.

Inventory fixed context/parameter arrays, active maps/support/origins and materialized versus broadcast representations; packed and unpacked accepted/trial coordinates; M residuals and frozen scales plus N coordinate scales; grouped beliefs/factors/cavities O(T*K*K*d*d); separators/projections/covariance correction words O(T*K*d*d); auxiliary targets/exposures O(T*K*K); centered learning arrays O(T*K*K*(d+1)^2) and any materialized physical views. Include retained derivative closures (local factors/solutions/probabilities/corrections/masks/scans), N-sized gradient/step/linear residual/direction, M-sized Jp, normal-product and boundary/Cauchy temporaries, trial/candidate banks, integer significand limbs, status and bounded diagnostics. Steihaug's existing loop carries three N-vectors plus counters/flags; ordinary Steihaug has no stored restart basis, so count any added basis or preconditioner separately. Custom JVPs mean forward integer temporaries are not automatically all retained in differentiation.

Installed JAX 0.10.2 source shows separate linearize and vjp calls build separate linearization/residual collections. This is NOT evidence of twice the physical memory: inputs/compiler work may be shared. A linear_transpose of the already-linearized JVP can reuse its nonlinear forward data for the same frozen-scale residual, preserving J-transpose-J, subject to complete JVP/transpose identity checks. Compare actual enclosing iterations, not isolated closures alone.

Reproducible future measurement: dynamic ShapeDtypeStruct coordinate/context/scale/vector arguments; logical leaves/dtypes/bytes for accepted evaluation, derivative closures, normal product, complete linear step, accepted/trial iteration and terminal certification/statistics. Lower/compile the actual enclosing iteration and record argument/output/temporary/alias bytes; do not sum unrelated temporary peaks. Compare required horizons and outer/linear caps; report process/compiler RSS separately. Returning closure leaves lengthens lifetimes and is a logical inventory aid, not a peak-iteration measurement. Source -> independent ffsg research report; hybrid_natural.py:72, hybrid_separator.py:221/377, hybrid_covariance.py:26, outer.py:465/542, installed jax/_src/api.py linearize/vjp/linear_transpose, tests/test_hybrid_projection.py dynamic inventory. Confidence high for shapes/API semantics and procedure; physical sharing and full iteration memory remain unmeasured. No solves, compilation, edits to the reference, budget selection or threshold changes occurred in this research.
