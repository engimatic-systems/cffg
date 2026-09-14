---
id: cff-8ois
status: in_progress
deps: [cff-ej67, cff-j191]
links: []
created: 2026-09-14T10:54:57Z
type: bug
priority: 2
assignee: coopwilliams
parent: cff-a3wz
tags: [rslds, inference, hybrid, numerical]
---
# Preserve local outgoing precision accuracy under strong Gaussian cavities

## Background in Plain English — WHY

A held-coordinate check before the joint solver exposed an inaccurate outgoing Gaussian precision. The source precision is 10000 and the desired outgoing precision is about 0.0126, so small errors in normalization can grow during subtraction. An independent 80-digit decimal calculation gives 0.0125764053902527907; native float32 returns 0.012979879975318909. This misses the existing arithmetic allowance by 1.6321 times. Promoted float64 misses its stricter allowance by 5.0698 times. The derivative agrees with independent mathematics, but this does not qualify the inaccurate values.

This ticket localizes and repairs that arithmetic before cff-y15d runs any candidate optimizer. It preserves D12's equations and approximation, the original failed evidence, and every qualification threshold.

## Outcomes / Acceptance Criteria — WHAT

1. Reproduce and identify the first loss of required accuracy using only the qualified local kernels, exact stored inputs, and independent cancellation-free algebra or high-precision arithmetic. Distinguish errors introduced by canonical assembly, Gaussian normalization, projection, and quotient formation.
2. At the original held input and its already declared perturbation grid, source outgoing precision passes the unchanged same-coordinate allowance atol=rtol=512*4*eps(dtype) in native float32 and float64. Native float32 is also exercised with JAX's x64 setting disabled. Independent references use exact stored inputs; rounded intermediate covariance is not the physical-input authority.
3. Repair the general numerical mechanism, preserving signed/vacuous incoming messages, proper aggregate and tilted beliefs, fixed origins and support, int32 status flags, and the meaning of retained numerical corrections. Unresolved arithmetic must fail explicitly rather than certify inaccurate success.
4. Value and derivative regressions cover the affected local host/JIT and single/vmap paths. Derivatives are compared with analytic or independently promoted exact-input references under unchanged derivative tolerances; the failed native finite-difference stencil remains documented separately from native value acceptance.
5. Sparse or empty support, tiny-positive underflow, improper aggregates, float32, and uncomfortable scale are checked before implementation and covered where affected. At least one larger local K/d or input-scale case exercises the diagnosed mechanism without launching a complete candidate solve.
6. Free energy and learning statistics remain finite and satisfy the inherited same-coordinate budgets on affected proper cases. Any change in retained words or local temporary storage is measured and documented for cff-a3wz; derived corrections never become solver unknowns.
7. The architecture handoff applies the Derived component coordinate and Runtime numerical or status invariant change paths. Local assembly, normalization, projection, quotient, FE/statistic readers and differentiation are checked; host/JIT and single/vmap are explicit pairs. Production compiler, scheduled execution, lifecycle, serialization and registered recovery are N/A.
8. Focused regression tests, independent review after implementation pauses, canonical documentation updates, dashboard registration for new test files, and one final short-suite gate pass. The ticket is closed with an atomic commit and a concrete cff-y15d handoff.

## Methodological Suggestions — HOW

- Inspect the exact-input exports and Decimal80 diagnostic cited in the ticket note, then minimize the reproduction to hybrid_natural and hybrid_separator APIs. No dependency on uncommitted hybrid_joint drafts is needed.
- Preserve information at its first lossy operation. A compensated inverse or precision representation may be appropriate, but choose the simplest mathematically faithful general repair after localization.
- Keep mathematical derivatives explicit for any numerical representation changes. Existing hybrid_covariance and hybrid_affine helpers may help; do not assume a once-rounded inverse followed by subtraction meets the final outgoing precision budget.
- Freeze any additional bounded diagnostic input grid before executing it. Ordinary implementation fixes and diagnostics are authorized; scope or mathematical-contract changes return to the supervisor.

## Future work

cff-y15d resumes complete-residual derivative checks and tiny candidate solver runs after this repair. cff-a3wz retains all T32/T100 native float32 convergence, initialization, FE, learning and actual iteration-storage requirements. Compiler integration and cff-x0if registered recovery remain downstream and must not run here.

## Definitions

- D12: the hybrid-regime-ec-v1 approximation specified in docs/DERIVATIONS.md.
- Held coordinate: fixed messages and auxiliary values evaluated without optimizer updates.
- Precision: the inverse Gaussian covariance.
- Canonical assembly: combining Gaussian factors using constants, linear coefficients and precision matrices.
- Projection and quotient: matching local belief moments, then removing an incoming factor to form an outgoing message.
- Retained word: an additional floating array holding rounding information, derived from the mathematical coordinate.
- JVP: a derivative applied to a vector without constructing its full matrix.
- JIT/vmap: JAX compiled and batched execution.
- FE: free energy. eps(dtype) is machine epsilon of the executed floating-point type.
- Atol/rtol: absolute and relative numerical comparison allowances.



## Notes

**2026-09-14T10:55:50Z**

Independent diagnosis — Question: does the complete float32 residual derivative fail, or are the values inaccurate? Answer: Decimal80 gives row-7 derivative -0.643678176516749; native float32 JVP differs by 1.074e-7, promoted float64 by about 1e-16. Independent complete-residual finite differences at the frozen float64 baseline step agree within 4.855e-5. The original native float32 discrepancy consists of +0.00079045 finite-stencil truncation and -0.13237558 native value error. Source outgoing precision alone misses among the exported local-output groups: base true value 0.0125764053902527907, float32 error 4.03474585e-4 (1.6321 times budget), promoted float64 error 2.33446e-12 (5.0698 times budget). Generic NumPy inverse-then-subtract also loses precision here; Decimal cancellation-free algebra is the authority for this field. Existing mild J191 solves are not invalidated by that stronger-input reference limitation.

Sources: exact exports /tmp/cff-y15d-held-inputs.json and /tmp/cff-y15d-held-values.json; reproducers /tmp/cff-y15d-export-held.py and /tmp/cff-y15d-export-values.py; independent /tmp/cff-y15d-independent-derivative.py/.json/.log, /tmp/cff-y15d-independent-values.py/.json/.log and /tmp/cff-y15d-independent-stencil-decomposition.json. Confidence high for values and derivative; exact earliest production loss awaits attribution. Preserve a compact committed reproduction/fixture and failed evidence instead of relying on temporary paths.

Authorized existing grid: original stored float32 fixture test_hybrid_affine.case(float32,10000,1e-4,[6.7],-67000), fixed origins, xi=0, right identity-precision message. Perturb the same exported packed x/v at baseline step times 1,2,4,8,16,32; repeat in native float32 and exact-input promoted float64. No candidate optimizer, moderate solve or registered fit has run. Y15D worker is fully paused with its drafts preserved in /Users/ravd/projects/cffg-d12-chain at committed boundary 85930ec. This isolated repair worktree starts from qualified PR144 head9c55579 and needs no candidate draft imports.

Dependency refinement from tk dep tree --full, tk ready and tk blocked; tk dep cycle passes.
Legend: prerequisite --> dependent (intermediate implementation tickets abbreviated below)
[x] EJ67 + [x] J191
          |
          +--> [ ] 8OIS (ready local arithmetic repair)
                    |
                    +--> [ ] Y15D (blocked candidate solver)
                              |
                              +--> [ ] A3WZ (complete numerical/storage checkpoint)
                                        |
[ ] K6G5 (independent ready declarations)-+--> [ ] 7SW5 --> ... --> F3JZ
                                                                   |
                                                                   +--> ... --> X0IF --> ZOV5
TK records remain authoritative. K6G5 is deliberately deferred until the early numerical checkpoint completes.

**2026-09-14T10:57:12Z**

Independent stage attribution — Question: where does the strong-cavity precision first lose required information? Answer: the exported normalized regional source covariance already differs from physical-input truth by -4.01487298e-12 in float32 and +2.32951971e-20 in float64, before EJ67 projection. Exact analytical projection/quotient of the saved regional probabilities, means and covariances reproduces the native bad outgoing precision within3.61e-7f32 /8.36e-16f64, well below the final output allowances. Replacing only regional covariance with exact physical-input covariance reduces output error to1.63e-6f32 /5.78e-15f64. Later projection/quotient is not the dominant loss. Source inspection finds plain Natural coefficient addition, ordinary Cholesky/cho_solve in gaussian_blocks, and single-word normalize covariance; _mix retains low words only afterward. Inference, not directly exported evidence: canonical precision rounding contributes -1.56537e-12 of float32 covariance error and subsequent inversion -2.44950e-12. The unexported pre-normalization matrix prevents treating that split as measured. Sources /tmp/cff-y15d-stage-attribution.py/.json/.log and inspected hybrid_natural.py. Confidence high for pre-projection loss, medium for its assembly/inversion split. No repository edits, new JAX evaluations, or optimizer runs during independent attribution.
