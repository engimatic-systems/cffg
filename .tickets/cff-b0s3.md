---
id: cff-b0s3
status: open
deps: [cff-ocxl]
links: [cff-ffsg]
created: 2026-09-14T03:21:20Z
type: bug
priority: 1
assignee: coopwilliams
parent: cff-zov5
tags: [rslds, inference, hybrid, numerical]
---
# Stabilize hybrid tilted beliefs under coordinate translation

## Background in Plain English — WHY

Independent review of cff-ffsg found and repaired two numerical mechanisms: common constants could corrupt normalization, and raw second moments could erase an auxiliary target. Follow-up diagnosis then found a separate error in the tilted belief itself. In float32, a source Gaussian with mean 100 and variance 0.0001 and a recurrence offset of -100 returns a pair-probability error of about 0.241 while reporting valid numerical status, compared with the identical centered problem.

The original caller also expands a large Gaussian constant before passing it to the local kernel. Some common-normalizer information is already rounded away there. This ticket must distinguish that irreversible input loss from cancellation introduced inside integration. A stable energy helper alone cannot repair incorrect tilted probabilities.

This newly discovered prerequisite keeps cff-ffsg open. A fresh implementer will complete the bounded local representation repair before the joint solver checkpoint cff-a3wz begins. It changes the numerical representation of the same D12 approximation, not its equations or qualification thresholds.

## Outcomes / Acceptance Criteria — WHAT

1. Local hybrid quadratics retain enough centering and normalization information to evaluate parameter-aware Gaussian sites, grouped transitions, projected beliefs, signed quotients, and ordinary adapters without expanding large canceling constants prematurely. Original Gaussian mean/covariance/log-mass and transition parameters have explicit construction paths.
2. Any anchored representation states its exact potential, coordinate translation, multiplication, projection, quotient, and gauge conventions. Centers are derived from retained sites or a declared initialization and frozen during a solve, not extra unconstrained stationarity unknowns. Internal beliefs retain mean displacements without adding then subtracting a large origin. Physical messages, moments, labeled entropy, and D12 objective are invariant under a legal change of center within stated arithmetic error.
3. The complete repeated local path is stable: construct a tilt, project it, divide by the cavity, reuse the resulting messages in another evaluation, and check the returned beliefs and objective. Fixing only the first tilt is insufficient. Signed/vacuous messages, different conditional means, and full label-dependent Gaussian normalizers remain valid.
4. The checked-in tests/hybrid_centering_reproduction.py defect is resolved without relaxing its label-probability budget, 512*4*eps(dtype), or changing the model. Full-normalizer checks use retained physical inputs so they do not demand recovery of bits already lost by the caller. Already-expanded Natural input has an explicit mathematical and numerical contract; no hidden reinterpretation supplies unavailable original constants.
5. Independent tiny references compare mean=100/variance=0.0001 and mean=10000/variance=1 in float32, mean=100000000/variance=1 in float64, and at least one two-dimensional case with off-diagonal covariance and different regime means. Compare probabilities, conditional covariance, centered means, all message blocks in a common basis, full normalizers, objective, xi residual, and learning statistics. Any mean-output rounding allowance is stated in advance from dtype resolution at the translated coordinate; label and existing qualification tolerances remain unchanged.
6. Same-coordinate free-energy and xi derivatives preserve the stable cff-ffsg energy repair. Independent finite differences or analytic identities check differentiation through retained centering, including xi=0 and sparse support. The known temporal EC approximation falsifier remains unchanged and distinct from numerical coordinate invariance.
7. Exact log zeros, empty support, support changes, inactive xi, tiny-positive underflow, improper aggregates/tilts, invalid shape/dtype, and active non-finite values keep explicit behavior. No precision clamp or absolute numerical floor hides failure. New centers and constants cannot turn an invalid state into success.
8. Host/JIT and single/vmap agree. Local storage remains O(T*K*K*d*d) for grouped beliefs and O(T*K*d*d) for messages plus explicitly counted center arrays; there is no dense trajectory covariance, path enumeration, or extra global Jacobian. Existing T=100, K=20, and d=10 local inventories stay polynomial. cff-a3wz still measures actual iterative storage separately.
9. Before implementation, the handoff applies Derived component coordinate, New factor or rule family, and Runtime numerical or status invariant, listing affected local duplicates. Production compiler, scheduling, model selection, learning persistence, and registered recovery fits remain N/A. If this repair needs a mathematical or supported-scope revision rather than algebraically identical numerical representation, stop with evidence for Cooper.
10. The executable reproduction becomes an asserted regression used by pytest, with no hidden failing diagnostic or weakened pass interpretation. Numerical docstrings, D12/ARCHITECTURE anchors, dashboard registration, focused tests, and the final short-suite gate agree. One passing short suite at the combined proposed PR head may satisfy this ticket and cff-ffsg; no second unchanged run is required.

## Methodological Suggestions — HOW

These suggestions are optional; another sound method meeting the outcomes is welcome.

- Consider an anchored potential a_i + eta_i^T*(x-c_i) - (x-c_i)^T*J_i*(x-c_i)/2, with explicitly retained centers and stable parameter-aware assembly. A signed sum of centered terms is another possible representation if its storage remains bounded.
- Read the exact reproduction and cff-ffsg review notes first. Separate the caller's lost common normalizer from the subsequent erroneous relative label weights.
- Reuse centered covariance merging and TransitionParameters from the bounded cff-ffsg repair. Do not replace them with raw augmented-second-moment contraction.
- Record independent diagnosis as question, answer, source, confidence in this ticket; canonical documentation stays in place.

## Future work

cff-ffsg closes its local-kernel delivery only after this repair and independent review pass. cff-a3wz then qualifies the complete message/auxiliary solver on its fixed tiny and moderate cases. Serialized declarations and compiler/production integration remain cff-k6g5, cff-7sw5, cff-idhk, cff-brcf, and cff-f3jz; learning and composition remain downstream. cff-x0if alone owns registered source-recovery fits and unchanged thresholds.

## Definitions

- D12: the hybrid-regime-ec-v1 approximation in docs/DERIVATIONS.md.
- Tilt: the local normalized product of a factor and incoming cavity messages.
- Cavity: the incoming product excluding the message being updated.
- Anchored quadratic: a quadratic log potential expressed around a retained numerical center.
- Natural coefficients: a constant, linear coefficient, and precision matrix defining a quadratic log potential.
- Gauge: a redundant common constant that does not change a normalized belief.
- Proper belief: a normalizable distribution with valid covariance.
- Xi: the squared logistic-bound auxiliary c*c.
- EC: expectation consistency, matching declared neighboring moments.
- JIT/vmap: JAX compiled and batched execution.
- Epsilon: the dtype machine resolution near one.


## Notes

**2026-09-14T03:23:28Z**

Independent diagnosis — Question: can locally centered construction alone repair the false valid translated tilt, or must centering survive repeated messages? Answer: retain fixed separator-label anchors c_ti throughout the complete local path. Represent log potential as a+h·(x-c)-(x-c)ᵀJ(x-c)/2. From retained log mass, mean and covariance, use r=mean-c, J=V^-1, h=Jr, a=logmass-(d*log(2*pi)+logdet(V)+rᵀJr)/2. Translate physical recurrence offsets to o+wᵀc_left and channel bias to A_j*c_left_i+b_j-c_right_j before expansion. Apparent pair dependence of translated channel bias is purely coordinates, not a change from destination-selected dynamics. Same-anchor products/quotients add/subtract coefficients; only a common scalar is gauge. For rebasing delta=c_new-c_old, h_new=h-J*delta and a_new=a+h·delta-deltaᵀJdelta/2. Keep solved displacements internally through marginalization, covariance merging and both adapters; world-mean round trips lose small updates. Anchored sufficient statistics are an invertible fixed transform, leaving the D12 stationarity zero set unchanged. Anchors add O(T*K*d) storage and are not solver unknowns.

Source: independent centered_tilt_diagnosis algebra and JAX/JIT/vmap demonstrator, D12.2–6, tests/hybrid_centering_reproduction.py. The existing artifact reproduces pair gap 0.2410068959, covariance gap zero, status OK. The independent retained-origin calculation preserves probabilities for origins 0/100/10000 including unequal masses, with f32 log-normalizer error about 1.2e-7. Confidence: high for mechanism and representation algebra; this does not qualify a production or joint solver.

Input limitation: at the example scale, float32 expanded constants have spacing about four. Different intended label masses can become identical caller-supplied arrays; the equal-label artifact also loses a common normalizer before entry. Raw inputs retain their stored-coefficient meaning, but cannot recover unavailable original label scales/evidence. Retained Gaussian construction is required for full intended-normalizer checks. Freeze any world-mean comparison allowance at 4*eps*(1+abs(origin)) plus the existing local arithmetic tolerance before execution; internal displacement and probability checks retain the existing local tolerance.

**2026-09-14T03:23:29Z**

Dependency refinement from tk dep tree --full cff-a3wz, tk ready, and tk blocked. Legend: prerequisite --> dependent; tk is authoritative.

[x] cff-ocxl
  +--> [ ] cff-b0s3 (ready numerical repair; fresh implementer next)
  |       --> [~] cff-ffsg (local completion awaits repair)
  |              --> [ ] cff-a3wz (complete solver checkpoint)
  +--> [ ] cff-k6g5 (independent ready declaration root)
        a3wz + k6g5 --> [ ] cff-7sw5 (compiler)
                            --> [ ] cff-idhk --> [ ] cff-brcf --> [ ] cff-f3jz
                                  --> downstream learning/composition/recovery cff-x0if
                                      --> [ ] cff-zov5

The existing FFSG implementation is available in committed branch ancestry through 38e4fd4; FFSG remains a completion gate, not an additional prerequisite of its own numerical repair. Compiler integration and registered recovery remain blocked. No dependency cycles; git diff --check validates this ticket-only refinement.

**2026-09-14T03:24:21Z**

Predecessor review is complete at 38e4fd4: original gauge normalization, centered auxiliary targets and parameter-aware actual energy were independently cleared with no additional actionable findings. Reviewer checked K=3,d=2 expectations and gradients across all six physical parameter blocks in dense/sparse f32/f64; maximum discrepancies about 1.5e-8 and 5.6e-17. The known tilt defect remains this ticket. Independent no-production-import reference is /tmp/cff-b0s3-reference.py; uv run python /tmp/cff-b0s3-reference.py passed its raw-input nonidentifiability, anchored JIT/vmap probability, and analytic local-normalizer assertions. Preserve the useful independent calculation in an appropriate executable test anchor, not a markdown report.
