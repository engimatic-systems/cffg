---
id: cff-toqa
status: open
deps: [cff-7kpx, cff-fc2y, cff-36x7, cff-qw7x]
links: []
created: 2026-09-25T03:46:25Z
type: feature
priority: 1
assignee: coopwilliams
parent: cff-xa6z
tags: [pcgym]
---
# Implement the evidence-backed nonlinear planning admissibility contract

## Background in Plain English

Nonlinear factors exist, but the PCGym planning experiment could only use its physical model after external linearization. The preceding investigation must identify whether the missing capability is initialization, scheduling or a supported local approximation; this ticket implements that specific contract, not an arbitrary noise floor.

Evidence comes from the PCGym report at `/Users/ravd/agent-share/PCGymCFFG/` and experiment code at `/Users/ravd/experiments/cffg-testing/pc-gym/00/cffg/`, reviewed against CFFG `fce60bd`. Copy only the minimal reproducible inputs into maintained fixtures; do not add external training artifacts or pickle dependencies.

## Outcomes / Acceptance Criteria

1. Before code changes, this ticket records the concrete contract and retained failing fixtures produced by the investigation.

2. The supported CSTR planning cases run through gaussian_nonlinear without caller-side linearization and satisfy finite-belief, adaptive-residual and free-energy certificates.

3. Unsupported cavities receive an explicit attributable failure rather than fabricated precision or silent success.

4. Any changed message or energy equation has a derivation and plain-English explanation in DERIVATIONS with executable oracle anchors.

5. Taylor/sigma-point paths, affine limits, precision-ratio grid, float32, zero/vacuous support and uncomfortable scale satisfy the agreed contract.

6. Public examples, architecture and limitations state the supported domain and remaining approximation boundaries.

## Methodological Suggestions

Choose implementation only after the prerequisite establishes the cause. Architecture: New factor or rule family and Runtime numerical or status invariant. Pairs: host/JIT, flooding/scheduled, single/vmap required; current/legacy decoding N/A unless serialized approximation identity changes, in which case test explicit compatibility/rejection.

These are suggestions; another method is acceptable if it satisfies the outcomes. Work branches continue from the preceding validated PR in the PCGym chain, whose initial base is `cff-ucvx-readme-capability-map` (PR #161). Update the intended base before branching. Code changes require focused tests and one passing final short suite; documentation-only work validates its changed artifacts.

## Future work

- cff-wt1c — Validate the PCGym follow-up chain with matched control and runtime comparisons.

## Definitions

Admissibility contract: the inputs on which an update is mathematically defined. Adaptive residual: mismatch measured for a model whose messages depend on current beliefs. Oracle: an independent reference calculation.

Shared terms: CFFG means constrained Forney-style factor graph; JAX is the numerical array/compiler library; dtype is a numeric storage type; free energy is the model's inference objective; host/JIT compares Python-side and compiled execution; single/vmap compares one instance with vectorized instances; N/A means not applicable.

## Notes

**2026-09-25T10:50:52Z**

Investigation handoff (cff-7kpx, implementation not started): distinguish startup readiness from a substitute Gaussian. Ready means a proper adaptive input cavity. Startup-deferred means finite symmetric PSD rank-deficient input with eta in Range(precision), no established nonneutral site, valid noise/parameters and finite symmetric output potential. It represents flat uninformed directions, not covariance=pinv(precision). A deferred factor must not evaluate its physical function, publish moments/finite energy, or certify convergence. Preserve neutral startup messages and explicit per-factor readiness while ordinary/ready-factor propagation continues under a bounded budget. Negative input curvature, incompatible nullspace information, invalid noise/parameters and admitted function/Jacobian/sigma-moment failures remain immediate attributable failures. Existing allowed indefinite OUTPUT potentials stay allowed when the full tilt is proper. An established nonneutral site must not be silently replaced by startup deferral after a later failure. Chart-wide proper edge beliefs are insufficient: every adaptive input must be ready before trust-region solving/certification. No precision floor, ridge, arbitrary missing-direction mean or hidden retry.

**2026-09-25T10:50:52Z**

Required independent authorities before implementation: (1) affine chain with proper prior/process noise and partial observations, both approximation methods, full means/covariances/evidence against dense NumPy conditioning, plus explicit readiness sequence; (2) permanently uninformed direction exhausts bounded startup without finite invented moments or convergence; (3) negative input precision and PSD precision with eta outside its range never defer; (4) proper cavity with a function singular at a required sigma point still fails. Retain same contracts for float32 and scale; preregister normalized residual tolerance=max(requested,64*eps), affine comparisons bounded by512*eps float32 and1e-8 float64. CSTR supported cases must separately show finite representable physical function/Jacobian/second moments. Wide-prior RK4 blow-up is NOT repaired by deferral. Sources: failure_provenance algebra/code review and cff-7kpx retained mechanism tests; high confidence. Split implementation further if needed to keep admission facts separate from runtime initialization.

**2026-09-25T11:22:42Z**

Implementation-boundary research: use a separate startup sweep controlled once by OuterTrustRegionStrategy, leaving strict local_result and the differentiated ordinary residual map unchanged. Neutral-site ownership is at GaussGroupPlan.scatter across BOTH roles; gather contains only incoming cavities. Initial eligibility plus run-local admission latches prevent re-deferring established factors. Neutral-at-entry is an explicit convention, not evidence of never-run state; State has no history flag. schedule._nodes_and_links excludes iterative Gaussian factors from directed nodes, so flood and mixed both reach execution.synchronous._gauss_sweep; flat/registerized scans remain unchanged exact-neighbor transports. A dedicated startup result can carry member readiness without broadening every normal Sweep/Propagation ABI. JIT execution must genuinely skip deferred callbacks: vmapped cond alone can evaluate inactive arithmetic. Adaptive Colony already delegates each member to Engine.run. Preserve zero budget, account and bound startup work, prioritize genuine failures, and never reopen startup during support/hard-phase restarts. Sources: failure_provenance bounded owner review of nonlinear_kernels, synchronous execution, schedule_program, outer_strategy and fixed_point; high confidence.

**2026-09-25T11:24:22Z**

Split before implementation into three bounded tickets using the intelligible-ticket skill. Source of truth is tk dependencies; snapshot built from tk dep tree --full, tk ready and tk blocked. Legend: prerequisite --> dependent.

[x] cff-7kpx investigation
  --> [ready] cff-fc2y input-readiness classification
        --> [blocked] cff-36x7 bounded startup execution
              --> [blocked] cff-qw7x reactor integration
                    --> [blocked] cff-toqa --> cff-wt1c --> cff-xa6z

cff-toqa directly requires all three children. Parallel ready roots include cff-qu2j and cff-ees0. cff-c5qf is a separate scope decision, not authorization to revive the abandoned partial-goal EC solver. tk dep cycle passes.
