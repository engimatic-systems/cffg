---
id: cff-36x7
status: open
deps: [cff-fc2y]
links: []
created: 2026-09-25T11:23:56Z
type: feature
priority: 1
assignee: coopwilliams
parent: cff-toqa
tags: [pcgym, nonlinear]
---
# Run bounded startup sweeps for neutral adaptive Gaussian factors

## Background in Plain English

The existing adaptive solver can propagate an exactly vacuous input but stops on partial input information. Investigation cff-7kpx shows that partial goals can arrive before the forward prior. A dedicated startup phase can wait for missing information without changing the strict nonlinear rule, provided it cannot conceal invalid models or replace established messages.

## Outcomes / Acceptance Criteria

1. Adaptive Engine runs have one explicit bounded startup phase that uses the prerequisite readiness classification. Zero iteration budget performs no extra work.
2. Eligibility comes from all outgoing message coordinates at each factor, not its incoming cavity. Only neutral-at-entry, not-yet-admitted factors may defer; admission is latched for the run.
3. Deferred factors do not evaluate their physical function, Jacobian or sigma-point calculation, publish finite moments/energy, or count as convergence. Existing nonneutral sites are never silently replaced with startup zeros.
4. Ready factors and ordinary neighbors advance while per-factor pending readiness remains available. Invalid noise, parameters, resolved invalid input/output information and failed admitted computations terminate immediately with originating factor diagnostics. The prerequisite classifier defines the floating-point resolution band; a partial classification permits only waiting, never a belief or convergence.
5. Startup work is included in the user iteration budget. Permanently unresolved inputs end truthfully with the pending factor identified. Normal trust-region solving requires both proper chart references and ready adaptive inputs.
6. Startup never reopens during later support restarts, hard phases or ordinary residual/gradient evaluation. Ordinary local_result and differentiated residual semantics stay strict.
7. Flood and mixed schedules, host/JIT execution and single/batched local members agree. Adaptive Colony instances retain independent readiness through their existing Engine delegation.
8. An affine chain with partial observations converges from fresh initialization and matches independent means, covariances and evidence. Negative precision, incompatible nullspace information, established-site regression and singular required sigma points remain failures.
9. Document the neutral-at-entry convention and unchanged checkpoint representation. Preserve non-adaptive behavior and public failure compatibility.

## Methodological Suggestions

Another sound design is acceptable. Architecture rows: Runtime numerical/status and New factor or rule family. Owner candidates: execution.synchronous for outgoing-site ownership, schedule_program for a dedicated startup callable/result, and OuterTrustRegionStrategy for a once-only run-local phase. Iterative Gaussian factors do not use directed flat/registerized scans; confirm these are N/A except as unchanged neighbor propagation. A vmapped conditional can evaluate inactive arithmetic: use a genuine execution guard. Prefer a separate readiness result over expanding every normal sweep ABI. Current/legacy checkpoint schemas are N/A unless the design adds persistent state. Focused tests and one final default suite are required.

## Future work

cff-qw7x establishes the finite-physics planning envelope and updates user-facing examples. cff-toqa remains the parent completion contract; cff-wt1c measures control/runtime outcomes. This does not repair unstable RK4 physics or partial-goal mixture collapse in cff-c5qf.

## Definitions

An adaptive factor changes its local Gaussian approximation with its input belief. A site is a factor-owned outgoing message. A latch records that a factor has already been admitted. Readiness means the input is normalized enough to evaluate the declared approximation, not that inference has converged. ABI is the shape/meaning of data exchanged between runtime components. Other numerical terms follow cff-fc2y.

## Notes

**2026-09-30T16:03:50Z**

Side research handoff: question — where can once-only startup defer without changing equations or hiding work? Answer — schedule startup should guard one-member Gaussian execution with scalar cond in lax.map/scan, because nonlinear_kernels.results uses vmap and can evaluate inactive branches. Freeze neutral g.scatter eligibility before propagation; keep run-local admitted latches. Control startup before hard phases in OuterTrustRegionStrategy, not solve_outer discovery/recursive restarts, and charge each composed sweep against n_iters. Require all adaptive cavities READY plus chart proper before ordinary solve. Output potentials may legitimately be indefinite; do not reuse PSD input readiness on output. Check finite/symmetric output, packed noise propriety and finite theta before deferral. Sources: execution/synchronous.py, nonlinear_kernels.py, nonlinear.py, outer_strategy.py and outer.py (startup_runtime_review). High confidence. These are method suggestions; validate exact accounting and observation-specific tilt handling in implementation.
