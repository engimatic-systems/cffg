---
id: opt-ewtf
status: in_progress
deps: [cff-qu2j]
links: [cff-wt1c]
created: 2026-10-02T03:40:20Z
type: task
priority: 1
assignee: coopwilliams
parent: cff-xa6z
tags: [pcgym, performance]
---
# Make generic adaptive nonlinear inference practically usable without weakening certificates

## Background in Plain English

Native adaptive nonlinear CSTR planning is now convergent but unusably slow. The retained validation reports about112-125 seconds per compilation-warm plan, with55-58 reported iterations. Each iteration can contain multiple graph maps, reconstructions, energies and derivative/Krylov work. The host-linearized controller's7ms is context only: its numerical model differs.

Cooper requests implemented and validated generic runtime improvements, not profiling alone, targeting100-1600x faster native planning. Keep the original experiment unchanged. Use independent worktrees/environments and land this work in one PR continuing the current chain.

## Outcomes / Acceptance Criteria

1. A self-contained performance fixture preserves the actual validation's P4/H15, four-dimensional states,19 adaptive nonlinear transitions,4-substep RK4 physics, true constants, declared Q/R/prior, preview goals, auto schedule and1e-6 tolerance. Any synthetic observations and selected budget are labeled, not invented historical invocation facts.

2. Baseline and optimized runs use separate worktrees and environments with identical dependency versions and workload. Report synchronized cold, compilation-warm repeated-base and changed-observation times, actual compilation/cache behavior, iterations/map evaluations and full objective/certificate results.

3. Implement generic optimizations and validate the100x minimum compilation-warm target, with1600x as the stretch target. Do not claim speedup from a shorter window, hidden warm-start advantage, excluded later recompilation, frozen approximations, missing objective terms or weaker stopping.

4. Returned states pass the original fixed-chart normalized residual, full current/successor free-energy, finite/proper belief, exact-support and inner-status conditions; compare moments/actions/objective against matched baseline results.

5. Chart metrics can change between solves without unnecessary compilation but remain fixed throughout each solve. Data, masks, parameters, base and hard values remain current on reused kernels.

6. Any inexpensive proposal refreshes adaptive physical approximations at every map, consumes the existing budget and passes the full certificate; otherwise robust trust inference remains authoritative. Startup sigma-point failure remains explicit and separate.

7. Validate float32/float64, zero/improper inputs, sparse support, changed observations, nonroot/fallback behavior and larger generic models. Run focused checks and the repository's final gate once after final changes.

8. Publish one PR with atomic commits, completed discovery tickets if any, concise measured tables and limitations. If achieving the target requires a correctness, workload or scientific compromise, stop and discuss options with Cooper.

## Methodological Suggestions

Optional: batch/compile chart conversion and complete evaluation; make numerical chart metrics dynamic kernel context; reuse reconstruction and same-point results; reuse linearized derivatives and candidate-family Jacobian products; consider fully adaptive certificate-gated Picard/Anderson proposals with trust fallback. Another correct method is acceptable.

Architecture change paths: Runtime numerical or status invariant; Factor rule or energy only if numerical formulas change. Before implementation inventory host/JIT, flooding/scheduled, Engine/Colony and training/shared fixed-point surfaces. Checkpoint current/legacy N/A unless schema changes.

## Future work

- cff-wt1c — Incorporate the validated native-runtime improvement into matched PCGym comparisons.
- cff-c5qf — Partial-goal hybrid scope remains a separate explicit decision.
- cff-d63w — Existing Gaussian accuracy work remains separate unless a necessary correction is completed and recorded in this PR.

## Definitions

CSTR: continuous stirred-tank reactor. RK4: fourth-order Runge-Kutta integration. Adaptive approximation: locally rebuilding a nonlinear Gaussian factor from current beliefs. Chart: dimensionless message coordinates with a reference metric fixed for one solve. Krylov solve: iterative linear algebra used inside trust steps. Certificate: the existing consistency/objective checks required for convergence. Picard/Anderson: fixed-point proposals and history-based acceleration, not frozen physics. Compilation-warm: kernels have already been compiled; messages may still restart from the original base. JIT: just-in-time compilation. Q/R: process/observation covariances. Exact support: literal allowed categorical states.


## Notes

**2026-10-02T03:41:03Z**

Pre-implementation architecture: Runtime numerical or status invariant; shared fixed_point/training and chart/kernel ownership also affected, numerical factor formulas stay unchanged. Surfaces: host/JIT conversion and complete evaluation; flooding/colored/mixed reconstruct+iter_step; Engine and Colony adaptive per-instance route; training RootProgram; discrete aliases/exact support and hardening; current/legacy serialization N/A (derived kernels only). Hostile contract: zero/improper Gaussian inputs handled by unchanged budgeted startup, no fast path before READY; sigma failures remain loud; sparse categorical support remains exact with invalidation; dtype-derived thresholds in f32/f64; 10x charts/long generic models covered. User explicitly requires pause before any scientific/correctness compromise. Separate worktrees /Users/ravd/projects/cffg-adaptive-runtime.fgpwdF/{baseline,optimized}, both uv sync --frozen Python3.12.13/JAX0.10.2/NumPy2.5.1; baseline92d6acc (code same180), optimizedbranchcff-adaptive-runtime, basePR184 updatedffonly. Do not alter original experiment.

**2026-10-02T03:41:03Z**

Source research: authoritative validation is external FEEDBACK.md62 dated2026-10-01/e ad32f1 plus validation/v2_cstr_nonlinear_chain.py and common/physics.py. Actual4 RK4substeps andQ=diag(1e-5,1e-5,4e-6,.25), P4/H15/P0diag(1,1,1e-4,1), historyDATA dimensions3/current2/goal2. Older repofixture uses8substeps/largerQ and cannot substitute. Closed CLI budget/observations/numbermeasuredplans are not retained; opengrid100/400explicit. Use labeledselectedbudget100/400 and sourceexactinitial[-1/3,.2], prospective syntheticchangedobservations, not claim historical59steps. Host7ms differsclipping/noise/historyandiscontextnot matchedoracle. Highconfidence source arrays; unknown originalclosedcommand. Agent research also identifies dynamic metrics needed before removingmetricbytesfromkernelkey, repeated reconstruction across map/energy/support, duplicate linearize/vjp, collinear JVPfamilies and needless Krylov on alreadycertified roots; proposedcheapadaptiveproposals must refresh approximations andpassfullcertificate.

**2026-10-02T04:00:09Z**

Chart implementation and independent tests: grouped pack/unpack retain exact sparse support, singleton gauges, non-diagonal precision transforms, EF aliases, primal/JVP behavior, float32/64, and 100x blocks. First-use JIT exposed a cached-tracer leak; metrics now cache only NumPy values and enter kernels as dynamic leaves. Mixed-dtype promotion matches the original active-block concatenate. Structural reuse binds fresh metrics; independent derivative test passes. Focused geometry/nonlinear/GFE run: 18 passed, 13 deselected. Research review confirms derivative transpose linearity; proposal acceptance will recheck raw mapped finiteness and both compiled and authoritative host energy/root certificates. Preliminary warm CSTR 0.103s vs matched 44.175s, not yet final validated claim.
