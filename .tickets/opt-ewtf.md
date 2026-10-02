---
id: opt-ewtf
status: in_progress
deps: [cff-qu2j, opt-un4a]
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

Native adaptive nonlinear CSTR planning is now convergent but unusably slow. The retained validation reports about 112–125 seconds per supposedly compilation-warm plan, with 55–58 reported iterations. Each iteration can contain multiple graph maps, reconstructions, energies and derivative/Krylov work. The host-linearized controller's 7 ms is context only: its numerical model differs. Changed observations must be checked for actual recompilation rather than assumed warm.

Cooper requests implemented and validated generic runtime improvements, not profiling alone, targeting 100–1600× faster native planning. Keep the original experiment unchanged. Use independent worktrees/environments and land this work in one PR continuing the current chain.

## Outcomes / Acceptance Criteria

1. A self-contained performance fixture preserves the actual validation's P4/H15, four-dimensional states, 19 adaptive nonlinear transitions, four-substep RK4 physics, true constants, declared Q/R/prior, preview goals, auto schedule and 1e-6 tolerance. Any synthetic observations and selected budget are labeled, not invented historical invocation facts.

2. Baseline and optimized runs use separate worktrees and environments with identical dependency versions and workload. Report synchronized cold, compilation-warm repeated-base and changed-observation times, actual compilation/cache behavior, iterations/map evaluations and full objective/certificate results.

3. Implement generic optimizations and validate the 100× minimum compilation-warm target, with 1600× as the stretch target. Do not claim speedup from a shorter window, hidden warm-start advantage, excluded later recompilation, frozen approximations, missing objective terms or weaker stopping.

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

**2026-10-02T04:01:41Z**

Final comparison acceptance, fixed before matched artifact runs: unchanged 1e-6 root and full current/successor FE certificate, finite/proper non-DATA marginals, no failure; all non-DATA edge means whitened by baseline posterior Cholesky differ <=1e-3, covariance differences whitened on both sides <=1e-3, and normalized total FE difference <=1e-6. These parity bounds acknowledge finite-tolerance roots, not a changed objective. Use identical cold + three repeated-base + three explicitly synthetic changed-observation plans, budget100. Record actual compiled executable cache sizes as well as provider misses. No host controller or warm-start substitution.

**2026-10-02T04:10:55Z**

Matched seven-plan comparison in progress, without concurrent numerical tests. Baseline repeated-observation median so far 36.8s. Changed observation 1 takes124.83s with provider miss1 and JAX trust executables increasing1->2; this reproduces the reported slowdown and proves skipped-first timing did not eliminate observation-dependent compilation. Fully adaptive optimization adds original host certificate on top of compiled raw/canonical successor checks, rejects nonfinite mapped/chart coordinates, preserves traces/budget on fallback, and reports actual proposal radius. New tests cover these failure mechanisms and the recent full CSTR/sigma startup separately. cff-wt1c now depends on opt-ewtf; no cycle.

**2026-10-02T04:18:27Z**

Independent source-only research confirms full model/physics/factor/variable ordering. Initial DATA now uses the retained normalization expression (not rounded [-1/3,.2]). Baseline diagnostic for change2: raw-vs-repacked/reconstructed FE drift3.58e-12 (normalized1.02e-13), fresh root3.10e-7, successor FE delta4.31e-7, original1e-6 checks pass. The unsupported64eps roundtrip assertion is replaced by a recorded drift bounded by original tolerance; neither solver nor preregistered posterior/FE parity tolerances change. Final comparison expanded (not shortened) to nine plans: seven original cases plus two labeled synthetic populated-history/mask/preview cases. Log every JAX JIT compilation inside timed bind/run/sync, alongside specifically labeled outer solver executable variants; compare equal uv.lock hash and hardware as well as versions. No matched denominator uses diagnostic/contended runs.

**2026-10-02T04:24:46Z**

Implementation committed in four logical steps:400c1fe batched geometry/shared dynamic RootProgram;4226d87 adaptive proposals and original host certificate/fallback;1ebe9a1 exact trust derivative/root-diagnostic reuse;fa3e405 maintained full-window fixture/benchmark/docs. Fresh focused runtime/workload tests10 passed; opt-in CSTR linearization/history/mask/preview and unchanged sigma failure2 passed. Earlier broader focused run60 passed with one incorrect test assertion that small budgets must always exhaust; fixed test now permits genuine trust convergence and checks no proposal execution/iteration overrun. Independent original-source template/array/insertion-order equality and physics/Jacobian parity pass. Source experiment is read-only. Final nine-case isolated timing restarted with source-exact normalization, whole-boundary JAX compile logging and matched lock/hardware; no production changes planned before final gate.

**2026-10-02T04:24:46Z**

Dependency overview (prerequisite --> dependent):
[x] cff-vxza --> [x] cff-qu2j --> [~] opt-ewtf (active, prerequisites complete)
                                      |
                                      +--> [ ] cff-wt1c (blocked by other PCGym prerequisites)
                                                |
                                                +--> [ ] cff-xa6z (PCGym epic)
Parallel ready roots cff-4yr6 (binding), cff-3kdk (readouts), cff-qtub/cff-wbhu (Colony), cff-ees0 (parameter hypotheses), cff-nd1q (action bounds) also lead to cff-wt1c through their recorded implementation dependencies. The runtime PR addresses only opt-ewtf; it does not complete those unrelated blockers. Built from tk dep tree --full, tk ready and tk blocked; tk dep cycle is clear.

**2026-10-02T04:44:20Z**

All nine matched cases pass original root/full FE/proper posterior audits. Baseline source-exact repeated median36.9305s; changed median120.6778s; full-history128.0774s and changedpreview116.4823s. Initial optimized repeated median.11014s, changed.10873s, history.14793s/.12188s, zero Krylov and no new outer solver variants. Broader JAX logging caught two dynamic_slice compilations for new FE/delta trace lengths (60/40 vs43); fixed readout to transfer fixed128-entry buffers then slice in NumPy. This does not alter inference, tolerances, outputs, or timers; rerun optimized nine cases and require zero warm timed JIT compilation before final gate. No baseline rerun is needed for this optimized-only readout fix.

**2026-10-02T04:49:14Z**

Validated final nine-case comparison (baseline92d6acc, optimized0fe5d80, separate frozen uv envs, Apple M2 Max; artifacts /Users/ravd/agent-share/CFFGAdaptiveRuntime/{baseline.json,optimized.json,report.html}): cold130.7154->26.4802s4.9x; repeated-base median36.9305->.105877s348.8x; changedobs median120.6778->.107526s1122.3x; populatedhistory128.0774->.129539s988.7x; historypreview116.4823->.104283s1117.0x. All nine converged under original1e-6 root/current+successor full-FE/domain checks; independent fresh-chart certificates and all58 non-DATA marginals proper. Max whitened mean2.7161e-5, whitened covariance1.1418e-6, normalized totalFE8.3143e-8, action4.2043e-6K: preregistered parity bounds pass. No warm timed JAX JIT compilation of any kind, including changed masks/history/preview; solver variants remain(0 trust,1diagnostic,1proposal). First source window maps176->45 and Krylov20536->0, iterations58->63 including startup (not per-sweep timing). 100x minimum exceeded;1600stretch not reached. No scientific compromise, original experiment unchanged, host7ms not substituted. Final suite remains the only pending numerical gate.

**2026-10-02T05:18:31Z**

Final gate at ef71254 passed2651 tests,192 deselected,0 failures; pytest1188.05s, measured full wall1202.469s. Discovered opt-un4a moves only a redundant three-state ownership diagnostic5.595s to opt-in slow; default estimate1196.874s and selection-only collection checks pass. All assertions remain, f32/f64/exact-support/uncomfortable-scale coverage stays default, and the already passing numerical gate is reused under the explicit budget rule (no rerun). Final gate history/dashboard updated. Scientific runtime code unchanged since0fe5d80; original source experiment and detached baseline remain unchanged.
