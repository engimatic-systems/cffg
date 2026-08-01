---
id: cff-lv2w
status: open
deps: [cff-weff]
links: [cff-wbhu]
created: 2026-07-31T14:44:58Z
type: bug
priority: 1
assignee: coopwilliams
external-ref: gh-48
tags: [gfe, performance, numerical-robustness]
---
# Recover mixed-GFE throughput without weakening certification

## Plain English

Proof-backed mixed-GFE inference will become fast enough for routine agent planning while retaining its exact outer-root, support, inner-solver, and free-energy certificates. The work will measure cold and warm execution separately and remove the avoidable compilation, candidate-evaluation, and host-dispatch costs exposed by cff-weff. Implementers may choose any optimization strategy that meets the hard correctness and performance gates below; the suggested techniques are intentionally non-binding.

## Evidence and scope

Draft PR #48 fixes a real correctness defect but its nearly-full-suite measurement rose from 361.268 seconds for 1,283 tests at b65b145 to 3,899.181 seconds for 1,286 tests at f6ac234. That approximately 10.8x suite increase is evidence of a severe mixed-GFE regression, not a claim that every Engine.run call is exactly 10.8x slower. The current correctness reference and performance baseline is PR-head commit 690d9b6.

The old Picard path remains a comparison oracle only on benign fixtures where an independent outer residual and unchanged FE ledger certify that it reached the same root. It is not an acceptable fallback for the retained learned-transition cycle.

## Change-path handoff

Applicable row: **Runtime numerical or status invariant**.

- Authority: `AGENTS.md` numerical contract; `docs/DERIVATIONS.md` D7; `src/cffg/engine.py` mixed-GFE dispatch and `RunResult`; `src/cffg/outer.py` fixed-support controller and trust kernel.
- Execution surfaces: cold compilation and warmed `Engine.run`; observation-GFE and D1 transition-GFE; repeated `RollingAgent` planning; single-instance and `Colony` execution; the ordinary BFE path as a no-regression control.
- Duplicated pairs: host/JIT and single-instance/vmap are affected; mixed/flooding comparisons are affected; exact two-pass execution, serialization/checkpoint decoding, and unrelated factor-family rules are N/A.
- Focused fixtures: canonical T-maze, D1 dynamics curiosity, repeated fixed-support `RollingAgent` ticks, the horizon-8 one-update corridor cycle, a non-GFE BFE control, and the existing 38-test opt-in slow regression selection.
- Documentation: update `docs/ARCHITECTURE.md` if caching, controller ownership, or device/host boundaries change. Record benchmark commands and results in ticket notes, not a new markdown file.

## Measurement protocol

- Compare against commit 690d9b6 on the same machine, backend, dtype, fixture inputs, tolerances, and JAX configuration.
- Separate fresh-process compile-plus-first-run time from warmed solve time. Synchronize JAX results before stopping each timer.
- Report the median of at least five warmed repetitions after one discarded warm-up, plus outer-map evaluations, trial evaluations, CG iterations, and solver compilations.
- Benchmark the common-case geometric mean over T-maze observation GFE, D1 dynamics curiosity, and ten repeated fixed-support `RollingAgent` planning ticks. Benchmark the retained horizon-8 cycle separately.
- Performance measurements must not weaken tolerances, reduce model sizes or horizons, omit FE/support certification, or change the objective.

## Hostile inputs

- Exact log-zero support: cached charts or compiled kernels must be keyed or invalidated so a support change rebuilds the exact active chart; no probability floor is permitted.
- Vacuous, improper, and non-finite states: optimization must preserve the existing infeasible/diverged/inner-failure distinctions and must not bypass validation on a fast path.
- Float32: compilation caches and stopping guards must distinguish dtype, and all guards remain dtype-derived.
- Uncomfortable scale: retain the horizon-8 learned-transition cycle, the 24-logit GFE-TM member, and the 32-position by 4-selector case without allocating a dense whole-graph Jacobian.
- Repeated runtime use: fixed structure/support/dtype agent ticks must reuse compiled work, while changed structure, support, dtype, or hardening phase must invalidate it safely.

## Dependency map

Legend: prerequisite --> dependent

    [x] cff-weff correctness reference
              |
              +--> [ ] mixed-GFE performance recovery (ready)
                          |
                          +--> [ ] cff-izjd hybrid GFE
                                      |
                                      +--> [ ] cff-ahps roadmap epic

    [ ] cff-wbhu compiled Engine/Colony loop (ready in parallel; linked)

The new performance ticket has no unresolved prerequisite after cff-weff. It immediately blocks cff-izjd from consuming the solver before the hard throughput gates pass; cff-wbhu can proceed in parallel and share device-loop machinery.

## Design

The following methods are suggestions, not requirements. They may be reordered, replaced, combined, or rejected when measurements support another design. Acceptance is based only on the hard gates below.

1. Establish isolated cold/warm baselines and attribute time to compilation, outer-map calls, derivative products, trial evaluation, free-energy certification, and host/device synchronization.
2. Try a bounded Anderson/Picard candidate phase for benign problems, returning only after the same independent root, support, inner-solver, and FE certificates pass; otherwise enter the trust-region method.
3. Cache fixed-support charts, residual functions, and compiled derivative/trust kernels by structure, support signature, dtype, and hardening phase.
4. Select one Cauchy-certified trial from the local model per trust iteration and use ordinary accept-or-shrink behavior instead of eagerly evaluating several alternatives. Activate the shadow sequence only after rejection or stagnation if evidence supports doing so.
5. Cache current residual, reconstruction, and FE values when the accepted state has not changed.
6. Move fixed-support controller state into a JIT `lax.while_loop` or equivalent device loop, retaining support changes and truthful termination mapping as explicit host restart boundaries. Coordinate reusable loop machinery with cff-wbhu.
7. Replace the expensive terminal trust solve used only for gradient diagnostics with an equivalent lightweight residual/VJP certificate if the derivation permits it.

## Acceptance Criteria

These criteria are mandatory regardless of implementation method.

1. **No correctness trade:** all existing cff-weff marginal, message, FE-ledger, exact-support, float32, status, and hostile-input tolerances remain unchanged. `CONVERGED` still requires the independent coupled outer-root certificate, converged inner solves, feasible support, and finite stationary FE. No objective reweighting, looser tolerance, reduced model, dense whole-graph Jacobian, or new solver dependency is allowed.
2. **Pathological case remains solved:** the unchanged horizon-8 one-update learned-transition fixture reaches the same certified root and termination reason as PR #48; legacy Picard still reproduces budget exhaustion and the finite outer cycle.
3. **Common-case speedup:** the geometric mean warmed time across canonical T-maze observation GFE, D1 dynamics curiosity, and ten repeated fixed-support `RollingAgent` planning ticks is at least 5x faster than commit 690d9b6 under the measurement protocol.
4. **Pathological-case speedup:** the warmed horizon-8 one-update cycle is at least 3x faster than commit 690d9b6 while preserving its model, tolerance, support, and certificates.
5. **Bounded correctness premium:** on each benign common-case fixture where explicit legacy Picard independently reaches the same certified root, warmed proof-backed inference takes no more than 2x the Picard wall time.
6. **No ordinary-engine regression:** the warmed non-GFE BFE control is no more than 5% slower than commit 690d9b6, using the median protocol.
7. **Compilation reuse:** after one warm-up, ten fixed-structure/support/dtype/hardening agent ticks trigger zero new mixed-GFE solver compilations. A deliberate support or dtype change triggers the required distinct compilation or chart rebuild and remains correct.
8. **End-to-end ceiling:** the existing 38-test `slow` selection passes in less than 15 minutes on the same machine. The default short suite also passes once at the proposed PR head; neither result is rerun without a later code change.
9. **Measured handoff:** the ticket closes with exact benchmark commands, environment/backend identification, raw per-case cold and warmed medians, operation/compilation counts, comparison ratios, and `tk dep cycle` plus `git diff --check` results recorded in a ticket note.
