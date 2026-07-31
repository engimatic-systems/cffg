---
id: cff-lv2w
status: closed
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

## Notes

**2026-07-31T16:58:27Z**

## Measured handoff

Implemented a certificate-gated damped candidate phase for deterministic mixed-GFE plans, with at least half of small budgets reserved for trust globalization and mu3 excluded so its block alternation keeps the full budget. Engine-local derivative kernels are cached by exact support, dtype, continuous chart metric, and hardening phase; observations, parameters, chart bases, and hard values remain dynamic. RunResult now reports outer-map evaluations, trial evaluations, Krylov iterations, and solver compilation/cache-miss counts. The retained trust controller, objective, tolerances, support semantics, local solves, and FE/root certificates are unchanged.

Environment: Apple CPU backend (`cpu:0`), JAX 0.10.2, Python 3.12.13, x64 enabled. Baseline was detached commit 690d9b6 at `/tmp/cffg-lv2w-baseline.evTd3p`; proposed behavior commit is 600a775. Each command ran in a fresh process, timed `Engine.run` after model construction, synchronized result messages, discarded one warm-up, and retained five warm samples:

    git worktree add --detach /tmp/cffg-lv2w-baseline.evTd3p 690d9b6
    PYTHONPATH=/tmp/cffg-lv2w-baseline.evTd3p/src uv run python tools/benchmark_mixed_gfe.py <case> --repetitions 5
    uv run python tools/benchmark_mixed_gfe.py <case> --repetitions 5

Cases were `tmaze`, `d1`, `rolling10` (ten planning calls with changing full-support beliefs), `horizon8`, and `bfe`; proof premiums used `tmaze_picard`, `d1_picard`, and `rolling10_picard`.

Baseline 690d9b6 raw seconds (cold; retained warm samples; warm median):

- T-maze: 6.754390; [4.386625, 4.384611, 4.436715, 4.470319, 4.449983]; 4.436715.
- D1: 4.629084; [2.732470, 2.689255, 2.740687, 2.709627, 2.740193]; 2.732470.
- RollingAgent x10: 76.203040; [74.468486, 74.845621, 74.933607, 74.595214, 74.592028]; 74.595214.
- Horizon-8 cycle: 256.823734; [252.749525, 252.666139, 252.184772, 252.551620, 252.571892]; 252.571892.
- BFE control: 0.561357; [0.000334, 0.000320, 0.000297, 0.000355, 0.000293]; 0.000320.

Head 600a775 raw seconds and per-retained-run counts (cold; retained warm samples; warm median; maps/trials/CG/compilations):

- T-maze: 2.905752; [0.067657, 0.068264, 0.067632, 0.066962, 0.067666]; 0.067657; 96/0/0/0 (cold compilation count 1).
- D1: 2.349271; [0.027770, 0.027552, 0.026678, 0.026845, 0.027065]; 0.027065; 40/0/0/0 (cold compilation count 1).
- RollingAgent x10: 3.944545; [0.474492, 0.469845, 0.471397, 0.476129, 0.474412]; 0.474412; 619/0/0/0 (cold batch compilation count 1).
- Horizon-8 cycle: 61.752973; [15.464459, 15.430563, 15.411948, 15.428583, 15.423295]; 15.428583; 180/12/9600/0 (cold compilation count 1).
- BFE control: 0.544025; [0.000346, 0.000321, 0.000352, 0.000319, 0.000285]; 0.000321; 0/0/0/0.

Warm speedups are 65.58x T-maze, 100.96x D1, 157.24x RollingAgent, and 16.37x horizon-8. The common-case geometric mean is 101.35x (gate 5x). BFE changed by +0.29% (gate <=5%). Warm Picard medians were 0.043624, 0.014146, and 0.272014 seconds, giving proof-backed premiums of 1.55x, 1.91x, and 1.74x (gate <=2x). Every post-warmup run reported zero solver compilations; changed runtime priors matched a fresh engine, while exact support and dtype produced distinct cache plans.

Correctness and suite validation:

- `uv run pytest -q tests/test_outer_gfe_formalism.py tests/test_gfe.py tests/test_b_epistemics.py tests/test_mu3.py tests/test_runtime_status.py tests/test_schedule.py -m 'not slow'`: 65 passed in 67.53s.
- `/usr/bin/time -p uv run pytest -q -m slow`: 38 passed in 421.44s pytest time / 427.88s wall (7:08, gate <15 minutes). This includes the unchanged horizon-8 certified root and the legacy Picard budget-exhausted finite cycle.
- `uv run python tools/dashboard.py --run-suite`: 1,250 passed, 38 deselected in 299.86s pytest time; dashboard recorded 305.249s wall and zero failures. This was the single final short-suite run after the last code change.
- `git diff --check`: clean. `tk dep cycle`: no dependency cycles.

Handoff: no implementation work remains. The benchmark driver is `tools/benchmark_mixed_gfe.py`; architecture, derivation, and README runtime diagnostics are updated in place. cff-izjd can now consume the solver.
