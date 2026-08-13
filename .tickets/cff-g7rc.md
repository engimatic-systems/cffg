---
id: cff-g7rc
status: open
deps: [cff-lv2w]
links: [cff-wbhu]
created: 2026-08-01T02:19:45Z
type: feature
priority: 2
assignee: coopwilliams
external-ref: gh-49
tags: [gfe, performance, numerical-robustness]
---
# Reduce fixed certification and cold-compile overhead for tiny mixed-GFE solves

## Background in Plain English

The proof-backed mixed-GFE solver from cff-weff prevents a finite nonroot or a failed local solve from being reported as converged. Ticket cff-lv2w then added a bounded candidate phase and cached matrix-free derivative kernels, reducing its original warm overhead by 65.58x on T-maze, 100.96x on D1, 157.24x on ten changing-belief plans, and 16.37x on the retained horizon-8 globalization case. Those results are anchored to behavior commit `600a775` and the measured handoff in cff-lv2w.

PR #49 also compared `600a775` with pre-soundness commit `931758d` on the same Apple CPU, JAX 0.10.2, Python 3.12.13, x64 setup. The proof-backed warm T-maze and D1 solves were 3.71x and 26.12x slower, giving a 9.84x two-case geometric-mean premium; their cold first runs were 3.16x and 3.05x slower. Ten changing-belief `RollingAgent` plans were already 8.2% faster, and ordinary BFE was 4.97% slower. The old horizon-8 run exhausted its budget, so it is not a latency-equivalent result for the certified solve.

This ticket reduces fixed construction, certification, derivative-compilation, and synchronization costs for tiny one-shot mixed-GFE solves. It retains the raw-root, exact-support, inner-solver, feasibility, and stationary-BFE certificate, while keeping repeated planning and the horizon-8 globalized solve as no-regression controls. It does not own iterative sweep dispatch, HMM scan fusion, or resource-estimator repair.

## Outcomes / Acceptance Criteria

1. The optimized path preserves the mixed-GFE convergence contract.
   1.1. `RunStatus.CONVERGED` requires the raw undamped coupled outer residual to satisfy the unchanged dtype-derived threshold.
   1.2. `RunStatus.CONVERGED` requires the active exact-support face to remain unchanged through certification.
   1.3. `RunStatus.CONVERGED` requires every local solver status to be converged.
   1.4. `RunStatus.CONVERGED` requires the candidate and its mapped state to be feasible.
   1.5. `RunStatus.CONVERGED` requires a finite stationary Bethe-free-energy value from `_make_bfe`.
   1.6. The optimization changes no mixed-GFE stopping tolerance.
   1.7. The optimization changes no term or weight in `_make_bfe`.
   1.8. `RunResult.outer_gradient_norm` retains a documented, falsifiable stationarity meaning on a candidate accepted before trust-region entry.
   1.9. A finite nonroot remains a non-converged result.
   1.10. A non-finite outer evaluation remains `RunStatus.DIVERGED`.
   1.11. An empty exact-support face remains `RunStatus.INFEASIBLE`.
   1.12. A failed local solve retains its existing inner-failure termination reason.
2. A fresh-process benchmark makes the fixed costs and baselines reproducible.
   2.1. The benchmark includes canonical T-maze observation GFE, D1 dynamics curiosity, ten changing-belief `RollingAgent` plans, the horizon-8 retained cycle, and an ordinary non-GFE BFE control.
   2.2. Each cold median contains at least five samples launched in separate operating-system processes.
   2.3. Each warm median contains at least five blocked samples after one discarded warm-up in the same process.
   2.4. Every timed JAX result is synchronized before its timer stops.
   2.5. The benchmark reports model construction, ordinary step compilation, chart construction, candidate residual evaluation, support and local checks, BFE certification, root-diagnostic compilation, trust-kernel compilation, and host/device synchronization as separately attributable phases or counts.
   2.6. Each result records the commit, hardware, backend, JAX version, Python version, x64 setting, fixture parameters, tolerance, and sample count.
   2.7. Baseline worktrees at `600a775` and `931758d` run the same fixture definitions and timing boundaries as the candidate change.
   2.8. The `931758d` horizon-8 budget-exhausted result is labeled as outcome-incomparable with certified convergence.
3. Tiny one-shot latency improves without moving cost into the controls.
   3.1. The T-maze cold `Engine.run` median is at least 2x faster than `600a775` under criterion 2.
   3.2. The D1 cold `Engine.run` median is at least 2x faster than `600a775` under criterion 2.
   3.3. Against `931758d`, the geometric mean of the T-maze and D1 warm-time ratios is no greater than 3.28x, one third of the recorded 9.84x premium.
   3.4. The warm median for ten changing-belief planning calls is no more than 5% slower than `600a775`.
   3.5. The ordinary BFE warm median is no more than 5% slower than `600a775`.
   3.6. The certified horizon-8 warm median is no more than 10% slower than `600a775`.
   3.7. The horizon-8 optimized result agrees with the `600a775` root under the existing fixture tolerance.
   3.8. The horizon-8 optimized result has the same termination reason as `600a775`.
   3.9. Explicit legacy Picard returns budget exhaustion on the horizon-8 fixture.
   3.10. Explicit legacy Picard reproduces the horizon-8 finite bounded cycle under the existing fixture tolerance.
4. Expensive derivative work is lazy and cache reuse is safe.
   4.1. A benign T-maze candidate that satisfies criterion 1 reports zero compilations of the Steihaug trust kernel.
   4.2. The benign result in criterion 4.1 reports zero trust-region trial evaluations.
   4.3. The benign result in criterion 4.1 reports zero Krylov iterations.
   4.4. Runtime diagnostics distinguish lightweight certificate compilation from trust-kernel compilation.
   4.5. A deliberate nonroot candidate compiles the trust kernel only when control transfers to the globalized solver.
   4.6. The deliberate nonroot case reaches the `600a775` root under the existing fixture tolerance.
   4.7. The deliberate nonroot case returns the same termination reason as `600a775`.
   4.8. Changing only observations reuses a compatible compiled artifact.
   4.9. Changing only parameter values reuses a compatible compiled artifact.
   4.10. Changing exact support selects a distinct chart and support-specific compiled artifact.
   4.11. Changing dtype selects a distinct compiled artifact.
   4.12. Changing hardening phase selects a distinct compiled artifact.
   4.13. Changing model structure selects a distinct compiled artifact.
   4.14. The T-maze and D1 fast path retains observations as runtime inputs.
   4.15. The T-maze and D1 fast path retains parameter values as runtime inputs.
   4.16. A candidate outside the tiny-chart fast path falls back to the globalized solver.
   4.17. The fast-path recognizer imposes no fixed chart-size ceiling.
5. Automated evidence covers the numerical robustness and duplicated execution surfaces.
   5.1. Literal log-zero support changes trigger the rebuild required by criterion 4.10.
   5.2. The optimized path introduces no probability floor.
   5.3. A vacuous incoming message retains its existing neutral-message behavior.
   5.4. An improper incoming state retains its existing non-converged classification.
   5.5. A non-finite incoming state retains its existing diverged classification.
   5.6. Float32 execution derives its residual guard from float32 machine properties.
   5.7. Float32 execution derives its rank guard from float32 machine properties.
   5.8. Float32 execution derives its trust guard from float32 machine properties.
   5.9. A float32 artifact does not reuse an x64 cache entry.
   5.10. The retained horizon-8 fixture exercises its 24-logit mixed-GFE chart without a dense whole-graph Jacobian.
   5.11. The retained D1 hostile fixture exercises 32 positions by 4 selectors without a dense whole-graph Jacobian.
   5.12. Host-controlled and JIT-compiled certificate calculations agree under their existing tolerance.
   5.13. Flooding and mixed schedules agree on the shared fixed-point equations under their existing tolerance.
   5.14. Sequential and `vmap` trust-kernel execution produce the same step under their existing tolerance.
   5.15. Sequential and `vmap` trust-kernel execution produce the same diagnostics under their existing tolerance.
6. Repository evidence and documentation make the result reviewable.
   6.1. Focused mixed-GFE, outer-formalism, runtime-status, agent, and benchmark tests pass.
   6.2. The opt-in slow selection passes once after the final behavior change.
   6.3. The dashboard-backed default short suite passes once after the final code change.
   6.4. `docs/ARCHITECTURE.md` is updated in place if cache ownership or the host/device boundary changes.
   6.5. `docs/DERIVATIONS.md` D7 is updated in place if the stationarity certificate in criterion 1.8 changes realization.
   6.6. `README.md` is updated in place if public `RunResult` diagnostics change.
   6.7. `docs/LIMITATIONS.md` is updated in place if the optimized path introduces an accepted capability boundary.
   6.8. The closing ticket note records raw cold and warm samples.
   6.9. The closing ticket note records comparison ratios.
   6.10. The closing ticket note records exact benchmark commands.
   6.11. The closing ticket note records the environment fields from criterion 2.6.
   6.12. The closing ticket note records phase timings or counts.
   6.13. The closing ticket note records operation counts.
   6.14. The closing ticket note records compilation counts.
   6.15. The closing ticket note records the `tk dep cycle` result.
   6.16. The closing ticket note records the `git diff --check` result.

## Methodological Suggestions

These methods are suggestions. An implementation may use another design if it satisfies every numbered outcome with equivalent evidence.

Start by extending `tools/benchmark_mixed_gfe.py` so repeated cold measurements are genuinely separate processes and the fixed-cost phases are observable. Preserve the cff-lv2w case builders and run detached baseline worktrees through the same harness. Treat the first `Engine.run` after model construction as the cold inference boundary, and report model construction separately rather than silently moving work outside the timer.

The likely executor seam pairs `src/cffg/engine.py:_outer_kernel_plan`, `src/cffg/outer.py:_make_root_diagnostics`, and `src/cffg/outer.py:_make_trust_kernel`. Profile whether splitting the lightweight root diagnostic from `OuterKernelPlan`, fusing repeated reconstruction/residual/support/BFE reads, or caching a structure/support/dtype-specific certificate across equivalent engines meets the gates. A root diagnostic may remain independently compiled; the Steihaug JVP/VJP kernel must stay lazy until a candidate fails certification. Any replacement for the `_make_root_diagnostics` gradient needs a derivation-backed conservative meaning and executable evidence before criterion 1.8 can pass.

Use the **Runtime numerical or status invariant** row in `docs/ARCHITECTURE.md`. The authorities are the `AGENTS.md` numerical contract, `docs/DERIVATIONS.md` D7, `Engine.run` and `_run_outer_trust_region` in `src/cffg/engine.py`, and the fixed-support controller in `src/cffg/outer.py`. New factor equations, IR fields, serialization, public readouts, learnable membership, and checkpoint migration are outside this change.

Review the duplicated execution surfaces explicitly:

- Host/JIT: compare host status mapping with the compiled residual, certificate, and derivative outputs.
- Cold/warm: use independent process samples for cold compilation and retained in-process samples for warm execution.
- Flooding/mixed: keep flooding as the independent shared-root comparison where the fixture permits both schedules.
- Single-instance/`vmap`: retain the matrix-free kernel parity test; cff-wbhu owns batching the controller loop itself.
- Current/legacy checkpoint decoding: not applicable because this ticket changes no persisted state.

Keep exact support, dtype, continuous chart metric, hardening phase, and model structure in any reusable-artifact key. Keep observations, parameter values, chart bases, hard values, and per-run radii dynamic. A cache may be engine-local or process-level, but its identity and invalidation rules must be reviewable. Run focused tests during implementation; after the final behavior change, run the required slow selection when the globalization behavior needs it and the dashboard-backed short suite once.

## Future work

- cff-wbhu owns JIT compilation of the iterative Engine/Colony controller loop, per-sweep dispatch removal, and its Colony throughput gate. This ticket may share low-level loop machinery but does not absorb that acceptance scope.
- cff-yl0a owns exact discrete HMM transition/equality/emission scan fusion and its strict low-level reference target.
- cff-j7mm owns persistent/transient arena byte accounting and supported-backend XLA cost reporting in the resource estimator.
- cff-g7rc is a terminal strict-performance continuation: it does not block cff-ahps or another downstream roadmap ticket.

## Definitions

- **Bethe free energy (BFE):** The scientific objective computed by `_make_bfe`; mixed-GFE convergence requires its finite value to be stationary at the accepted outer root.
- **Candidate phase:** The bounded damped fixed-point sequence that may propose a benign root before trust-region globalization.
- **Cold `Engine.run`:** The first inference call in a fresh process after model construction; its timer includes compilation caused by that call.
- **Compilation count:** A diagnostic count of new compiled certificate or trust artifacts created during one measured call.
- **D1 dynamics curiosity:** The transition-parameter epistemic fixture derived in `docs/DERIVATIONS.md` D1.
- **Exact support:** The categorical states whose log messages are not the reserved `NEG` sentinel; changing this set rebuilds the active chart.
- **Fresh process:** A new operating-system process with no inherited JAX executable cache from another sample.
- **Geometric-mean premium:** The square root of the product of the T-maze and D1 candidate-to-baseline warm-time ratios.
- **Hardening phase:** One runtime phase with a frozen set of active hard constraints.
- **Horizon-8 retained cycle:** The eight-position, two-mode learned-transition fixture that the proof-backed solver converges and legacy Picard cycles on.
- **JVP/VJP:** Jacobian-vector and vector-Jacobian products used to apply derivatives without constructing a dense Jacobian.
- **Krylov iteration:** One matrix-free linear iteration inside the Steihaug trust-region solve.
- **Lightweight root diagnostic:** The root merit and stationarity evidence evaluated without entering the Steihaug trial-step kernel.
- **Mixed GFE:** A graph whose coupled outer map contains Generalised-free-energy composites and ordinary message updates.
- **One-shot solve:** A solve measured before repeated calls amortize its compilation and fixed certification work.
- **Raw outer root:** A fixed point of the complete undamped mixed-GFE outer map, before merit minimization or damping changes its interpretation.
- **Stationary BFE:** A finite `_make_bfe` value whose change under one complete outer-map evaluation satisfies the dtype-derived threshold.
- **Steihaug trust kernel:** The compiled matrix-free Gauss-Newton/Levenberg-Marquardt trust-region step implemented by `_make_trust_kernel`.
- **T-maze:** The canonical observation-epistemics model defined by `src/cffg/models/tmaze.py`.
- **Warm median:** The median blocked latency after the discarded warm-up has completed.

## Notes

**2026-08-08T04:43:39Z**

Retained wiggle candidates after the bounded current scan (candidate -> intended meaning): canonical -> the literature-standard T-maze fixture; fixed -> constant per-solve overhead or a pinned chart/support configuration as named; preserve -> semantic or numeric invariant; public -> documented RunResult consumer contract; reusable -> compiled-artifact or fixture reuse; same -> value equality, equivalent measurement configuration, or samples from one process as named; implement -> a named repository function or planned construction; support -> literal probability support; complete -> the full undamped outer-map evaluation or finished warmup; remaining -> a runtime property that continues unchanged; close -> administrative ticket completion; certify -> falsifiable mixed-GFE runtime certificate; check -> named automated, support, or validation check; converge -> RunStatus convergence criterion; equivalent -> equal observable solver result or acceptance evidence; estimate -> cff-j7mm resource-estimator output; evidence -> named test, benchmark, derivation, or ticket artifact; exact -> literal NEG-based support, exact command text, or the named exact-HMM future scope; fixed_point -> root of the complete undamped outer map or flooding/mixed equation equality; parity -> tolerance agreement for a named duplicated execution pair; pass -> certificate satisfaction or automated tests_succeed; reproduce -> executable reproduction of the bounded Picard cycle or benchmark protocol; robust -> the AGENTS.md numerical robustness contract; sound -> the proof-backed convergence semantics contrasted with pre-soundness commit 931758d; test -> automated executable check; validate -> numerical comparison or repository validation command named by the sentence; infer -> fixed-parameter mixed-GFE state calculation; learn -> the retained learned-transition fixture; learnable -> the declared persistent-parameter category excluded from this change path; trial -> one trust-region candidate evaluation; performance -> measured cold or warm latency; small -> lightweight certificate cost or tiny chart size; speed -> measured latency ratio; can -> mathematical possibility or optional implementation capability as selected by the sentence; intended -> declared semantic meaning in this audit note; may -> implementation permission; explicit_owner -> cff-g7rc owns one-shot fixed overhead while named Future-work tickets own their listed surfaces; named_ticket_handoff -> cff-wbhu, cff-yl0a, and cff-j7mm are concrete owners; scope_exclusion -> factor equations, IR, persistence, readouts, and downstream roadmap gates are deliberately outside this ticket; any -> each member of the complete reusable-artifact key class; at_least -> inclusive sample-count or speedup lower bound; every -> each member of a locally named solver-status, result, criterion, or timing set; only -> exact single-variable cache-key changes or ticket boundary as stated locally; current -> the named current/legacy checkpoint review pair; existing -> repository tolerance, fixture, or behavior that predates this optimization; read -> machine access to reconstruction, residual, support, or BFE state. Each retained occurrence is bounded by its sentence, symbol, metric, commit, or named artifact.

Intelligible-ticket review: complete.
