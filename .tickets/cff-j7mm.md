---
id: cff-j7mm
status: open
deps: [cff-44we]
links: [cff-fxmj, cff-44we]
created: 2026-07-10T18:05:50Z
type: bug
priority: 2
assignee: coopwilliams
tags: [adversarial-reviewer, performance, estimator]
---
# Resource estimator ignores non-discrete arenas and its XLA analysis no longer calls step correctly

## Background in Plain English

The resource estimator must tell users how much compiled inference state a model stores, how much work its executable performs, and how long that work takes. Its persistent-memory totals now include the eight shape-segmented arena arrays added by cff-44we, but the rest of the report is still unreliable. The XLA analysis calls an obsolete two-array step signature, catches the resulting exception, and quietly returns no FLOP or byte-traffic values.

A 20-step, two-dimensional LGSSM exposed the defect: the old report showed 1,104 bytes although the state leaves occupied 5,088 bytes, and it called discrete `message_bytes` zero although Gaussian messages occupied 3,744 bytes. Memory reports must distinguish exact persistent JAX-array storage from backend-reported temporary buffers. Timing reports must also distinguish compilation, first execution, a flooding sweep, and an active scheduled run; those units are not interchangeable.

cff-6s31 consumes this ticket as the trustworthy compiled-arena baseline for compact plate estimates. This ticket covers already compiled plate arenas. cff-6s31 separately derives resource counts from compact `PlateSpec.dims` before compilation.

## Outcomes / Acceptance Criteria

1. The shape-segmented prerequisite is satisfied before this ticket closes.
   1.1. cff-44we remains closed with exact flat storage for discrete message segments.
   1.2. cff-44we remains closed with exact flat storage for Gaussian eta segments.
   1.3. cff-44we remains closed with exact flat storage for Gaussian precision segments.
   1.4. cff-44we remains closed with named persistent byte counts for the eight arena arrays.
2. Persistent-memory accounting equals the initialized `State` pytree.
   2.1. The report names `messages` bytes.
   2.2. The reported `messages` bytes equal `state.messages.nbytes`.
   2.3. The report names `g_eta` bytes.
   2.4. The reported `g_eta` bytes equal `state.g_eta.nbytes`.
   2.5. The report names `g_lam` bytes.
   2.6. The reported `g_lam` bytes equal `state.g_lam.nbytes`.
   2.7. The report names `g_obs` bytes.
   2.8. The reported `g_obs` bytes equal `state.g_obs.nbytes`.
   2.9. The report names `g_mask` bytes.
   2.10. The reported `g_mask` bytes equal `state.g_mask.nbytes`.
   2.11. The report names `ef_eta` bytes.
   2.12. The reported `ef_eta` bytes equal `state.ef_eta.nbytes`.
   2.13. The report names `ef_obs` bytes.
   2.14. The reported `ef_obs` bytes equal `state.ef_obs.nbytes`.
   2.15. The report names `ef_mask` bytes.
   2.16. The reported `ef_mask` bytes equal `state.ef_mask.nbytes`.
   2.17. The report names each `state.params` leaf by its pytree path.
   2.18. Each reported parameter-leaf byte count equals that leaf's `nbytes`.
   2.19. The parameter total equals the sum of the reported parameter-leaf bytes.
   2.20. The arena total equals the sum of the eight named arena-array byte counts.
   2.21. The persistent-state total equals the sum of every array leaf in `State`.
   2.22. An unused arena appears with a zero-byte value instead of disappearing from the report.
   2.23. The compatibility property `message_bytes` continues to return discrete `messages` bytes.
   2.24. The report does not label `message_bytes` as total message storage.
3. Compiled-executable analysis uses the executable's complete input signature.
   3.1. Flooding-step lowering receives `messages`.
   3.2. Flooding-step lowering receives `g_eta`.
   3.3. Flooding-step lowering receives `g_lam`.
   3.4. Flooding-step lowering receives `g_obs`.
   3.5. Flooding-step lowering receives `g_mask`.
   3.6. Flooding-step lowering receives `ef_eta`.
   3.7. Flooding-step lowering receives `ef_obs`.
   3.8. Flooding-step lowering receives `ef_mask`.
   3.9. Flooding-step lowering receives the complete parameter pytree.
   3.10. Flooding-step lowering receives the selected damping value.
   3.11. Flooding-step lowering receives the selected hardening value.
   3.12. An exact two-pass engine reports compiled analysis for its `_exact_unit` executable.
   3.13. A colored engine reports compiled analysis for its `_iter_step` executable.
   3.14. A mixed engine reports compiled analysis for its `_iter_step` executable.
   3.15. An engine with a prime stage reports compiled analysis for its `_prime` executable.
   3.16. Each compiled-analysis record names its executable work unit.
   3.17. Each compiled-memory record names device argument bytes from XLA `memory_analysis()`.
   3.18. Each compiled-memory record names device output bytes from XLA `memory_analysis()`.
   3.19. Each compiled-memory record names device alias bytes from XLA `memory_analysis()`.
   3.20. Each compiled-memory record names device temporary bytes from XLA `memory_analysis()`.
   3.21. Each compiled-memory record names device generated-code bytes from XLA `memory_analysis()`.
   3.22. Each compiled-memory record names host argument bytes from XLA `memory_analysis()`.
   3.23. Each compiled-memory record names host output bytes from XLA `memory_analysis()`.
   3.24. Each compiled-memory record names host alias bytes from XLA `memory_analysis()`.
   3.25. Each compiled-memory record names host temporary bytes from XLA `memory_analysis()`.
   3.26. Each compiled-memory record names host generated-code bytes from XLA `memory_analysis()`.
   3.27. The device peak-byte figure equals device argument bytes plus device output bytes minus device alias bytes plus device temporary bytes.
   3.28. The host peak-byte figure equals host argument bytes plus host output bytes minus host alias bytes plus host temporary bytes.
   3.29. The report labels XLA temporary bytes as compiler-planned buffers rather than process peak RSS.
4. XLA cost analysis is usable and failures remain visible.
   4.1. The repository CPU backend reports a positive FLOP count for the deterministic T-maze fixture.
   4.2. The repository CPU backend reports a positive byte-traffic count for the deterministic T-maze fixture.
   4.3. Cost analysis accepts a backend result returned as one dictionary.
   4.4. Cost analysis accepts a backend result returned as a one-element list.
   4.5. Cost analysis accepts a backend result returned as a one-element tuple.
   4.6. Each cost record names the JAX backend.
   4.7. The flooding cost record names its analyzed executable as a flooding step.
   4.8. An unavailable backend cost model produces a structured `unavailable` diagnostic.
   4.9. The unavailable diagnostic includes the backend name.
   4.10. The unavailable diagnostic includes the exception type.
   4.11. The unavailable diagnostic includes the exception message.
   4.12. A missing `flops` key produces a structured malformed-result diagnostic.
   4.13. A missing `bytes accessed` key produces a structured malformed-result diagnostic.
   4.14. A non-finite FLOP value produces a structured malformed-result diagnostic.
   4.15. A non-finite byte-traffic value produces a structured malformed-result diagnostic.
   4.16. An exception not classified as backend unavailability propagates to the caller.
5. Exact byte accounting covers each compiled model family and batch shape.
   5.1. A discrete-only fixture's reported persistent bytes equal its actual state leaves.
   5.2. An observed Gaussian fixture's reported persistent bytes equal its actual state leaves.
   5.3. An observed exponential-family fixture's reported persistent bytes equal its actual state leaves.
   5.4. A hybrid discrete/Gaussian fixture's reported persistent bytes equal its actual state leaves.
   5.5. A GFE fixture's reported parameter bytes include its derived moment arrays.
   5.6. A prepared mu3 fixture's reported parameter bytes include its particle arrays.
   5.7. A compiled plate fixture's reported persistent bytes equal its actual state leaves.
   5.8. A one-instance Colony batch has the same persistent byte total as its unbatched state.
   5.9. A Colony batch of size `B` reports `B` times each arena-array byte count.
   5.10. A Colony batch of size `B` reports `B` times each parameter-leaf byte count.
   5.11. A Colony batch report names its batch size.
   5.12. A vmapped Colony executable reports its compiled temporary bytes separately from persistent batch bytes.
   5.13. The cff-44we heterogeneous fixture reports the theoretical sum-K discrete bytes.
   5.14. The cff-44we heterogeneous fixture reports the theoretical sum-d Gaussian eta bytes.
   5.15. The cff-44we heterogeneous fixture reports the theoretical sum-d-squared Gaussian precision bytes.
6. Calibration separates compilation, first execution, and warm execution.
   6.1. A calibration record names the executable work unit it measures.
   6.2. Flooding calibration names its work unit `flooding_step`.
   6.3. Scheduled calibration names the active schedule kind.
   6.4. Lowering time is reported separately from compilation time.
   6.5. Compilation time is reported separately from first-execution time.
   6.6. First-execution time is reported separately from warm-execution time.
   6.7. Warm timing begins after the analyzed executable has compiled and completed once.
   6.8. Calibration blocks until every output array leaf is ready before stopping a timer.
   6.9. Calibration reports the number of executed work units.
   6.10. Iterative-schedule time per work unit divides by the executed count rather than the requested budget.
   6.11. Exact two-pass timing reports one status-bearing run rather than a fictional number of sweeps.
   6.12. The calibration result records its `RunStatus` when the measured unit returns one.
   6.13. Warm latency is the median of at least five measured repetitions.
   6.14. A calibration sample count below one is rejected before lowering.
   6.15. `predict_seconds` states the work unit used by its prediction.
   6.16. `predict_seconds` refuses to extrapolate an exact one-shot schedule as repeated sweeps.
   6.17. `predict_seconds` returns no prediction after `DIVERGED` calibration.
   6.18. `predict_seconds` returns no prediction after `INFEASIBLE` calibration.
7. The estimator obeys the numerical robustness contract.
   7.1. Sparse and dense parameter values with equal shapes produce equal persistent byte counts.
   7.2. Vacuous and informative messages with equal shapes produce equal persistent byte counts.
   7.3. Improper and proper Gaussian messages with equal shapes produce equal persistent byte counts.
   7.4. Float32 arena-array byte counts equal the actual float32 leaves.
   7.5. Float64 arena-array byte counts equal the actual float64 leaves.
   7.6. Float32 parameter-leaf byte counts equal the actual float32 leaves.
   7.7. Integer and Boolean parameter leaves retain their respective dtype-sized byte counts.
   7.8. The 1,000-small-domain plus K=10,000 and d=128 fixture is the uncomfortable-domain-scale case.
   7.9. A Colony batch size of at least 128 is the uncomfortable-batch-scale case.
   7.10. A zero batch size is rejected.
   7.11. A negative batch size is rejected.
   7.12. A batch byte-count product above `numpy.iinfo(numpy.int64).max` fails before allocation.
   7.13. The batch-overflow diagnostic names the requested batch size.
   7.14. The batch-overflow diagnostic names the overflowing byte category.
8. The public report and duplicated execution surfaces remain honest.
   8.1. `Estimate.summary()` labels persistent bytes as exact.
   8.2. `Estimate.summary()` labels XLA FLOPs as backend-reported.
   8.3. `Estimate.summary()` labels XLA byte traffic as backend-reported.
   8.4. `Estimate.summary()` labels calibrated latency as empirical.
   8.5. `Estimate.summary()` includes an unavailable or malformed XLA diagnostic instead of only `n/a`.
   8.6. Host-side byte arithmetic equals the JAX pytree leaf total.
   8.7. Flooding-step analysis uses the same argument order as `Engine.step` delegates to `_step`.
   8.8. Scheduled calibration exercises the engine's active scheduled path.
   8.9. Single-instance and `vmap` persistent totals agree after division by batch size.
   8.10. Checkpoint decoding is structurally irrelevant because the estimator adds no checkpoint state or schema field.
   8.11. JSON and `.cffgz` decoding are structurally irrelevant because estimates are derived from a reconstructed engine.
   8.12. The implementation adds no runtime dependency.
   8.13. The implementation adds no development dependency.
9. Durable documentation and executable evidence cover the repaired estimator.
   9.1. `README.md` explains the difference between persistent bytes and compiled temporary bytes.
   9.2. `README.md` explains the difference between a flooding step and an active scheduled run.
   9.3. `README.md` shows structured XLA diagnostics.
   9.4. `docs/ARCHITECTURE.md` removes the statement that XLA cost analysis is untrustworthy after supported-backend evidence passes.
   9.5. `docs/ARCHITECTURE.md` documents the estimator's exact and backend-dependent boundaries.
   9.6. `docs/LIMITATIONS.md` records backends that do not expose cost or memory analysis.
   9.7. Focused estimator, compiler, discrete, Gaussian, EF, hybrid, GFE, plate, Colony, schedule, and hostile-input tests pass.
   9.8. The dashboard-backed short suite passes after the final code change.

## Methodological Suggestions

These suggestions are optional. Another method is acceptable if it satisfies the numbered outcomes with equivalent artifacts and executable evidence.

- Use the **Runtime numerical or status invariant** row in `docs/ARCHITECTURE.md`. Walk flooding and scheduled executables, host and JIT boundaries, single-instance and `vmap` shapes, status consumers, the estimator report, and documentation. Checkpoint and bundle codecs are N/A for implementation but need the recorded reasons in criteria 8.10 and 8.11.
- Derive persistent bytes by flattening `State` with paths, then aggregate the eight arena fields and the parameter subtree. This automatically includes future parameter-state leaves without maintaining another hard-coded key list.
- Lower one wrapper with the full `_step` argument list shown by `Engine.step`. Reuse its compiled executable for `cost_analysis()`, `memory_analysis()`, and flooding calibration so the reported metrics refer to one artifact.
- Give unsupported backend analysis a narrow exception classifier. Let programming errors and signature drift fail tests instead of converting them to missing metrics.
- Treat scheduled-run calibration as a separate empirical record. Exact two-pass, colored, mixed, and flooding schedules perform different work even when callers pass the same `n_iters` value.
- Test byte totals against `jax.tree_util.tree_leaves` and XLA metrics against the installed CPU backend. Use injected analysis results for list, tuple, unavailable, malformed, and non-finite branches.
- Preserve `message_bytes` as a compatibility property, but make `arena_state_bytes` and `state_bytes` the unambiguous totals.
- Run focused tests during implementation. After the final code change, run the dashboard-backed short suite once under the `AGENTS.md` protocol.

## Future work

- cff-6s31 extends this compiled-engine baseline with analytic resource counts derived directly from compact plate dimensions before arena construction.
- No other future work is identified.

## Definitions

- **Active schedule:** The flooding, exact two-pass, colored, or mixed execution plan selected by an `Engine`.
- **Alias bytes:** Output storage that reuses input storage according to XLA's compiled memory plan.
- **Arena:** Flat, shape-segmented JAX arrays and their static addressing plans.
- **Byte traffic:** XLA's estimate of bytes read and written while one compiled executable runs; it is not allocated memory.
- **Calibration:** A synchronized timing measurement of an already identified executable work unit.
- **Colony:** CFFG's batching layer for several model instances with one compiled structure.
- **Compiled peak bytes:** XLA argument bytes plus output bytes minus alias bytes plus temporary bytes for one compiled executable and one memory space.
- **EF:** Exponential family; the scalar natural-parameter arena used by conjugate factors.
- **FLOP:** One floating-point operation counted by the backend compiler's cost model.
- **Flooding step:** One synchronous update of every rule group through `Engine.step`.
- **GFE:** Generalized free energy, including P-substituted objective plans and their derived runtime state.
- **Hybrid model:** A graph whose factor plans use more than one message arena.
- **LGSSM:** Linear Gaussian state-space model.
- **Persistent bytes:** The exact `nbytes` sum of arrays retained in an initialized `State` between inference steps.
- **Pytree path:** The route returned by JAX for one array leaf inside nested dataclasses, dictionaries, tuples, or lists.
- **RunStatus:** One of `CONVERGED`, `BUDGET_EXHAUSTED`, `DIVERGED`, or `INFEASIBLE`.
- **Shape-segmented arena:** Flat storage in which each logical slot occupies its respective cardinality- or dimension-sized segment.
- **Temporary bytes:** Buffers in the named memory space that XLA's compiled plan creates for one executable and does not retain in `State`.
- **Warm execution:** A synchronized invocation after compilation and one completed first execution.
- **Work unit:** The executable operation named by a cost or timing record, such as one flooding step or one exact two-pass run.
- **XLA:** The compiler used by JAX to lower and optimize array programs for a backend.

## Notes

**2026-08-08T04:59:29Z**

Intelligible-ticket review retained 34 scan candidates because their local meanings are bounded: `n_a` means checkpoint and bundle implementation are structurally irrelevant for this derived report; `preserve` means compatibility of the named `message_bytes` property; `public` means the README-documented `Estimate` report; `reusable` means one compiled executable is shared by analysis and calibration; `same` means byte equality, argument-order equality, or an equal caller iteration value in the stated sentence; `expose` means the LGSSM revealed the defect or a backend provides an analysis API; `implement` means work required by this ticket or its dependency boundary; `provide` means a backend returns its analysis data; `support` means the installed backend accepts cost analysis; `complete` means the exhaustive input or pytree named by the sentence; `ready` means JAX output synchronization has completed; `remaining` means a named cff-44we property or diagnostic visibility continues unchanged; `approximate` means XLA byte traffic is a compiler-derived quantity rather than allocated memory; `close` means administrative ticket closure after acceptance; `check` means the separate capability-ownership audit; `converge` names the explicit runtime status; `correct` in the title means repair of the obsolete step call; `equivalent` means alternate artifacts proving the numbered outcomes; `estimate` means the CFFG resource estimator except where XLA byte traffic is explicitly identified as a compiler approximation; `evidence` means named CPU results, tests, or documentation; `exact` means literal `nbytes`, shape-segmented storage, or the exact two-pass schedule kind; `pass` means an exact two-pass algorithmic traversal or automated test success according to context; `robust` means the repository hostile-input contract; `test` means the named automated fixtures or selections; `infer` means model-state inference at supplied parameters; `performance` is the preserved ticket tag covering memory and latency measurements; `scale` means increased domain size or Colony batch size; `small` identifies the binary/scalar population in the cff-44we hostile fixture; `explicit_owner` and `named_ticket_handoff` identify the cff-j7mm, cff-44we, and cff-6s31 boundaries stated after the mapping; `at_least` is an inclusive lower bound of five repetitions or batch size 128; `every` distributes over the explicitly named state leaves, output leaves, or rule groups; `only` marks a discrete-only fixture or excludes silent `n/a` from the diagnostic; and `read` means memory traffic loaded by the compiled executable. Capability ownership was checked separately: cff-j7mm owns compiled-engine persistent bytes, XLA cost and memory records, structured diagnostics, scheduled and flooding calibration, and batch accounting; cff-44we owns the closed shape-segmented storage prerequisite; cff-6s31 owns downstream analytic counts from compact plate dimensions before compilation.

Intelligible-ticket review: complete.
