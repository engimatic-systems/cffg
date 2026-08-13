---
id: cff-yl0a
status: open
deps: [cff-d56d]
links: [cff-wbhu]
created: 2026-07-27T05:44:29Z
type: feature
priority: 2
assignee: coopwilliams
tags: [performance, scheduling]
---
# Chain slice fusion: composite scan kernels to close the schedule 20x gap

## Background in Plain English

Ticket cff-d56d already made exact inference on a chain use one inward and one outward pass, compiled periodic rounds into `ScanStage` objects, carried cross-step messages in scan registers, and certified the result with status, support, residual, and Bethe-free-energy checks. Its corrected benchmark showed that the public CFFG kernel is often within the original 20x envelope of a matched full-posterior HMM smoother, but the hostile 256-step, two-state chain still measured about 27.6x against the stricter minimal-output reference. The earlier comparison of a K=8 CFFG chain with the recorded 0.0198 ms reference did not use matching shapes; this ticket uses the corrected K=2 model convention and retains 0.0198 ms only as the recorded strict-protocol target.

The remaining cost is inside each scan step. The generic registerized executor separately gathers, pads, evaluates, normalizes, and records the transition, equality, and emission factor outputs, while a dedicated HMM smoother performs a smaller composite recurrence. This ticket recognizes only the exact discrete HMM slice already represented by a registerized `ScanStage` and lowers that slice to one role-aware composite scan body without weakening the CFFG message, objective, or runtime-status contracts.

The strict low-level reference is an optimization ceiling, not an equivalent public API: it precomputes emission likelihoods and projects only the requested smoother output inside JIT, whereas CFFG preserves every directed message needed by its public inference and ledger machinery. Benchmark evidence therefore keeps the strict minimal-output reference and the matched full-posterior reference in separate columns. This ticket does not reopen schedule selection, change `Schedule.describe()`, vectorize public readouts, or introduce an LGSSM fast path.

## Outcomes / Acceptance Criteria

1. The scheduled executor has a bounded HMM-chain fusion path.
   1.1. The fusion recognizer selects a registerized discrete `ScanStage` only when its compiled group types, role order, cardinalities, row dependencies, and transition/equality/emission repetition equal the supported HMM slice contract.
   1.2. A structurally different stage selects the generic scheduled executor implemented by `_registerized_scan_fn` or `_flat_scan_fn`.
   1.3. A stage containing an unsupported factor family selects the generic scheduled executor implemented by `_registerized_scan_fn` or `_flat_scan_fn`.
   1.4. A max-product stage selects the generic scheduled executor implemented by `_registerized_scan_fn` or `_flat_scan_fn`.
   1.5. An automated test selects the generic executor for a recognized HMM stage so that fused and unfused execution remain independently comparable.
   1.6. Each fused scan step computes the role-specific directed outputs of the transition, equality, and emission factors as one composite JAX scan body.
   1.7. The fused scan writes every directed message slot that the unfused `ScanStage` writes.
   1.8. The fused scan performs no Python dispatch per time step.
   1.9. The fused scan carry uses O(K) numeric storage for state cardinality K.
   1.10. The required posterior-message output uses O(TK) storage for chain length T and state cardinality K.
2. Fusion is an execution detail rather than a new schedule kind.
   2.1. `Schedule.describe()` for a fused model equals its value when fusion is disabled.
   2.2. `Schedule.fingerprint` for a fused model equals its value when fusion is disabled.
   2.3. Template serialization and reconstruction select the same fusion decision for the same arena structure.
3. The fused path preserves the sum-product inference result under explicit dtype-scaled tolerances.
   3.1. Every final directed discrete message from fused and unfused scheduled execution agrees using relative and absolute tolerances of `128 * finfo(dtype).eps` after accounting for the existing log-message gauge.
   3.2. Every posterior marginal from fused and unfused scheduled execution agrees using relative and absolute tolerances of `128 * finfo(dtype).eps`.
   3.3. The Bethe free energy from fused and unfused scheduled execution agrees using relative and absolute tolerances of `256 * finfo(dtype).eps` times the number of chain slices.
   3.4. Fused and unfused scheduled execution return the same `RunStatus`.
   3.5. Fused and unfused scheduled execution return the same iteration count.
   3.6. The fused fixed-point certificate satisfies the same elementwise dtype-scaled residual bound as the unfused exact schedule.
   3.7. Gradients of the final Bethe free energy with respect to the prior, transition probabilities, and emission probabilities agree between fused and unfused execution using a stated dtype-scaled tolerance.
   3.8. Fused scheduled marginals agree with converged synchronous flooding marginals using the tolerance in criterion 3.2.
   3.9. Fused scheduled Bethe free energy agrees with converged synchronous flooding Bethe free energy using the tolerance in criterion 3.3.
4. Hostile inputs retain the runtime's numerical meaning.
   4.1. Literal log-zero entries in sparse transition and emission tables remain outside probability support; the fused path introduces no probability floor.
   4.2. An HMM whose observations make global support empty returns `RunStatus.INFEASIBLE` through the existing exact-unit support check.
   4.3. Vacuous incoming messages act as multiplicative identities and match the unfused result.
   4.4. An incoming message with no supported category does not produce a finite converged result.
   4.5. Float32 fusion returns float32 message outputs.
   4.6. Float32 fusion preserves literal log zeros.
   4.7. A supported float32 fixture produces finite marginals that satisfy criteria 3.1 through 3.3 using float32 epsilon.
   4.8. Non-finite propagated messages, residuals, or Bethe free energy return `RunStatus.DIVERGED` rather than `RunStatus.CONVERGED`.
   4.9. A K=2, T=1024 identity-dynamics chain with only its final state observed completes one fused exact run without `RunStatus.DIVERGED`.
   4.10. The T=1024 result in criterion 4.9 has finite supported marginals.
5. The strict HMM kernel gap closes under a reproducible benchmark.
   5.1. The target fixture is a K=2, T=256 identity-transition HMM with a uniform initial state and only the final state observed, using the corrected `B @ d` initial convention and the same column-stochastic transition and emission matrices on both sides.
   5.2. On an Apple M2 Max CPU with JAX x64 enabled, the fused warm propagation kernel has median latency no greater than 0.396 ms, which is 20 times the recorded 0.0198 ms strict reference.
   5.3. On the same benchmark run, the fused warm propagation kernel has median latency no greater than 20 times the live strict minimal-output reference.
   5.4. The strict minimal-output reference precomputes HMM emission likelihoods, projects its requested output inside JIT, and is labeled as an optimization ceiling.
   5.5. The matched full-posterior reference returns the state-posterior sequence that is numerically checked against CFFG.
   5.6. The benchmark reports compile, schedule construction, cold JIT, unfused warm propagation, fused warm propagation, fused status-bearing run, public readout, and flooding as separate phases.
   5.7. The benchmark reports the matched full-posterior and strict minimal-output references in separate columns.
   5.8. Each warm latency is the median of at least 40 blocked repetitions after at least 20 untimed warmups in one process.
   5.9. The benchmark artifact records hardware, JAX platform, dtype mode, model dimensions, output projection, and repetition counts.
   5.10. The benchmark reports fused and unfused propagation for T in {32, 64, 128, 256} so its scaling rows expose a chain-length regression.
6. Focused automated checks cover recognition, fallback, message and marginal parity, Bethe-free-energy parity, status and certificate parity, gradients, sparse support, vacuous and empty support, float32, and the T=1024 scale case.
7. Documentation agrees with the delivered fusion boundary.
   7.1. `docs/ARCHITECTURE.md` is updated in place if the delivered executor choice changes its claim at implementation start.
   7.2. `README.md` is updated in place if the delivered benchmark command changes its claim at implementation start.
   7.3. `docs/LIMITATIONS.md` is updated in place if the supported HMM boundary changes its claim at implementation start.

## Methodological Suggestions

These are suggested ways to reach the outcomes. An implementer may choose another method if it satisfies every numbered criterion with equivalent evidence.

Use `src/cffg/engine.py:_scan_stage_fn` as the executor-selection seam. `src/cffg/schedule.py` already discovers periodic rounds and constructs deterministic `ScanStage` row tables, while `_classify_scan` already proves which values are static, same-step, or previous-step dependencies. Match the supported slice from compiled group metadata and those dependency plans rather than from factor names. Keep `_registerized_scan_fn` as the correctness oracle and fallback.

Derive the composite body by algebraically composing the registered `matrix_cat` and `equality` sum-product updates. Preserve their log-space normalization, literal `NEG` support representation, role order, and directed outputs. A likely implementation uses one `jax.lax.scan` body for the HMM recurrence and scatters its stacked role outputs into the same arena slots after the scan. Do not omit messages merely because the strict comparator does not return them, and do not bypass `_make_exact_unit` support checks, fixed-point certificate, or Bethe-free-energy ledger.

The applicable `docs/ARCHITECTURE.md` change-path row is **Runtime numerical or status invariant** because the optimized scheduled path must continue to agree with status, support, certification, and ledger semantics. No new IR field, factor type, serialization field, learnable parameter kind, or checkpoint format is intended. Review the deliberate execution pairs as follows:

- Host/JIT: the fused path is JIT execution; compare it with the independently selectable generic path and scalar forward-backward references.
- Flooding/scheduled: scheduled execution changes, while flooding remains an independent fixed-point oracle.
- Single-instance/`vmap`: exercise the fused exact unit both directly and through an existing transformed or Colony path if that path accepts exact schedules; otherwise record the concrete reason this pair is not applicable before implementation closes.
- Current/legacy checkpoint decoding: not applicable because fusion changes neither `State` layout nor persisted data.

Extend the focused schedule tests rather than creating a second schedule test vocabulary. Add a benchmark mode to `src/cffg/benchmarks/scheduling.py` that times fused and deliberately unfused kernels with identical states and blocking. Implement the strict low-level HMM comparator as a separately labeled function so dead-code elimination and precomputed likelihoods are visible in review. Preserve raw benchmark output or a compact ticket note containing the measurements and environment. During implementation, run focused schedule and benchmark checks as often as useful; after the final code change, run the dashboard-backed short suite exactly once as required by `AGENTS.md`.

## Future work

- cff-3kdk owns vectorization and end-to-end latency of `Engine.marginals` and `Engine.gaussian_marginals`; this ticket reports public readout time but does not optimize it.
- cff-wbhu owns Python sweep-loop dispatch, JIT loop control, and Colony throughput for iterative runs; this ticket changes only the exact HMM `ScanStage` propagation body.
- No LGSSM fusion is delegated here. If product-facing LGSSM latency remains a priority after cff-3kdk, profile the corrected full output path and create a separate ticket before changing Gaussian scan lowering.

## Definitions

- **Bethe free energy (BFE):** The factor-local energy and entropy sum with edge-entropy overcounting corrections used by exact sum-product inference.
- **Composite scan body:** One `jax.lax.scan` step that evaluates the supported transition/equality/emission slice while retaining all required directed outputs.
- **Directed message:** The message written from one factor port toward the variable edge's opposite endpoint.
- **Dtype-scaled tolerance:** A comparison bound derived from `finfo(dtype).eps` and the explicitly named accumulation count.
- **Exact unit:** The JIT-compiled propagation, support, certification, and BFE calculation used by the two-pass runtime.
- **Fusion:** Replacement of several algebraically composable factor-rule calls inside one scan step by one specialized calculation with the same observable inference result.
- **Generic executor:** The existing `ScanStage` implementation that executes registered factor rules without HMM-specific algebraic composition.
- **HMM:** Hidden Markov model, here a discrete chain with an initial distribution, state transition matrix, and per-state emission probabilities.
- **K:** The hidden-state cardinality.
- **Literal log zero:** The runtime's reserved `NEG` representation of probability zero.
- **Matched full-posterior reference:** A separately implemented HMM smoother timed while returning the state-posterior sequence that is numerically compared with CFFG.
- **Registerized scan:** A `ScanStage` executor whose scan carry contains only messages that cross time steps and whose flat-arena writes occur after the scan.
- **Role-specific output:** The separate outgoing message for each factor port, retained even when a reduced reference implementation would omit it.
- **ScanStage:** A deterministic schedule stage containing repeated factor-group rows executed through `jax.lax.scan`.
- **Strict minimal-output reference:** A low-level HMM recurrence with emission likelihoods prepared before the timed call and only one requested smoother output projected inside JIT; it is a performance ceiling, not the CFFG public contract.
- **T:** The number of HMM transition/emission slices.
- **Warm propagation kernel:** The already-compiled scheduled propagation call timed with device completion and without status, certificate, BFE, or public readout work.


## Notes

**2026-07-27T14:13:01Z**

Scope update after the 2026-07-27 review round: the fused status path plus live matched-shape referencing brought every benchmark configuration inside the ticket's 20x envelope (hostile K=2 run 13-19x, ordinary K=8 7-8x, LGSSM d=4/m=2 17-19x), so slice fusion is no longer an acceptance blocker for cff-d56d. It remains the right lever if the envelope must tighten (LGSSM mid-sizes sit at ~19x with little margin) or if the stricter recorded reference figure (~2x below live) becomes the standard.

**2026-07-27T15:47:43Z**

BOUND AFTER cff-d56d CLOSURE (2026-07-27). This is the strict-kernel continuation, not a reopening of the scheduling P1. Keep the public matched-model/full-posterior reference and the minimal-output low-level reference as separate reported columns: the latter projects only the requested smoother output inside JIT (and precomputes HMM likelihoods), so it is an optimization ceiling rather than the same output contract. Scope stays HMM chain lowering: role-specific directed outputs plus fused transition/equality/emission scan, with generic-vs-fused marginal/FE/status/gradient parity and hostile-input regressions. Do not absorb readout vectorization (cff-3kdk) or a speculative LGSSM fast path; profile and file the latter separately only if the product-facing LGSSM envelope remains a priority after readout is fixed.

**2026-08-08T04:35:45Z**

Retained wiggle candidates after the bounded current scan (candidate -> intended meaning): end_to_end -> the complete public readout call from a converged State to its returned dictionary; fixed -> defect_repaired in the preserved historical note; generic -> the named non-HMM-specific ScanStage executor; preserve -> value_or_semantic_invariant; public -> documented Engine or benchmark boundary; register -> scan-carried storage or executable registered factor rules as named by the sentence; same -> value equality, identical benchmark run, or shared artifact as named; available -> named evidence present for reviewer inspection; expose -> benchmark scaling rows make a chain-length regression visible; implement -> planned construction or repository function; support -> literal probability support or documented capability boundary as named; complete -> execution_finished or device_completion; deliver -> repository artifact produced by this ticket; remaining -> the unfinished per-step latency gap; close -> latency-gap reduction or historical ticket closure as named; certify -> the automated fixed-point certificate; check -> automated numerical, support, or review check as named; converge -> RunStatus convergence criterion; correct -> corrected benchmark convention or proven executor behavior; equivalent -> equal observable API or execution behavior; evidence -> named tests, benchmark rows, or preserved measurement artifact; exact -> two-pass schedule/unit, literal support, or mathematically exact BFE semantics as named; fixed_point -> messages invariant under one certification sweep within the dtype-scaled bound; match -> identical model shapes, recognizer equality, or tolerance agreement as named; oracle -> independently implemented numeric reference; parity -> agreement under the criterion-specific tolerance or status equality; pass -> algorithmic inward/outward traversal or tests_succeed as named; reproduce -> deterministic reconstruction or repeatable benchmark protocol; test -> automated executable check; accumulate -> the explicitly named number of BFE summands; infer -> fixed-parameter state posterior calculation; learnable -> the declared persistent-parameter feature category; performance -> measured warm kernel latency; scale -> dtype-relative numeric bound, T domain size, or measured chain-length behavior as named; small -> lower per-step operation count; speed -> runtime latency in the preserved historical note; can -> demonstrated capability of an automated check or evidence available to a reviewer; intended -> declared ticket scope; may -> implementation permission; explicit_owner -> this ticket owns discrete HMM scan fusion while the named Future-work tickets own their stated exclusions; named_ticket_handoff -> cff-3kdk and cff-wbhu are concrete future owners; all -> the locally named required directed outputs; at_least -> inclusive warmup or repetition minimum; every -> each member of a named message, marginal, criterion, or benchmark set; only -> ticket scope, observation placement, or projected-output restriction as stated locally; current -> the named current/legacy checkpoint review pair; existing -> repository functions and contracts identified by path or symbol. Each retained occurrence is bounded by its sentence, symbol, artifact, metric, or historical note.

Intelligible-ticket review: complete.
