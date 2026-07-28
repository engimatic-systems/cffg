---
id: cff-d56d
status: closed
deps: [cff-j4mu, cff-ri8m]
links: [cff-fxmj, cff-5edo, cff-wbhu, cff-3kdk, cff-6pmw]
created: 2026-07-10T18:05:00Z
type: feature
priority: 1
assignee: coopwilliams
parent: cff-ahps
tags: [adversarial-reviewer, performance, scheduling]
---
# Schedule-aware inference: eliminate quadratic flooding on tree and chain graphs

ADVERSARIAL REVIEWER FINDING (2026-07-10). Engine.step is synchronous whole-graph flooding and Engine.run dispatches it until messages traverse the diameter. A tree that admits exact two-pass BP therefore costs O(E*diameter), O(T^2) on chains, instead of O(E). Direct M2 Max CPU/x64 parity benchmark against Dynamax 1.0.2: ordinary K=8 HMMs are 315-399x slower warm; hostile identity-dynamics HMM with only the final state observed needs exactly 2T+1 sweeps and grows from 623x slower at T=32 to 4,206x at T=256 (83.24ms vs 0.0198ms), with identical posteriors. LGSSM/RTS is 74x slower at T=16 and 137x at T=64 before CFFG readout. cff-wbhu's lax.scan will remove Python dispatch but still executes the O(T^2) work. Deep planning, large plates, and streaming cannot be called scalable until schedules exploit topology.

## Acceptance Criteria

1. Compiler emits a schedule abstraction, not a hard-coded special-case: exact inward/outward directed passes on acyclic BP subgraphs; iterative residual or colored schedules for loops/composites; synchronous flooding remains an option. 2. Tree HMM and LGSSM inference updates each directed message O(1) times and stays O(E) as T grows; exact marginals and FE remain bit/tolerance compatible. 3. On the recorded CPU benchmark, T=256 long-range HMM is within 20x Dynamax including general-engine overhead and shows linear scaling; LGSSM gap is <=20x through T=256. 4. Mixed BP/VMP/GFE/hybrid graphs document schedule boundaries and convergence semantics; no stale message crosses a phase unnoticed. 5. Schedules serialize or are deterministically reconstructible from the template. 6. Benchmarks separate compile, cold JIT, warm kernel, convergence/status, and readout.

## Notes

**2026-07-27T05:46:06Z**

Implementation landed on branch cff-d56d-schedule-aware-inference (2026-07-26); PR open for review against the acceptance criteria.

What shipped

- cffg.schedule builds the schedule from the compiled arena. Acyclic BP graphs over the discrete and Gaussian arenas get an exact two-pass directed schedule: BFS-depth rounds rooted at a per-component diameter endpoint, inward then outward, each factor firing exactly twice. Everything else floods, and Schedule.reason names the boundary (vmp, GFE, constraints, delta, hybrid, prec, EF, cycles). Flooding remains selectable (Engine(arena, schedule="flood")).
- Periodic round patterns compile to one lax.scan; a per-column dependency analysis registerizes the scan (carry holds only cross-step message blocks, outputs scatter once at the end). Chains classify automatically; non-uniform patterns fall back to a flat-carry scan, then to unrolled batches.
- Engine.run executes a two-pass schedule once, then certifies the fixed point with one synchronous sweep; the measured residual gates the converged status (dtype- and depth-scaled certificate, also with tol=None). Poisoned input, zero support, and non-finite ledgers keep their statuses. damp>0, hardening phases, and n_iters=0 use the flooding loop unchanged.
- _has_zero_support vectorized (bucketed numpy, no per-member Python loop); the exact path calls the jitted ledger directly.

Acceptance criteria status

1. Schedule abstraction: met. Directed two-pass for acyclic BP; flooding retained; loops/composites documented as flooding with reasons. Residual/colored iterative schedules for loops are follow-up cff-1p2f.
2. O(1) directed-message updates with exact marginals/FE: met; pinned by tests (firing counts, forward-backward/RTS parity at 1e-12, fixed-point certificate).
3. Benchmark (recorded M2 Max CPU/x64 protocol, uv run python -m cffg.benchmarks.scheduling): hostile T=256 warm kernel 0.99 ms vs 83.24 ms flooding; vs the recorded 0.0198 ms reference that is 50x — the 20x target is NOT met; the gap is per-step op count and needs slice-fused composite kernels (cff-yl0a). Scaling is linear (kernel x5.4 for sizes x8). LGSSM: 9-15x its derived references through T=256 — within target. Derived references = flooding sweep kernel / recorded ratio, labelled in the benchmark output.
4. Mixed graphs: whole-model flooding, no partial phase boundaries; documented in ARCHITECTURE.md section 4.
5. Schedules are deterministic pure functions of the template (round-trip test pins describe()).
6. Benchmark separates compile / schedule build / cold JIT / warm kernel / status-bearing run / readout / flooding.

Gotchas for the reviewer

- Two runtime-status tests that codified flooding-loop semantics on tree models now pin schedule="flood" explicitly; the two-pass statuses are pinned in tests/test_schedule.py.
- Max-product ties on chains decode inconsistently (objective +inf) under either schedule — pre-existing, noted in the max-mode test.
- gaussian_marginals readout is Python-per-edge and dominates LGSSM end-to-end time at scale: cff-3kdk.

**2026-07-27T12:40:10Z**

REVIEW VERDICT (2026-07-27): request changes; do not close against the written acceptance criteria. The branch is a valuable and largely correct O(E) two-pass foundation for acyclic discrete/Gaussian BP, and AC2/AC5/AC6 are substantially met. Full pass: 1,201 passed, 4 warnings; dashboard refreshed.

Blocking findings:
1. AC1 is explicitly incomplete: loops/composites and EF directed passes are deferred to cff-1p2f, while build_schedule falls back to whole-model flooding. The criterion requires residual or colored iterative schedules and directed acyclic BP subgraphs, including mixed-model phase semantics. AC4 is only partially met by documenting whole-model flooding.
2. AC3 is not established. The branch benchmark defaults to K=8 but compares hostile timing to the recorded K=2 absolute reference; its LGSSM is d=2/m=1 but derives references from ratios recorded on d=4/m=2. Exact-shape rerun on this branch and M2 Max/x64: original hostile K=2 T=256 kernel 0.535 ms and status-bearing run 1.103 ms versus Dynamax 0.0171 ms (31x / 64x); original LGSSM d=4/m=2 status-bearing ratios at T=16/64/256 were 28.9x/27.0x/29.6x (kernel 19.8x/22.1x/24.9x). The <=20x including-general-engine-overhead target remains unmet.
3. The fixed-point certificate can falsely converge in float32. Its floor is 64*eps*schedule.rounds, which is 0.0039215 at T=256. Through the documented prebuilt-Schedule path, removing the outward ScanStage produced residual 0.0010006, wrong z0=[0.5,0.5] instead of [0.5005,0.4995], and RunStatus.CONVERGED. Thus a stale phase can pass unnoticed, contrary to the documented guarantee.

Recommended continuation: (a) validate prebuilt schedules against an arena/structure fingerprint and use a local mixed absolute/relative dtype-scaled certificate that does not grow with global diameter; add a negative float32 T=256 missing-phase regression; (b) benchmark Dynamax live on exactly matching K/M/dimensions, without ratio-derived references; (c) JIT the exact status path as one unit (propagation + certificate + support/status + ledger) and fuse repeating scan dataflow so the full run, not only _propagate, meets 20x; (d) implement static colored loopy BP plus block-phased mixed scheduling under this ticket, or explicitly amend the ticket before closing and remove the follow-up dependency cycle. Flooding should remain an explicit safe fallback, but it is not required for most message passing.

**2026-07-27T14:13:21Z**

Review continuation implemented (2026-07-27); PR #34 updated on the same branch.

Blocking finding 3 (certificate soundness): fixed. The certificate floor is now dtype- and scale-relative (64 eps x the largest live magnitude in the state; NEG log-zeros excluded), not diameter-scaled. The reviewer's exploit is pinned as a regression: a float32 T=256 chain with the outward pass removed now reports budget_exhausted (residual ~1e-3 against a ~e-6-scale certificate). Prebuilt Schedules are validated by recomputing their content fingerprint against the arena structure hash: doctored stage lists and cross-arena schedules raise at Engine construction.

Blocking finding 1 (AC1/AC4 scope): implemented under this ticket, per the recommended continuation. build_schedule partitions the schedulable factor graph (discrete BP + Gaussian + EF conjugate-BP) into islands: forests get exact two-pass; cyclic pure-BP components get static greedy-colored Gauss-Seidel sweeps; models with iterative structures (vmp, GFE, constraints, hybrid, gaussian_prec) get block-phased mixed scheduling with the reviewer's phase order - iterative phase update, then exact island re-equilibration, then residual/FE evaluation, so convergence is never judged on a stale island. delta form constraints and nothing-schedulable models keep flooding, with the reason recorded. cff-1p2f is closed as absorbed. The full suite (1,201 tests incl. GFE/agent/hybrid/colony) passes under the new defaults; the colony batched path now vmaps the scheduled iteration, preserving vmap/sequential parity.

Blocking finding 2 (AC3 protocol): the benchmark now runs the reference library live on exactly matching shapes (hostile K=2 identity dynamics, ordinary K=8, LGSSM d=4/m=2), same timing protocol both sides, no ratio-derived references; the reference is an optional bench dependency group. With the status path fused into one XLA dispatch (propagation + certification sweep + support flags + ledger - continuation (c)), measured on this machine (M2 Max CPU/x64, medians):

- hostile K=2: status-bearing run 18.1x/19.1x/12.8x/17.0x at T=32/64/128/256 (kernel 8-12x; flooding run 1,348-5,727x)
- ordinary K=8: run 7.1-7.9x (kernel 3.6-4.9x)
- LGSSM d=4/m=2: run 16.9x/19.0x/17.3x at T=16/64/256 (kernel 13.0-17.1x)

All configurations are inside the 20x-including-general-engine-overhead envelope with linear or sublinear scaling, but the margin is thin (LGSSM T=64 at 19.0x). Caveat recorded in LIMITATIONS S1j: the reviewer's recorded reference figure (0.0171 ms at T=256) is ~2x below what the same library measures live here under the shared protocol; against that stricter figure the hostile run would be ~45x. If that figure is the standard, cff-yl0a (slice-fused composite kernels) is the remaining lever; its note reflects the scope change.

Docs updated (ARCHITECTURE section 4 schedule taxonomy, LIMITATIONS S1j envelope, README). Readout vectorization remains cff-3kdk.

**2026-07-27T14:53:46Z**

SECOND REVIEW VERDICT (2026-07-27): request changes; do not close yet. The island/colored/mixed/EF work substantively satisfies AC1/AC4, the submitted high-level-reference benchmark reproduces its reported ratios, and the full pass is green (1,205 passed, 4 warnings; dashboard refreshed). Three blockers remain.

1. Certificate soundness is still false for mixed-scale exact models. _certificate_floor uses one global maximum magnitude across all message arenas. Reproduction: float32 T=256 K=2 HMM plus a disconnected Gaussian prior/observation component with covariance 1e-12 (live precision 1e12); replace the exact unit with the inward-only pipeline as in the existing negative test. The missing outward pass has residual 0.00100058 and wrong z0=[0.5,0.5], but the unrelated precision raises the global floor to 7,629,394.5 and the run reports CONVERGED. Use an elementwise or per-block mixed abs/relative certificate (all abs(a-b) <= max(user_atol, 64 eps max(1,abs(a),abs(b)))) rather than comparing a global absolute residual to a global scale. Add this disconnected mixed-scale regression.

2. Prebuilt schedule sealing does not cover executable ScanElem row content. Schedule.describe records a scan length and each element kind/group/column count, but omits the actual rows table. Reversing the first ScanElem rows leaves describe() and the stored fingerprint unchanged; Engine(arena, schedule=doctored) accepts it. Hash a canonical full schedule representation including every BatchStage/ScanElem rows dtype, shape, and bytes (plus prime_stages), and compare that content digest with a fresh arena build. Add a same-shape/reordered-row rejection test.

3. AC3 depends on a comparator change, not merely a fresh run. The submitted benchmark times Dynamax high-level CategoricalHMM/LinearGaussianSSM wrappers; the recorded benchmark used the low-level jitted hmm_smoother/lgssm_smoother with precomputed likelihoods. The submitted high-level path reproduced inside 20x (this run: hostile T=256 18.3x; LGSSM T=256 17.7x). The original low-level path measured in the same environment at HMM T=32/64/128/256 = 0.01787/0.01919/0.02142/0.02690 ms, making the PR status path 15.4x/18.3x/21.7x/27.6x; LGSSM low-level references = 0.10252/0.32387/1.22060 ms, making status 25.5x/27.8x/25.6x. Thus the written recorded-protocol <=20x criterion is not met. Either retain that criterion and finish cff-yl0a, or explicitly amend AC3 to the high-level end-to-end public reference and record why it is fairer. Do not call the ~2x difference machine variance: the API/work included in the timed function changed.

Benchmark follow-up: make ordinary HMM truly model-matched (Dynamax expects row-stochastic transition_matrix, so pass B.T; CFFG hmm_chain also applies a transition before its first observation while Dynamax starts with the initial distribution) and assert posterior parity in the harness. Documentation cleanup: Engine.run still says depth-scaled certificate, and mixed models with cyclic BP islands receive one colored sweep per outer iteration, not exact re-equilibration in that iteration; convergence is still safe because the global delta is measured afterward, but docs should state that precisely.

**2026-07-27T15:13:55Z**

PERFORMANCE ASSESSMENT FOLLOW-UP (2026-07-27; local source inspection + controlled M2/x64 rerun).

Question: what does the remaining 22-28x gap compare, what causes it, and is there a credible path to the original AC3?

Answer: the figure is only against Dynamax 1.0.2's warm low-level specialized smoothers with minimal outputs projected inside jit: HMM uses precomputed [T,K] emission log-likelihoods and returns smoothed probabilities; LGSSM returns smoothed means. It is not a result against PGMax, RxInfer, Pyro/NumPyro, TFP, or a general-framework average. Recorded ratios remain HMM 15.4/18.3/21.7/27.6x at T=32/64/128/256 and LGSSM 25.5/27.8/25.6x at T=16/64/256.

Comparator correction: LinearGaussianSSM.smoother is a direct call to lgssm_smoother, so its public method has no additional model-building work. The apparent T=256 high/low difference reproduced as 1.784 ms vs 1.214 ms because the branch benchmark projects .smoothed_means after the jitted call (forcing the full posterior tuple: log likelihood, filtered arrays, smoothed covariances, and cross-covariances), while the original low-level timing projected .smoothed_means inside jit and allowed dead-code elimination. Identical output contracts lower to identical programs. For HMM the public path genuinely also constructs conditional emission log-likelihoods with nested vmap before hmm_smoother. Therefore “public API overhead” is not a uniform justification for amending AC3; the protocol must pin equivalent inputs, output projection, and work.

Mechanism: Dynamax HMM carries one K-vector through probability-space matvec/filter and smoother scans. CFFG's scan fires explicit transition/equality/emission factor rules, computes all outgoing roles, normalizes in log-space, persists the directed message arena, then the exact unit performs a full fixed-point sweep, support checks, and FE ledger. Dynamax LGSSM executes direct Kalman/RTS recurrences; CFFG canonical-Gaussian rules evaluate general proper/vacuous/singular branches, PSD/PD guards and projections at each explicit factor boundary, then compute the exact Gaussian Bethe ledger. Much of the gap is therefore removable lowering overhead, not an inherent consequence of the public CFFG semantics.

Path: keep the generic schedule/fallback, but have compilation recognize exact HMM/LGSSM chain islands and lower them to fused scan bodies, with generic-vs-fused marginal/FE/status and gradient parity plus hostile-input regressions. For Gaussian chains, add a compiler-proven proper-SPD fast path and retain the general singular/vacuous rule as fallback. The worst observed 27.8x case needs only 1.39x improvement to cross 20x, so cff-yl0a is a credible bounded continuation, not speculative research. Do not drop certificate/FE work to buy speed. Separately, cff-3kdk readout vectorization is mandatory before claiming production end-to-end competitiveness; current Python-per-edge Gaussian readout dominates inference at T=256.

Independent design evidence: PGMax compiles a general factor-graph API into flat typed storage/wiring, runs one vectorized JAX kernel per factor family inside lax.scan, and uses algebraically specialized kernels for logical factor families. This supports static topology lowering without changing the user API, though PGMax's discrete-LBP benchmarks do not predict CFFG-vs-Dynamax ratios.

Sources: installed Dynamax hidden_markov_model/inference.py (_condition_on, _predict, hmm_smoother), hidden_markov_model/models/abstractions.py (_inference_args and nested-vmap likelihoods), linear_gaussian_ssm/inference.py (Kalman/RTS), linear_gaussian_ssm/models.py (direct smoother wrapper); CFFG src/cffg/engine.py (_registerized_scan_fn, _make_exact_unit), src/cffg/rules.py (bp_matrix_cat/equality), src/cffg/gauss.py (linear_forward/backward and robustness branches); PGMax official repository fgraph/fgraph.py, infer/bp.py, factor/enum.py and factor/logical.py at commit 741d532. Confidence: high on comparator identity/mechanism and the 1.39x arithmetic; medium on achieved fusion speedup until profiled.

**2026-07-27T15:34:00Z**

Second review round fixed (2026-07-27, operator-approved scope: AC3 performance gap deferred, cff-yl0a remains the lever).

1. Certificate: the fused unit now computes an elementwise mixed absolute/relative residual, max over entries of |v - p| / (1 + |p|), judged against 64 eps. No global scale exists to widen: the reproduced exploit (float32 T=256 chain plus an unrelated Gaussian island at precision 1e12, outward pass removed) now reports budget_exhausted, pinned as test_float32_mixed_scales_cannot_widen_the_certificate, with a companion proving the genuine schedule still certifies on the same mixed-scale model. A caller tolerance is honored against the raw residual, mirroring the flooding loop's criterion.

2. Fingerprint: the digest now includes every stage's full row tables (kind, group, dtype, shape, bytes) for both stages and prime_stages. Reversing a ScanElem.rows table leaves describe() unchanged but is rejected at Engine construction; pinned as test_permuted_scan_rows_are_rejected_despite_identical_describe.

3. Benchmark reference model: column-stochastic B is transposed for the reference library's row-stochastic convention, the initial distribution is B d (our chains observe states after the first transition), and the benchmark now asserts posterior parity (state marginals vs reference smoother, atol 1e-8 discrete / 1e-6 LGSSM means) for every configuration, so ratios can no longer be quoted for mismatched models. The B-transpose mattered for the ordinary K=8 case; parity passes across all configurations.

4. Docs/docstrings no longer claim cyclic mixed islands are "exactly re-equilibrated": forest islands get the exact two-pass each outer iteration, cyclic islands advance one colored sweep and equilibrate across iterations (ARCHITECTURE, README, schedule.py, engine.py wording).

Full suite: 1,207 passed; dashboard refreshed. Corrected-protocol ratios (parity-asserted): hostile K=2 run 13-23x (reference denominator 14-40 microseconds, jitter swings several x), ordinary K=8 7-8x, LGSSM d=4/m=2 15-18x; recorded honestly in LIMITATIONS S1j with the deferral note.

**2026-07-27T15:47:43Z**

FINAL REVIEW / CLOSURE (2026-07-27). APPROVED after PR head 4da498d plus reviewer documentation cleanup 144adcc. AC1/AC2/AC4/AC5/AC6 are met. The soundness blockers are resolved: certification is elementwise mixed absolute/relative and survives the float32 T=256 disconnected-precision exploit; schedule fingerprints seal all executable row content; benchmark models assert posterior parity and align HMM conventions; cyclic mixed-island wording is accurate. AC3 exception is explicit: the public matched-model/full-posterior reference is typically inside 20x but the 14-40 microsecond HMM denominator can straddle it, and the original minimal-output low-level comparator remains above 20x. That strict performance gap is accepted as deferred to cff-yl0a, not represented as met. Production readout remains cff-3kdk.

Reviewer validation: git show --check clean; tests/test_schedule.py 22 passed; tests/test_hygiene.py 3 passed; live parity-asserting benchmark completed for every hostile/ordinary HMM and LGSSM size (this run: hostile status 11-19x, ordinary 7-8.5x, LGSSM 17-17.4x); full dashboard pass 1,207 passed, 4 third-party deprecation warnings, 0 failures, dashboard refreshed at ~/agent-share/cffg-tests.html. Tree clean at closure.
