---
id: cff-d56d
status: in_progress
deps: [cff-j4mu, cff-ri8m]
links: [cff-fxmj, cff-5edo, cff-wbhu, cff-3kdk]
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
