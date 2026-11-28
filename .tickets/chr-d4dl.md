---
id: chr-d4dl
status: in_progress
deps: []
links: [cff-wbhu]
created: 2026-11-25T06:41:38Z
type: task
priority: 2
assignee: coopwilliams
tags: [performance, colony]
---
# Reach 8× ordinary Colony throughput on the frozen 32-member CPU Hallway benchmark

On the 64 GB Apple M2 Max, moving ordinary iteration onto the device cut one 32-member Colony run from 9.550 s to 2.568 s. The same 32 compiled Engine calls take 2.466 s sequentially. Cooper retained the 8× within-revision throughput target as a separate performance outcome after accepting the controller and its numerical checks. This task can proceed independently of M7.

## Design

Start with tools/benchmark_ordinary_iteration.py at b2bbcb3, its qualified runtime 8ae5bdb and pristine baseline 51ae6a7. Separate cold compile/first action, synchronized warm public wall, actual PJRT submissions, memory and host concurrency. Preserve every member and all 60 updates in the timing denominator. Capture device execution for the complete 1- and 60-sweep runs and mark truncated profiler output as incomplete. Profile exact shared input, scheduler and data layout improvements independently, with representative smaller and Broadcast control shapes. Investigate costly dense reductions before selecting an implementation. Evaluate alternatives in isolated candidates before changing defaults. Exact, mixed trust, stochastic and GPU throughput are separate workloads.

## Acceptance Criteria

On the frozen 32-prior, 60-sweep non-GFE Hallway workload, the median time of 32 public sequential Engine runs divided by the median time of one public Colony run is at least 8. Use the same initial states, objective, damping, fixed work, dtype, default CPU configuration and hardware for both arms. Show no single-engine regression against the qualified compiled baseline. Preserve all statuses, local failure attribution, counters, performed delta and defined-energy traces, final states and decoded or posterior readouts against the established parity checks. Keep the short-suite numerical and failure-isolation tests green. Report the achieved ratio and resources, including any remaining shortfall.


## Notes

**2026-11-25T06:41:53Z**

Change path: Runtime numerical or status invariant in docs/ARCHITECTURE.md. Compare public Engine and Colony, sequential and batched scheduled sweeps, per-member support/FE validation, status, state, host result reconstruction and synchronized timings. Model and checkpoint schema changes are outside scope. The 320-member/600-sweep fixture, float32 and float64, exact zeros, vacuous inputs, early failure and late failure must remain covered. Use the archived paired receipts in ~/agent-share/cffg-hierarchical-runtime/ordinary-comparison.json. Standalone 8-thread Engine probing reached 3.33× versus serial with bitwise matching public fields. Its queueing, 194 ms p95 in-service latency and peak RSS are part of its operating tradeoff. Reproduce before selecting an execution strategy.

**2026-11-25T06:43:58Z**

Evidence for selection: on 32 frozen Hallway priors a width-8 exact device tile reduced public 60-sweep wall from 2.523 to 2.301 s and compiler-reported scratch from 153.25 to 50.83 MB. Eight-member direct public path measured 0.566 s in both arms. The small three-category/32 case regressed from 3.106 to 3.567 ms. The frozen 128-member balanced Broadcast run improved from 50.57 to 68.01 images/s with the same per-image counts and maximum posterior error 4.846e-13. Hallway’s worst FE trace difference was 5.69e-14. Full-width dynamic parameter sharing slowed the raw 12-sweep group; optimized CPU HLO already forwards static CPTs and invariant finite masks through the loop. Current default remains 8ae5bdb. Raw public/diagnostic receipts, scripts and optimized HLO: ~/agent-share/cffg-hierarchical-runtime/ordinary-diagnostics. A separate threaded public Engine example reached 3.33× over sequential, using additional active cores and with higher per-member p95 latency; see ordinary-thread-probe/measured/summary.json.

**2026-11-25T06:43:58Z**

Legend: prerequisite --> dependent; parallel branches have no dependency on each other.

[x] cff-j4mu, cff-d56d, cff-44we --> [in progress] cff-wbhu --> [ ] cff-9xmq / cff-5edo
[ready] chr-d4dl (P2 independent throughput qualification, linked to cff-wbhu)
[x] cff-mjku --> [ready] cff-pyy9 (M7)

The existing wbhu prerequisites are resolved. The linked performance target introduces no blocking edge into the roadmap or M7. Verify with tk dep tree --full, tk ready, tk blocked and tk dep cycle.

**2026-11-28T05:01:20Z**

Architecture change path: Runtime numerical or status invariant. The isolated timing probe reuses public Engine and Colony, the existing synchronous compiled loop, result reconstruction, and decoded marginal readouts. It changes the local transition contraction, atom-level Bethe term and device support predicate together for the candidate. The stock scalar and scheduled executors and host support checks remain the comparators. Serialization and checkpoints are outside the experiment. The frozen Hallway transition has 495 supported atoms of 18,000 per slice. Keep the finite log-zero floor for outgoing rows and the existing probability cutoff for every joint atom. Qualify float32 and float64, and retry through the unmodified dense controller if the guarded input range or terminal checks fail. No default runtime change is proposed before results and control-shape qualification.

**2026-11-28T07:22:12Z**

The isolated indexed-transition probe at 207a3da met the Hallway throughput target on the exact 32 members and 60 sweeps. Three synchronized CPU x64 repetitions on a 64 GB M2 Max gave medians of 2.46107 s for public sequential Engine, 2.61228 s for stock public Colony and 0.21736 s for Colony with the indexed transition hooks, a ratio of 11.32× versus sequential. No member retried with the dense controller. At one through 60 sweeps, the maximum scaled free-energy and final-message differences were 9.99e-14 and 6.36e-16, decoded probability absolute error was 4.44e-16, and all delta traces, statuses, iteration counts and exact support agreed. Local float32 and float64 checks were bitwise. Cold indexed 60-sweep time was 1.038 s versus 3.267 s stock Colony, including 7.34 ms to build the indexed tables. Receipt: ~/agent-share/cffg-backlog-gates/hallway-indexed-207a3da.json. This diagnostic injects the local hooks during tracing. The production runtime, small and Broadcast controls, complete device-submission traces and full suite remain separate release qualifications. Keep this ticket in progress for production integration.
