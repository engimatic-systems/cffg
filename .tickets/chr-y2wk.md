---
id: chr-y2wk
status: closed
deps: []
links: [chr-odhx]
created: 2026-11-25T06:05:45Z
type: task
priority: 1
assignee: coopwilliams
parent: cff-wbhu
---
# Compile and qualify mixed-GFE fixed-support controller phases

Mixed generalized-free-energy runs retain a host decision per numerical iteration. Move fixed-support work behind a reusable device program while keeping the current undamped root, stationary free energy, inner status and exact support as the certification authority. Compare single and batched outcomes and record costs.

## Design

Architecture change-path: runtime numerical/status change. Compare Engine, Colony and the host trust reference. State and checkpoint format are unchanged. Preserve dynamic parameters, support discovery, phase boundaries, failure facts and each member’s counters. A fixed support is the set of feasible labels on each message. A trust step limits its movement in reduced message coordinates.

## Acceptance Criteria

Use targeted boundary, inner-failure, scale and float32 tests. Measure warm/cold, representative budget and batch footprint. Document any controller stage whose qualification remains open. Larger runtime integration and final suite belong to cff-wbhu.


## Notes

**2026-11-25T06:34:29Z**

Cooper chose the existing host trust controller as the default for Engine and Colony. Select outer_solver="compiled_trust_region" for deterministic mixed GFE with repeated or batched workloads. The exact support face, frozen chart metric, radius, inner status, counters and acceleration history belong to each member. The compiled stages produce candidates and trust steps. The host retains support restarts, hardening and final undamped-map / ledger certification, and can finish a phase after an invalid proposal. Adaptive Gaussian and functional updates keep their previous controllers. JAX specializes stage executables by chart, input layout, batch width and trace capacity. No GPU measurement or one-submission bound on exceptional paths is claimed.

**2026-11-25T06:34:29Z**

Measured public CPU float64, tolerance 1e-10, no clipping. Reproduction driver, 16 first/repeat records covering eight fresh processes, and hash-verified snapshots of all seven measured controller modules are in ~/agent-share/cffg-hierarchical-runtime/mixed-controller/{benchmark.py,public.jsonl,measured-sources/}. On 32 T-maze lanes with different priors and budget 10, auto/compiled first calls 10.875/17.424 s, repeats 2.582/0.501 s, peak process RSS 865/1369 MiB. All 32 agree on status, iterations, accepted and trial counts, max absolute ledger difference 5.33e-15, max residual difference 4.03e-15. The optional controller counted one extra map for boundary certification. One-update factored corridor (budget 400) returned the same root and free energy, with 8 accepted steps, 11 trials, 7680 linear iterations; first 10.402/13.734 s, repeat 1.148/0.736 s, peak 1043/1364 MiB. Default built zero optional stages on all runs. The 240-budget T-maze candidate timings predate the final independent candidate sweep and are retained as development evidence. The last sweep changes candidate-root exits only. The large trust compilation has a measurable first-call and memory cost.

**2026-11-25T06:35:11Z**

Final focused gate passed: OPENBLAS_NUM_THREADS=1 uv run --frozen --no-sync pytest -q -o addopts= tests/test_mixed_controller.py tests/test_outer_gfe_formalism.py tests/test_outer_failures.py -k "not float32_reactor_returned_state" (41 passed, 1 deselected in 86.63 s). New public tests exercise actual accepted trust steps in f32/f64, a warm member finishing early, different exact support, an isolated failed lane, and standalone rejection of a forged candidate. The excluded adaptive test fails identically at unchanged 51ae6a7 before its numerical assertions; see chr-odhx. Checked ticket cycles and diffs. Parent owns the shared final short-suite gate.
