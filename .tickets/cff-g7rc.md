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

## Plain English

Tiny one-shot mixed-GFE models will retain the full soundness certificate while paying substantially less fixed proof and compilation overhead. The work will profile and fuse or defer fixed-cost machinery that repeated agent planning already amortizes. Repeated planning and the retained hard globalization case remain no-regression controls.

## Evidence and scope

PR #49 recovered routine planning throughput relative to the initial sound
implementation, but its like-for-like comparison with pre-soundness commit
`931758d` exposed the remaining fixed premium. At behavior commit `600a775`,
warm T-maze and D1 solves were respectively 3.71x and 26.12x slower than
before soundness, while ten repeated `RollingAgent` plans were already 8.2%
faster. Cold first runs remained 3.16x and 3.05x slower for T-maze and D1.
The retained horizon-8 case has no latency-equivalent old result because the
old solver never converged.

This ticket owns fixed compilation and proof/certification costs for tiny,
one-shot mixed-GFE solves. `cff-wbhu` owns Python sweep dispatch and compiled
Engine/Colony batching and is linked because device-loop machinery may be
shared. `cff-yl0a` owns exact HMM chain-kernel fusion, and `cff-j7mm` owns
resource-estimator/XLA cost reporting; neither substitutes for this work.

## Change-path handoff

Applicable row: **Runtime numerical or status invariant**.

- Authority: `AGENTS.md` numerical contract; `docs/DERIVATIONS.md` D7;
  `src/cffg/engine.py` mixed-GFE dispatch and certification; and
  `src/cffg/outer.py` fixed-support controller and compiled trust kernels.
- Execution surfaces: cold and warm `Engine.run`; T-maze observation GFE and
  D1 transition GFE; candidate certification; lazy trust-kernel entry;
  flooding/mixed parity; and per-engine/global cache ownership.
- Duplicated pairs: host/JIT, cold/warm, and flood/mixed are affected.
  Single/`vmap` remains a parity surface. Exact two-pass schedules,
  factor-family equations, serialization, readouts, and checkpoint migration
  are N/A.
- Focused fixtures: canonical T-maze, D1 dynamics curiosity, ten changing-
  belief planning calls, the horizon-8 retained cycle, and ordinary BFE.
- Documentation: update `ARCHITECTURE.md` when compilation/cache ownership or
  the host/device boundary changes; benchmark evidence remains in this ticket.

## Hostile inputs

- Exact support changes must invalidate or rebuild every support-specific
  certificate and derivative kernel. No probability floor is permitted.
- Vacuous, improper, and non-finite states retain the existing truthful
  infeasible/diverged/inner-failure classifications on every fast path.
- Float32 keeps dtype-derived residual, rank, and trust guards; caches key on
  dtype and cannot reuse x64 artifacts.
- The horizon-8 24-logit retained cycle and D1 32-by-4 hostile case remain
  unchanged scale controls. An optimization for tiny charts must fall back to
  the globalized method rather than imposing a small static limit.
- Dynamic observations and parameters remain inputs, while structure,
  support, dtype, and hardening phase determine safe compilation reuse.

## Dependency map

Legend: prerequisite --> dependent

    [x] cff-weff proof-backed outer solver
              |
              +--> [x] cff-lv2w throughput recovery
                          |
                          +--> [ ] cff-g7rc fixed one-shot overhead (ready)

    [ ] cff-wbhu compiled Engine/Colony loop (ready in parallel; linked)

`cff-g7rc` has no immediate blocker and does not block a downstream roadmap
ticket; it is an independent strict-performance continuation. `cff-wbhu`
remains on the `cff-ahps` critical path through long-rollout planning.

## Design directions

Measure before selecting among these non-binding levers:

1. Separate model construction, ordinary step JIT, chart construction,
   residual evaluation, derivative compilation, support/local checks, FE
   certification, and host/device synchronization in fresh processes.
2. Do not compile the matrix-free trust/JVP/VJP path until the bounded
   candidate fails its independent raw-root, support, local-solver, and FE
   certificate. Report lightweight-certificate and trust compilation counts
   separately.
3. Fuse repeated reconstruction/residual/support/FE reads at an unchanged
   candidate, while keeping `_make_bfe` the scientific ledger and the raw
   undamped outer residual the stationarity certificate.
4. Reuse safe structure/support/dtype artifacts across equivalent Engines if
   process-level caching measures better than only deferring work. Any cache
   must retain dynamic observations and parameters as runtime inputs.
5. Coordinate any device-side certification loop with `cff-wbhu`; do not
   absorb its Colony speedup acceptance criteria into this ticket.

## Acceptance Criteria

1. No correctness trade: `CONVERGED` still requires the unchanged raw coupled
   outer root, exact support, converged inner solves, feasibility, and finite
   stationary FE. Existing tolerances, objective, and hostile regressions are
   unchanged.
2. A fresh-process benchmark attributes the fixed costs above and reports at
   least five cold process samples plus five retained warm samples for T-maze,
   D1, repeated planning, horizon-8, and ordinary BFE on one recorded
   environment.
3. Against `600a775`, the cold median improves by at least 2x on both T-maze
   and D1. Against pre-soundness `931758d`, the geometric-mean warm premium of
   those two tiny solves improves by at least 3x from the recorded 9.84x
   baseline, without describing the old uncertified result as equivalent.
4. A benign candidate that passes every independent certificate does not
   compile or enter the trust-region derivative kernel. A deliberate failed
   candidate enters it lazily and reaches the same truthful result as the
   current globalized path.
5. Ten changing-belief planning calls, ordinary BFE, and the certified
   horizon-8 result regress by no more than 5%, 5%, and 10% respectively.
   Horizon-8 retains the same root and termination reason; legacy Picard still
   reproduces its bounded cycle.
6. Host/JIT, flood/mixed, changed-support, changed-dtype, float32, improper,
   and sequential/`vmap` regressions pass. The slow tier and one final default
   dashboard suite pass under the repository protocol.
7. The closing note records raw measurements, commands, compilation and
   operation counts, environment, `tk dep cycle`, and `git diff --check`.
