---
id: cff-wjqg
status: open
deps: [cff-j4mu, cff-d56d, cff-44we]
links: []
created: 2026-08-13T04:33:03Z
type: feature
priority: 1
assignee: coopwilliams
parent: cff-wbhu
tags: [performance, runtime, status]
---
# Compile ordinary Engine and Colony iteration with per-instance statuses

## Background in Plain English

`Engine.run` currently launches a compiled message update from a Python loop.
`Colony.run` vectorizes one update across compatible instances, but Python still
launches each sweep and synchronizes status data with the host. A recorded
32-instance CPU benchmark therefore measured only a 1.5-times throughput gain.

This ticket compiles the ordinary deterministic iteration controller and its
per-instance stopping state. It covers flooding, colored, mixed-schedule
non-GFE, and max-product iterative runs. It does not move the deterministic
mixed-GFE trust-region controller; `cff-0fw2` owns that separate numerical
surface.

## Outcomes / Acceptance Criteria

1. Existing `Engine.run`, `Colony.run`, `RunResult`, and `RunStatus` public call
   forms and meanings remain compatible.
2. One compiled ordinary controller carries state, logical iteration count,
   termination code, residual, previous finite objective value, message-delta
   trace, and defined objective trace.
3. Flooding, colored, mixed-schedule non-GFE, and max-product iterative runs
   have no Python loop over sweeps.
4. The controller calls the selected existing iteration implementation and
   does not reimplement message rules, schedule selection, support semantics,
   or objective terms.
5. Device submissions are independent of the requested sweep count within one
   soft or hardening phase. Trace arrays are converted and truncated on the
   host only after the phase returns.
6. Compiled stopping preserves the runtime contract.
   6.1. Sum-product convergence requires both finite free-energy stationarity
   and message coherence under the existing effective tolerance.
   6.2. Max-product convergence requires message coherence and a finite decoded
   objective.
   6.3. Budget exhaustion, divergence, and infeasibility remain distinct.
   6.4. A non-finite input, update, residual, inner status, or objective cannot
   be returned as converged.
   6.5. A proven empty support returns infeasible and positive-infinity free
   energy where the ledger is defined.
7. Forest priming, exact two-pass execution, delta hardening, damping
   restrictions, schedule fingerprints, and logical iteration counts retain
   their documented behavior.
8. One same-structure Colony group runs its ordinary iteration as a masked
   batch with per-instance active flag, status, count, residual, and traces on
   the device.
   8.1. An inactive instance keeps its terminal state and trace while peers
   continue.
   8.2. Diverged, infeasible, converged, and exhausted peers do not overwrite
   one another's results.
   8.3. A one-instance Colony result agrees with direct `Engine.run`.
   8.4. Exact two-pass groups use the exact schedule rather than repeated
   flooding.
9. Characterization covers host and compiled control, flooding and scheduled
   updates, direct and one-instance Colony execution, and the discrete,
   Gaussian, exponential-family, mixture, recurrence, precision, and
   hard-constraint paths that use the ordinary controller.
10. Hostile coverage includes exact zero support, vacuous messages, improper
    Gaussian or EF beliefs, float32 effective tolerances, 320 same-structure
    Colony instances, and a 600-sweep requested budget.
11. A committed benchmark reproduces the historical 32-instance workload and
    records compilation, first execution, warmed execution, hardware, dtype,
    JAX versions, repetition count, device submissions, logical iterations,
    sequential time, batched time, and their confidence interval.
12. Warm Colony throughput on the recorded workload is at least eight times
    sequential throughput, and compiled single-engine warm time does not
    regress by more than ten percent with a 95-percent paired bootstrap
    confidence interval excluding a larger regression.
13. `docs/ARCHITECTURE.md` and `docs/LIMITATIONS.md` state the measured ordinary
    loop envelope and retain the mixed-GFE host-controller limitation until
    `cff-0fw2` closes.

## Methodological Suggestions

These suggestions are optional. An implementer may choose another sound method
that satisfies the numbered outcomes with equivalent evidence.

- Compile control flow around the existing iteration primitive with
  `lax.scan` or `lax.while_loop`; do not compile a second inference engine.
- Treat a Colony group as a masked batch of independent runs. Keep stopped
  members immutable in the carry.
- Commit the benchmark and characterization tests before replacing the loop.
- Use the **Runtime numerical or status invariant** change path and walk
  host/JIT, flooding/scheduled, direct/Colony, and checkpoint consumers.

## Future work

`cff-0fw2` consumes this device-side status carry for deterministic mixed-GFE
controllers. `cff-xdip` consumes it for depth-aware agent planning.
`cff-rqqc` separately owns stochastic `mu3` state in Colony. Slice-fused chain
kernels remain `cff-yl0a` work.

## Definitions

- **Active flag:** Per-instance state indicating that a batched run still needs
  another logical update.
- **Controller:** The loop that sequences message updates and classifies
  termination without owning factor mathematics.
- **Device submission:** One host request to execute compiled work on a JAX
  backend.
- **Ordinary iteration:** Deterministic iterative execution outside the
  mixed-GFE trust-region controller.
- **Warm throughput:** Completed inference work per unit time after compilation
  and one-time setup.
