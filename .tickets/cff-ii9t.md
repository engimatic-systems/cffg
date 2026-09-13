---
id: cff-ii9t
status: open
deps: [cff-5xdy, cff-yglj]
links: [cff-jrbw, cff-sf29, cff-jby3]
created: 2026-09-02T04:17:42Z
type: chore
priority: 1
assignee: coopwilliams
parent: cff-ws3c
tags: [engine, architecture, runtime, refactor]
---
# Extract iterative run control beneath Engine.run

## Background in Plain English — WHY

At authoring snapshot e488df3, `src/cffg/engine.py:916-1190` and lines
1657-1752 are scope anchors for ordinary iteration plus finite-state and zero-
support checks; they are not promised future line maps. The implementation
starts from the exact `cff-sf29` engine baseline. This responsibility selects
propagation, tracks free energy and deltas, applies damping/hardening, and
distinguishes convergence, exhaustion, divergence, and infeasibility.

This ticket moves only ordinary iterative control behind a typed outcome.
Specialized deterministic outer trust-region and exact-run strategies remain in
`cff-jrbw`. Engine and Colony share one termination policy and result finalizer;
each public entry point delegates its result construction to that owner.

## Outcomes / Acceptance Criteria — WHAT

1. `cff-5xdy` is closed with lifecycle services preserving repaired state.
2. One iterative controller accepts prepared `State`, `ScheduleProgram`,
   objective evaluator, and explicit tolerance/budget/damping controls and
   returns an immutable `IterationOutcome`.
3. `IterationOutcome` contains final internal state, iteration count, delta/FE
   trace facts, finite/support signals, and one typed termination reason. It is
   not a public `RunResult` and does not choose a public status enum.
4. One lower-level result owner interprets termination facts, validates the
   outcome, maps it to public `RunStatus`, and constructs `RunResult`.
   Engine.run and Colony.run both delegate to it. Their ordinary iterations
   share incremental termination assessment instead of retaining separate
   finite/support/FE/status policies. Public results and enum identity remain
   compatible; runtime value definitions live below the facade and are
   re-exported through the existing imports.
5. Non-finite delta/FE is divergence; exhausted budget is not convergence;
   literal zero support is infeasible; existing convergence tolerances and
   logical iteration meaning remain unchanged.
6. The affected row is **Runtime numerical or status invariant**. **New factor
   or rule family** applies only through the shared schedule/objective inputs;
   **New IR field** and **Learnable or checkpoint feature** are N/A because no
   schema or restored-state decoding changes.
7. Flooding/scheduled controllers consume the same executor/schedule owners.
   The cff-sf29 multi-instance/multi-structure Colony fixture and host/JIT
   termination facts retain their recorded contract. Restored states enter
   through the same prepared-state boundary. Colony keeps batching and active
   instance freezing; it is not replaced with a sequential Engine loop.
   7.1. A bounded batch case includes one initially invalid or infeasible
        instance beside a healthy instance, and instances that stop at
        different iterations. Results retain independent status and state.
   7.2. Numerical step/termination facts remain fixed-shape arrays usable by
        JAX transformations. Converting traces to public Python tuples occurs
        at the result boundary. Compiling the convergence loop is cff-wbhu.
8. Literal zero support, vacuous/improper state, non-finite update or FE,
   float32, and uncomfortable-scale cases retain their exact status reason and
   trace behavior.
9. Existing runtime-status and family oracle tests remain authoritative.
   Old/new outcome characterization protects the refactor; any newly discovered
   scientific evidence gap becomes a separate ticket unless it proves the move
   unsafe.
10. No solver/globalization rule, objective formula, factor behavior, public
    status, diagnostic field, checkpoint schema, or performance target changes.
11. The iterative-controller module is below 1,000 physical lines and the
    ticket owns ordinary-loop and support/finite classification plus the
    corresponding duplicated Colony policy. It adds no new run strategy.
12. Focused runtime-status/schedule tests and the final short suite pass.
13. `docs/ARCHITECTURE.md` names `IterationOutcome`, the controller, and the
    final public status boundary in the extraction commit.

## Methodological Suggestions — HOW

These suggestions are optional. An implementer may choose another sound method.

- Use typed numerical reason codes beneath one shared public-result finalizer;
  avoid parallel enum-to-status mappings in Engine and Colony.
- Pass the already-built schedule and objective callables; do not make the
  controller inspect factor types.
- Characterize traces and status facts field by field rather than adding a
  long-running benchmark.

## Future work

`cff-jrbw` extracts explicit outer trust-region and exact strategies under the
same shared result boundary. cff-wbhu compiles the ordinary convergence loop;
cff-rqqc owns stochastic Colony state/PRNG isolation.

## Definitions

- **IterationOutcome:** Internal state and termination facts returned to the
  public status authority.
- **Ordinary iterative control:** Repeated scheduled/flooding updates governed
  by existing delta and free-energy convergence rules.
- **Termination reason:** Internal converged, exhausted, diverged, infeasible,
  or unsupported fact before public status construction.
- **Result finalizer:** The shared operation that validates termination facts
  and constructs the public result for Engine or Colony.

## Notes

**2026-09-09T03:29:39Z**

PR #108 research — question: does one-instance vmap test Colony result handling? Answer: no; Colony bypasses its batched path for one instance and constructs RunResult itself for actual batches. Engine and Colony need a shared termination/result owner, with real mixed-input batches and independent stopping checks. Sources at 7885bf5: colony.py:111-118, 149-280. Confidence: high. This moves existing status semantics; cff-wbhu still owns compiling the host convergence loop.

**2026-09-10T00:09:36Z**

The cff-sf29 inventory found and directly reproduced the initial-state result discrepancy now isolated in cff-yglj. That ticket is an additional prerequisite here: it establishes one corrected initial-failure contract before this behavior-preserving extraction. Do not preserve Colony's old inner_status=converged/inner_residual=0 for an initially non-finite state merely because the pre-repair characterization baseline recorded it. The overall status was already diverged; the repair concerns consistent inner facts and retaining the rejected state. Ordinary valid-state baseline fixtures remain unchanged.

**2026-09-10T00:49:06Z**

cff-yglj is now closed. Source 1308688 repairs initial Colony result facts and freezes rejected inputs across priming; its final gate passed 1,843 tests with 41 slow deselections and four existing warnings, recorded in e30b330. tests/test_colony_failures.py contains 25 reference cases for the controller extraction. Preserve this corrected behavior, not the pre-repair discrepancy recorded at 76251b4. The original valid-state characterization remains unchanged.
