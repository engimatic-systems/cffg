---
id: cff-ii9t
status: open
deps: [cff-5xdy]
links: [cff-jrbw]
created: 2026-09-02T04:17:42Z
type: chore
priority: 1
assignee: coopwilliams
parent: cff-ws3c
tags: [engine, architecture, runtime, refactor]
---
# Extract iterative run control beneath Engine.run

## Background in Plain English — WHY

At e488df3, `Engine.step` and the ordinary iterative portion of `Engine.run`
occupy `src/cffg/engine.py:916-1190`; finite-state and zero-support checks live
at lines 1657-1752. They select propagation, track free energy and deltas, apply
damping/hardening, and distinguish convergence, budget exhaustion, divergence,
and infeasibility.

This ticket moves only ordinary iterative control behind a typed outcome.
Specialized deterministic outer trust-region and exact-run strategies remain in
`cff-jrbw`. Public `Engine.run` retains final `RunStatus`/`RunResult` authority.

## Outcomes / Acceptance Criteria — WHAT

1. `cff-5xdy` is closed with lifecycle services preserving repaired state.
2. One iterative controller accepts prepared `State`, `ScheduleProgram`,
   objective evaluator, and explicit tolerance/budget/damping controls and
   returns an immutable `IterationOutcome`.
3. `IterationOutcome` contains final internal state, iteration count, delta/FE
   trace facts, finite/support signals, and one typed termination reason. It is
   not a public `RunResult` and does not choose a public status enum.
4. `Engine.run` selects the ordinary controller, validates its outcome, maps
   termination reason to the final public `RunStatus`, attaches diagnostics,
   and remains the only public `RunResult` constructor.
5. Non-finite delta/FE is divergence; exhausted budget is not convergence;
   literal zero support is infeasible; existing convergence tolerances and
   logical iteration meaning remain unchanged.
6. The affected row is **Runtime numerical or status invariant**. **New factor
   or rule family** applies only through the shared schedule/objective inputs;
   **New IR field** and **Learnable or checkpoint feature** are N/A because no
   schema or restored-state decoding changes.
7. Flooding/scheduled controllers consume the same executor/schedule owners.
   Direct/one-instance `vmap` and host/JIT termination facts agree where
   supported. Restored states enter through the same prepared-state boundary.
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
    ticket fits one context because it owns the bounded ordinary-loop and
    support/finite classification regions only.
12. Focused runtime-status/schedule tests and the final short suite pass.
13. `docs/ARCHITECTURE.md` names `IterationOutcome`, the controller, and the
    final public status boundary in the extraction commit.

## Methodological Suggestions — HOW

These suggestions are optional. An implementer may choose another sound method.

- Use an internal termination enum distinct from the public enum so
  `Engine.run` visibly remains authority.
- Pass the already-built schedule and objective callables; do not make the
  controller inspect factor types.
- Characterize traces and status facts field by field rather than adding a
  long-running benchmark.

## Future work

`cff-jrbw` extracts explicit outer trust-region and exact strategies under the
same public status boundary.

## Definitions

- **IterationOutcome:** Internal state and termination facts returned to the
  public status authority.
- **Ordinary iterative control:** Repeated scheduled/flooding updates governed
  by existing delta and free-energy convergence rules.
- **Termination reason:** Internal converged, exhausted, diverged, infeasible,
  or unsupported fact before public status construction.
