---
id: cff-sf29
status: open
deps: [cff-edoz]
links: [cff-ny7f]
created: 2026-09-02T04:02:01Z
type: task
priority: 1
assignee: coopwilliams
parent: cff-ws3c
tags: [engine, architecture, characterization]
---
# Pin engine contracts before extraction

## Background in Plain English — WHY

The compiler track gives runtime consumers stable typed plans. The next work
will move execution, schedules, objective accounting, and readouts out of a
3,973-line engine module. Before that move, reviewers need direct evidence for
the values and failures each path currently produces.

This ticket adds characterization evidence and an ownership map for a named
baseline commit. It does not move production engine code or redefine a public
result.

## Outcomes / Acceptance Criteria — WHAT

1. `cff-edoz` is closed with its compiler characterization and typed-stage
   outcomes satisfied.
2. The characterization record names the exact compiler-track commit used as
   its baseline.
3. Compact fixtures cover ordinary discrete, Gaussian, exponential-family,
   mixture, recurrence, structured recurrence, precision, hard-constraint,
   fixed-A GFE, transition GFE, hybrid GFE, and functional `mu3` execution.
4. The fixtures pin applicable observable values:
   4.1. outgoing messages and public beliefs;
   4.2. run status, logical iteration count, residual, and termination reason;
   4.3. schedule kind and fingerprint;
   4.4. BFE or GFE total and independently testable local terms;
   4.5. factor and parameter diagnostics;
   4.6. public readouts and their shapes; and
   4.7. prepared stochastic state, key cursor, and cache identity where
        applicable.
5. Every converged fixture is paired with the independent analytic,
   enumeration, quadrature, literature, or source-specific oracle named by its
   baseline test. Self-parity alone is not the evidence for mathematical
   correctness.
6. Failure fixtures cover budget exhaustion, divergence, infeasibility,
   unsupported execution, non-finite state, and an inner-solver failure.
7. A compact ownership table identifies every block at the named baseline that
   gathers inputs, selects parameters, calls rules, writes messages, builds
   schedules, computes objective terms, or constructs readouts. Repeated
   interpretations are named for later removal.
8. Deliberate execution pairs are explicit.
   8.1. Flooding and scheduled execution agree where both are supported.
   8.2. Direct and one-instance `vmap` execution agree where both are
        supported.
   8.3. Host result classification agrees with status values produced by JIT.
   8.4. Current and supported legacy agent checkpoints reconstruct results that
        satisfy the same public engine invariants; checkpoint decoding itself
        remains outside the engine extraction.
9. Hostile fixtures include literal zero support, a vacuous message, an
   improper continuous belief, a non-finite update, float32, and one domain or
   horizon at least ten times its smallest fixture.
10. No production engine behavior, public API, serialized structure, checkpoint
    field, or performance target changes in this ticket.
11. Focused runtime, schedule, rule-family, ledger, readout, agent, and
    checkpoint tests pass. The dashboard-backed short suite passes after the
    final test change.

## Methodological Suggestions — HOW

These suggestions are optional. An implementer may choose another sound method.

- Extend existing oracle tests and expose focused internal term comparisons
  only when needed for characterization.
- Record values field by field rather than storing one opaque engine snapshot.
- Use the **Runtime numerical or status invariant** row as the primary
  change-path checklist. Also use **New factor or rule family** and **Learnable
  or checkpoint feature** for affected consumers.
- Keep the ownership map in the ticket handoff until implementation changes
  architecture; update `docs/ARCHITECTURE.md` with the delivered owners, not
  the pre-refactor duplication.

## Future work

`cff-ny7f` extracts execution and schedule owners after this evidence passes.
No other future work is identified.

## Definitions

- **Independent oracle:** A calculation or source that does not call the
  production path being checked.
- **Local term:** One named factor, parameter, constraint, or edge contribution
  to BFE or GFE.
- **Ownership map:** A list connecting each repeated operation to the code that
  currently performs it.
- **Termination reason:** The detailed cause associated with a public run
  status.
