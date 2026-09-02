---
id: cff-gl89
status: open
deps: [cff-ny7f]
links: [cff-3ola, cff-cj5h]
created: 2026-09-02T04:02:01Z
type: chore
priority: 1
assignee: coopwilliams
parent: cff-ws3c
tags: [engine, architecture, ledger, refactor]
---
# Extract the free-energy ledger owner

## Background in Plain English — WHY

At the `cff-sf29` baseline, the engine computes BFE and GFE after execution,
but objective assembly lives beside message updates, schedules, readouts, and
public orchestration. This makes it harder to prove that one factor's message
path and objective term use the same parameters and local belief.

After `cff-ny7f` gives rule execution a focused owner, this ticket gives
objective accounting its own focused owner. The mathematical formulas and
independent oracles do not change.

## Outcomes / Acceptance Criteria — WHAT

1. `cff-ny7f` is closed with execution and schedule owners passing their
   characterization evidence.
2. One objective owner assembles the existing:
   2.1. ordinary factor-local BFE terms;
   2.2. GFE and P-substitution terms;
   2.3. shared parameter terms, including functional `mu3` state;
   2.4. hard-constraint and delta terms; and
   2.5. edge-entropy overcounting corrections.
3. Each registered same-arena kernel contributes through its declared energy
   callback. Each specialized plan contributes through one explicit local-term
   owner.
4. The objective owner consumes typed compiled plans and beliefs. It does not
   perform message updates, select actions, decode states, or construct public
   agents.
5. `Engine.bfe`, `Engine.objective`, convergence checks, and any public
   objective diagnostics delegate to the same local-term implementations.
6. The true BFE and GFE invariant remains intact.
   6.1. Node-local energy and entropy terms retain their formulas.
   6.2. Each interior edge receives its existing entropy correction exactly
        once.
   6.3. P-substituted nodes retain their declared GFE replacement.
   6.4. Named approximations remain named approximations and do not become
        labeled exact through the extraction.
7. Every applicable `cff-sf29` objective total, local term, status, and
   independent oracle remains unchanged within its dtype-derived tolerance.
8. Flooding and scheduled states produce the same objective terms where they
   produce equivalent beliefs. Direct and one-instance `vmap` objective
   evaluation agree where both are supported.
9. Literal zero support produces the existing infinite or infeasible ledger
   result. Vacuous and improper beliefs, non-finite local terms, float32, and
   uncomfortable-scale fixtures remain finite or fail loudly according to
   their existing contracts.
10. The extraction adds no objective, approximation, message rule, convergence
    criterion, diagnostic promise, or performance threshold.
11. Each new source file stays below 1,000 physical lines and imports neither
    public agent code nor public engine orchestration.
12. Focused tree, GFE, hybrid, recurrence, functional `mu3`, constraint,
    runtime-status, and objective tests pass. The dashboard-backed short suite
    passes after the final code change.
13. `docs/ARCHITECTURE.md` names the objective owner and dependency direction
    in the same commit as the extraction.

## Methodological Suggestions — HOW

These suggestions are optional. An implementer may choose another sound method.

- Move one local-term family at a time while keeping `Engine.bfe` as a
  compatibility delegate.
- Retain independent analytic, enumeration, quadrature, and literature oracles;
  do not replace them with old-versus-new self-parity.
- Use the **New factor or rule family** and **Runtime numerical or status
  invariant** change-path rows.
- Keep shared parameter terms explicit rather than assigning them to an
  arbitrary factor instance.

## Future work

`cff-3ola` extracts public readouts and leaves the public engine as
orchestration. `cff-cj5h` may later add diagnostics through the focused
objective owner; that feature is not part of this ticket.

## Definitions

- **Edge-entropy correction:** The term that removes entropy counted more than
  once by neighboring factor beliefs.
- **Ledger:** The labeled collection of local terms whose sum is BFE or GFE.
- **Shared parameter term:** An objective contribution owned once by a
  parameter belief used by several factors.
