---
id: cff-gl89
status: open
deps: [cff-jrbw]
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

At authoring snapshot e488df3, `src/cffg/engine.py:3670-3973` is the scope
anchor for objective assembly; it is not a promised future line map. At the
`cff-sf29` contract snapshot, that responsibility still sat beside message
updates, schedules, readouts, and public orchestration. This makes it harder to
show that one factor's message path and objective term use the same parameters
and local belief.

After `cff-jrbw` completes execution, schedule, lifecycle, and run-control
ownership, this ticket gives objective accounting its own focused owner. The
mathematical formulas and existing authorities do not change.
Implementation starts from the final published `cff-jrbw` head, including
the intervening parameter-ownership and failure-diagnostic repairs. The
earlier snapshot is an inventory, not a baseline to restore.

## Outcomes / Acceptance Criteria — WHAT

1. `cff-jrbw` is closed with execution, schedule, lifecycle, and explicit
   run-control owners passing bounded characterization.
2. One objective owner assembles the existing:
   2.1. ordinary factor-local BFE terms;
   2.2. GFE and P-substitution terms;
   2.3. shared parameter terms, including functional `mu3` state;
   2.4. hard-constraint energies and the existing treatment of discrete delta-constrained beliefs; and
   2.5. edge-entropy overcounting corrections.
3. Each registered same-arena kernel contributes through its declared energy
   callback. Each specialized plan contributes through one explicit local-term
   owner.
4. The objective owner consumes typed compiled plans and the corresponding
   beliefs or already decoded assignment, plus parameter, observation/mask,
   and evaluator snapshots. Decoding remains outside this owner. It does not
   perform message updates, select actions, or construct public agents.
5. One objective owner supplies distinct operations for belief free energy
   and decoded assignment energy. Engine.bfe and FE convergence checks share
   BFE/GFE local terms; Engine.objective and max-mode scoring share declared
   assignment-energy callbacks. They do not substitute one quantity for the
   other. BFE remains rejected in max-product mode.
   5.1. The final ordinary/outer/exact controllers consume this objective
        evaluator through explicit injection by the facade or a lower-level
        protocol. The objective owner never imports run-control owners, and
        run-control owners never import the public facade.
6. The true BFE and GFE invariant remains intact.
   6.1. Node-local energy and entropy terms retain their formulas.
   6.2. Each interior edge receives its existing entropy correction exactly
        once.
   6.3. P-substituted nodes retain their declared GFE replacement.
   6.4. Named approximations remain named approximations and do not become
        labeled exact through the extraction.
   6.5. The cff-kfo7 ownership mapping connects factors, absorbed goals/priors,
        edge exclusions, and shared parameter terms to existing authority
        tests. A two-slice mu3 case retains one shared parameter KL term.
7. Existing analytic, enumeration, quadrature, literature, and family-specific
   oracle tests remain authoritative where present. Applicable contract
   cases compare old/new objective totals and local terms against the final
   `cff-jrbw` source, unchanged within their
   dtype-derived tolerance. This move does not require a new scientific oracle.
8. Flooding and scheduled states produce the same objective terms for
   equivalent beliefs and matching parameter, observation/mask, and evaluator
   snapshots. Direct and one-instance `vmap` objective
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
14. The ticket fits one implementation context because it moves the ledger
    responsibility anchored by the bounded e488df3 region after run-control
    owners already accept an injected compatibility evaluator.

## Methodological Suggestions — HOW

These suggestions are optional. An implementer may choose another sound method.

- Move one local-term family at a time while keeping `Engine.bfe` as a
  compatibility delegate.
- Retain existing independent analytic, enumeration, quadrature, and literature
  authorities. Use old/new term characterization only to protect the move, not
  to claim new mathematical validation.
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
- **Discrete delta-constrained belief:** A belief restricted to one category,
  with probability one there and zero elsewhere.

## Notes

**2026-09-10T07:38:50Z**

Pre-implementation inspection at cff-jrbw source baaad19: Engine.bfe (1163+) owns max-mode rejection, functional readiness and host exact-support rejection before _bfe; preserve that order. Engine.objective (1187+) decodes once, then scores ordinary/equality/constraint factors using current State.params; decoding stays with the facade/readout work. _constraint_ok is a compatibility static helper. _make_bfe (1417+) is the remaining numerical assembly. Runtime consumers now receive objectives through IterationServices, ExactRunStrategy, OuterServices and the ScheduleProgram factory; the ledger must not import these controllers. Preserve engine._make_bfe as a compatibility alias because exact-schedule fault-injection tests build units from it. Next implementation should inventory local terms once, use current shared-goal and mu3 ownership plans, and characterize against final jrbw rather than pre-repair sf29.

**2026-09-10T07:44:13Z**

Local-term inventory from baaad19 (read once, no numerical work while jrbw gate runs): _gauss_bfe combines declared Gaussian/equality energies with Gaussian interior entropy; _mix_bfe, _recurrence_bfe, _hybrid_gfe_bfe and _prec_bfe are explicit local adapters; _ef_bfe combines EF node terms with EF interior entropy. _make_bfe orders those, Dirichlet VMP, ordinary declared energies, current-parameter constraints, GFE structural terms, one shared_goal_energy call, one KL per mu3 plan, transition-GFE terms and discrete interior entropy. Preserve summation order for characterization. No standalone delta-energy function exists in this baseline: hardening changes the beliefs consumed by those terms; the acceptance wording now explicitly preserves that treatment rather than requesting an invented new term. EF non-equality energy callbacks still come from kernels.KERNELS at trace time; do not silently change callback lookup timing during a move. No external consumers of the six private local helpers or _constraint_ok were found in src/tests; _make_bfe remains used by exact-schedule instrumentation. Local source inspection only, not independent formalism review.

**2026-09-10T08:04:58Z**

Prerequisite cff-jrbw is closed and published as PR #131. Start from its final published agent/cff-jrbw-run-strategies head after an ff-only pull. Source baaad19 passed the final 2,081-test gate (41 slow deselections, four existing warnings), recorded at aeaf18c; wall 1,165.949 seconds. All three controllers now consume injected evaluators and run_results owns public statuses. Local-term inventory and compatibility-alias details are in the preceding handoff; no design decision is pending.
