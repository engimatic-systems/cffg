---
id: cff-0qry
status: open
deps: [cff-qg1d]
links: []
created: 2026-08-30T16:00:47Z
type: feature
priority: 1
assignee: coopwilliams
parent: cff-0qfy
tags: [consolidation, learning, rslds]
---
# Port declared rSLDS fitting onto hybrid learning

## Background in Plain English — WHY

PR #65 replaces rSLDS-specific statistic discovery and posterior updates with
the declared learning families supplied by PR #64. Its delta must follow the
new hybrid-learning branch without importing either old branch's ancestry or
ticket history.

The applicable architecture change-path rows are **Learnable or checkpoint
feature** and **Runtime numerical or status invariant**. The template and
`Learnables` own parameter declarations and updates. The rSLDS model owns graph
construction and model-specific diagnostics. The fitter owns outer-coordinate
policy and terminal certification.

## Outcomes / Acceptance Criteria — WHAT

1. The production, test, and canonical-documentation behavior from PR #65 is
   present relative to the new PR #64 port.
2. The rSLDS template declares the learned initial regime, switching dynamics,
   recurrence coefficients, and emission parameters. The continuous initial
   state remains a fixed initializer output.
3. Learning sources and targets are resolved from serialized declarations and
   blob identities. Renaming factors does not change fitted values.
4. Sequence replacement and distinct-trial accumulation use public
   `Learnables` operations and update all selected posterior and point values
   atomically.
5. The returned state is inferred after the final parameter write. Diverged,
   infeasible, non-finite, and budget-exhausted attempts cannot look converged.
6. Sparse or unused regimes retain their documented zero contribution;
   improper beliefs and non-finite values fail before mutation; float32 and the
   retained longer-horizon fixture remain finite and truthfully status-bearing.
7. The change-path surfaces agree: template construction, compiler learning
   plans, engine readouts, named writes, checkpointing, fitter coordination,
   and result reporting use one declared contract.
8. Duplicated surfaces are checked as follows: flooding and scheduled inference
   produce equivalent statistics; host and JIT status checks remain consistent;
   the fitter is single-instance and adds no `vmap` path; current and legacy
   checkpoint behavior remains explicit.
9. `README.md`, `docs/ARCHITECTURE.md`, and `docs/LIMITATIONS.md` reconcile the
   declared rSLDS fit boundary with `main` commit `d805135`.
10. Old `.tickets/cff-11jj.md` state and old generated runtime metadata are
    absent.

## Methodological Suggestions — HOW

These suggestions are optional. The implementer may choose another sound
method.

- Reapply the four production commits from PR #65 and omit its old ticket-close
  commit.
- Preserve model-specific initialization and diagnostics while moving only
  reusable declared-learning coordination behind generic interfaces.
- Run hybrid-learnable, recurrence-link, and rSLDS-fit focused tests.

## Future work

cff-rl7r depends on this ticket and will correct the recurrence covariance
statistic used by this learning path. No new optimizer or stochastic retry
contract is added here.

## Definitions

- **Outer coordinate:** One inference, posterior-replacement, parameter-write,
  and convergence-check cycle.
- **Terminal certification:** The final inference run that proves the returned
  state matches the returned parameters.
- **rSLDS:** Recurrent switching linear dynamical system.
- **Status-bearing:** Returning an explicit converged, budget-exhausted,
  diverged, or infeasible state instead of only a value.
