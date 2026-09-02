---
id: cff-ny7f
status: open
deps: [cff-sf29]
links: [cff-nyad, cff-wbhu]
created: 2026-09-02T04:02:01Z
type: chore
priority: 1
assignee: coopwilliams
parent: cff-ws3c
tags: [engine, architecture, refactor]
---
# Extract per-plan execution owners

## Background in Plain English — WHY

At e488df3, `src/cffg/engine.py:2511-3040` gathers inputs, selects compiled
parameters, invokes rules, and writes messages for mixture, recurrence, hybrid
GFE, precision, Gaussian, exponential-family, constraint, and ordinary groups.
Flooding and schedule builders risk repeating this rule-call interpretation.

This ticket extracts only per-plan execution. Schedule classification,
propagation, support checks, and run control stay in later tickets.

## Outcomes / Acceptance Criteria — WHAT

1. `cff-sf29` is closed with the bounded engine characterization baseline.
2. Typed execution owners cover every plan class exercised by `cff-sf29`:
   registered discrete/Gaussian/EF, mixture, recurrence, precision, constraint,
   GFE/P-substitution, and functional evaluator plans.
3. Each executor accepts a typed compiled-plan view, state-array view, and
   compiled parameter view and returns an `ExecutionDelta` containing only the
   messages/internal values to write.
4. Each executor gathers inputs, selects compiled parameters/observations, and
   calls one existing mathematical rule. It does not build schedules, loop to
   convergence, classify public status, or construct a public `Engine`.
5. Registered same-arena plans dispatch from kernel metadata. Each multi-arena,
   GFE, evaluator, and structural plan keeps an explicit typed executor.
6. Flooding and existing scheduled wrappers invoke the same rule-call owner for
   equivalent updates. `cff-nyad` later moves schedule construction without
   creating a second factor interpretation.
7. A direct executor and one-instance `vmap` agree where batching applies.
   Host/JIT status classification and checkpoint decoding are N/A to this
   ticket; they consume the resulting state downstream.
8. The affected rows are **New factor or rule family** and **Runtime numerical
   or status invariant**. **Learnable or checkpoint feature** applies to
   parameter/statistic consumers. **New IR field** is N/A because compiled plan
   fields do not change.
9. Literal zero support, vacuous messages, improper beliefs, non-finite inputs,
   float32, and the uncomfortable-scale fixtures retain their baseline update
   values or internal loud failure.
10. Existing family oracle tests remain mathematical authorities. Old/new
    observable characterization is permitted to protect this refactor seam; no
    new scientific benchmark is required unless a separately ticketed gap
    proves the extraction unsafe.
11. No schedule kind, solver, rule, factor family, objective, public status,
    checkpoint field, diagnostic, or performance target changes.
12. The execution-owner source is below 1,000 physical lines and the ticket
    fits one context because it moves only the e488df3 rule-call region.
13. Focused family/runtime tests and the final short suite pass.
14. `docs/ARCHITECTURE.md` names the execution inputs, output, specialized
    owners, and shared flooding/scheduled rule-call invariant.

## Methodological Suggestions — HOW

These suggestions are optional. An implementer may choose another sound method.

- Extract the `_mix_sweep` through `_make_step` family-call region before
  moving schedule builders.
- Use narrow typed views instead of handing every executor the full arena.
- Keep temporary wrappers in engine/schedule code until both use the owner.

## Future work

`cff-nyad` extracts schedule and propagation construction around these
executors. `cff-wbhu` remains the separate compiled-loop performance owner.

## Definitions

- **ExecutionDelta:** Typed messages or internal values produced by one plan
  update before the caller writes them to state.
- **Rule-call owner:** The only runtime owner that translates one compiled plan
  into a call to its existing mathematical rule.
- **State-array view:** The smallest immutable selection of message, belief,
  observation, and parameter arrays needed by an executor.
