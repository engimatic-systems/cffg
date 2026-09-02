---
id: cff-ny7f
status: open
deps: [cff-sf29]
links: [cff-gl89, cff-wbhu]
created: 2026-09-02T04:02:01Z
type: chore
priority: 1
assignee: coopwilliams
parent: cff-ws3c
tags: [engine, architecture, refactor]
---
# Extract per-arena and schedule execution owners

## Background in Plain English — WHY

At the `cff-sf29` named baseline, the engine gathers inputs, selects parameters,
calls factor rules, writes messages, and builds schedule functions in the same
module as public orchestration. Flooding, staged batches, registerized scans,
exact passes, and specialized GFE paths repeat parts of that interpretation.

`cff-sf29` first pins values and failures at a named baseline. This ticket
then gives execution and scheduling focused owners without changing any
numerical method or public result.

## Outcomes / Acceptance Criteria — WHAT

1. `cff-sf29` is closed with its engine characterization evidence passing.
2. Typed execution owners cover the discrete, Gaussian, exponential-family,
   mixture, recurrence, precision, constraint, GFE, and functional-evaluator
   plan classes exercised by `cff-sf29`.
3. Each execution owner has one responsibility:
   3.1. gather the typed inputs for its plan;
   3.2. select already-compiled parameters and observations;
   3.3. call the existing mathematical rule; and
   3.4. return the messages or internal result that its caller writes.
4. Registered same-arena factors dispatch from kernel metadata. Multi-arena,
   GFE, evaluator, and structural plans keep explicit typed executors rather
   than one callback with unrelated fields.
5. Flooding, staged batches, and scan stages call the same family execution
   primitives where they implement the same update.
6. One schedule owner constructs flooding, batch-stage, scan-stage,
   exact-pass, support-check, and propagation functions from typed compiled
   plans.
7. Schedule selection, fingerprints, forest priming, exact two-pass behavior,
   damping restrictions, delta hardening, and logical iteration meaning remain
   unchanged.
8. Public `Engine.run` remains the status-bearing orchestration authority.
   Extracted execution and schedule modules do not import or construct the
   public `Engine` class.
9. Every applicable `cff-sf29` message, belief, schedule, status, diagnostic,
   and stochastic-state characterization remains unchanged.
10. Deliberate execution pairs remain equivalent.
    10.1. Flooding and scheduled paths share the extracted rule-call owner.
    10.2. Direct and one-instance `vmap` paths consume the same executor.
    10.3. Host status conversion uses the termination values produced by the
          same compiled execution path.
    10.4. Checkpoint decoding is unchanged; restored agent states execute
          through the same extracted owners.
11. Literal zero support, vacuous messages, improper beliefs, non-finite
    updates, float32, and the uncomfortable-scale fixtures from `cff-sf29`
    retain their documented values or loud statuses.
12. The extraction adds no solver, rule, factor family, schedule kind,
    diagnostic field, checkpoint field, or runtime-speed requirement.
13. Each new source file stays below 1,000 physical lines and has one named
    responsibility.
14. Focused runtime, schedule, rule-family, agent, and checkpoint tests pass.
    The dashboard-backed short suite passes after the final code change.
15. `docs/ARCHITECTURE.md` names the execution and schedule owners in the
    same commits that move those responsibilities.

## Methodological Suggestions — HOW

These suggestions are optional. An implementer may choose another sound method.

- Move input gathering and rule calls before moving schedule builders that
  close over them.
- Keep compatibility wrappers until flooding and scheduled consumers both use
  the extracted primitive.
- Use the **Runtime numerical or status invariant** and **New factor or rule
  family** change-path rows. Use **Learnable or checkpoint feature** for the
  restored-agent consumer check.
- Make each owner accept the smallest typed arena view that it needs.

## Future work

`cff-gl89` extracts objective accounting after execution ownership is stable.
`cff-wbhu` may later compile more loop control around these owners, but its
performance outcome is not part of this ticket.

## Definitions

- **Executor:** A function that applies an existing rule to one typed compiled
  plan.
- **Propagation:** Ordered application of schedule stages to message state.
- **Rule-call owner:** The only runtime module that translates a compiled plan
  into one mathematical rule invocation.
- **Schedule stage:** One compiled batch or scan unit selected by the schedule.
