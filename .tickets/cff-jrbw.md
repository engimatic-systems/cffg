---
id: cff-jrbw
status: open
deps: [cff-ii9t]
links: [cff-gl89]
created: 2026-09-02T04:17:42Z
type: chore
priority: 1
assignee: coopwilliams
parent: cff-ws3c
tags: [engine, architecture, runtime, refactor]
---
# Extract explicit outer and exact run strategies

## Background in Plain English — WHY

At authoring snapshot e488df3, `src/cffg/engine.py:1191-1656` is the scope
anchor for deterministic outer-kernel planning, trust-region/globalized mixed-
GFE control, and exact two-pass run control; it is not a promised future line
map. The implementation starts from the exact `cff-sf29` engine baseline.
These strategies have different state and acceptance facts from the ordinary
iterative loop.

This ticket gives outer and exact control separate typed owners. It does not
force them through one solver callback. Engine.run delegates final status to
the shared result owner delivered by cff-ii9t.

## Outcomes / Acceptance Criteria — WHAT

1. `cff-ii9t` is closed with ordinary iterative outcomes passing.
2. An `OuterTrustRegionStrategy` accepts only the compiled outer-kernel plan,
   prepared state, propagation/objective callables, and existing trust-region
   controls; it returns a typed `OuterOutcome`.
3. An `ExactRunStrategy` accepts only the exact schedule unit, prepared state,
   support/objective callables, and tolerance; it returns a typed
   `ExactOutcome`.
4. Neither strategy accepts irrelevant state for the other. There is no
   universal solver callback. Shared termination facts may use a small typed
   protocol consumed by `Engine.run`.
5. `Engine.run` selects the applicable explicit strategy and submits its typed
   outcome to the shared result finalizer for validation and public
   `RunStatus`/`RunResult` construction. No second status mapping is introduced.
6. Trust-region merit/acceptance, budget and rejection counts, inner status,
   exact two-pass behavior, support classification, traces, and diagnostic
   fields retain the `cff-sf29` baseline behavior.
7. The affected row is **Runtime numerical or status invariant**. **New factor
   or rule family** applies to the explicit GFE plan input; **New IR field** and
   **Learnable or checkpoint feature** are N/A because plan/checkpoint schemas
   do not change.
8. Flooding/scheduled propagation enters through the same schedule owner.
   Host/JIT termination facts agree; direct/`vmap` applies where each strategy
   is supported; restored states use the same prepared-state boundary.
9. Literal zero support, improper beliefs, non-finite merit/FE, rejected outer
   steps, float32, and uncomfortable-scale fixtures retain their baseline
   outcome or loud status.
10. Existing mixed-GFE, exact-schedule, runtime-status, and family oracle tests
    remain authoritative. Old/new strategy characterization protects this
    refactor; no new scientific or globalization experiment is required.
11. No solver/globalization rule, objective formula, approximation, factor
    behavior, public status, checkpoint field, or performance target changes.
12. Each strategy owner is below 1,000 physical lines. The ticket fits one
    context because it moves the responsibility anchored by one contiguous
    e488df3 region into two explicit strategy values after ordinary control
    already has its boundary.
13. Focused outer/exact/runtime tests and the final short suite pass.
14. `docs/ARCHITECTURE.md` names both strategy owners and the public status
    authority in the extraction commit.

## Methodological Suggestions — HOW

These suggestions are optional. An implementer may choose another sound method.

- Move `_outer_kernel_plan`/`_run_outer_trust_region` and `_run_exact` into
  distinct owners with narrow constructor values.
- Reuse a termination-facts record only for fields genuinely shared by both.
- Keep the public facade's strategy selection explicit and static.

## Future work

`cff-gl89` extracts objective accounting after execution, schedules, lifecycle,
and every run-control strategy have named owners.

## Definitions

- **ExactRunStrategy:** The owner of existing exact two-pass inference control.
- **OuterTrustRegionStrategy:** The owner of existing deterministic outer
  globalization and acceptance control.
- **Strategy outcome:** Internal state, traces, and termination facts returned
  to `Engine.run` without constructing a public status.
