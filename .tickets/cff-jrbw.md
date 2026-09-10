---
id: cff-jrbw
status: in_progress
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
map. The implementation starts from the final published `cff-ii9t` head,
including its dependency repairs; `cff-sf29` remains the contract inventory,
not an instruction to restore pre-repair discrepancies.
These strategies have different state and acceptance facts from the ordinary
iterative loop.

This ticket gives outer and exact control separate typed owners. It does not
force them through one solver callback. Engine.run delegates final status to
the shared result owner delivered by cff-ii9t.

## Outcomes / Acceptance Criteria — WHAT

1. `cff-ii9t` is closed with ordinary iterative outcomes passing.
2. An `OuterTrustRegionStrategy` accepts an explicit outer-kernel plan provider,
   the structural chart/hardening inputs it actually needs, prepared state,
   propagation/objective callables, and existing trust-region controls; it
   returns a typed `OuterOutcome`.
   2.1. The provider selects or builds kernels for the current chart and hard
        message positions. One permanently fixed kernel cannot represent
        support-dependent charts and later hardening phases. Preserve the
        existing cache key, creation timing and public compilation count.
   2.2. Chart bases, observations, masks, parameters and hard message values
        stay current runtime inputs, not values captured from the first
        instance using a cached kernel. Keep chart construction's structural
        dependencies explicit; wrapping an entire Engine in a plan object is
        not a narrow interface.
3. An `ExactRunStrategy` accepts only the exact schedule unit, its objective
   mode, prepared state, support/objective callables, and tolerance; it returns
   a typed `ExactOutcome`. Exact certification does not accept trust-region
   controls, functional-particle state services or an outer-kernel provider.
4. Neither strategy accepts irrelevant state for the other. There is no
   universal solver callback. Shared termination facts may use a small typed
   protocol consumed by `Engine.run`.
5. `Engine.run` selects the applicable explicit strategy and submits its typed
   outcome to the shared result finalizer for validation and public
   `RunStatus`/`RunResult` construction. No second status mapping is introduced.
6. Trust-region merit/acceptance, budget and rejection counts, inner status,
   exact two-pass behavior, support classification, traces, and diagnostic
   fields retain the repaired prerequisite baseline behavior.
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
- **Kernel-plan provider:** The owner that reuses or creates the derivative
  kernels for a particular outer-coordinate chart and hard message positions,
  then supplies that call's runtime context separately.

## Notes

**2026-09-10T07:04:26Z**

Pre-implementation handoff at source 20c0367 while cff-ii9t's final gate runs. Source anchors: Engine._outer_kernel_plan, _run_outer_trust_region and _run_exact; outer.build_outer_chart, outer_chart_signature and solve_outer. This is local source inspection, not independent agent review.

- The kernel cache key is chart signature plus hard index dtypes/shapes/bytes. The context separately carries chart.base, observations/masks, State.params and hard values. Preserve that separation and lazy creation; a constructor-time single-kernel snapshot would be wrong.
- outer.py currently uses Arena edges, goal_parameter_plans and three index helpers for chart construction, plus edges in solve_outer. Resolve that explicit structural interface when choosing a chart-plan/provider shape; do not hide an Engine behind a generic callback. No new solver or mathematical chart representation is authorized by this refinement.
- Exact certification has distinct result facts: initial non-finite input is zero iterations with inner divergence; pre-support rejection retains the input; post-support rejection is one logical iteration but an empty delta trace and FE=+infinity. Therefore do not reuse ordinary IterationOutcome's iterations=len(deltas) property as a universal strategy count.
- Exact tol=None can converge from its structural certificate. Outer ROOT_CONVERGED with tol=None still maps to budget exhaustion. Preserve the existing mode-specific meaning; one shared finalizer does not mean one convergence rule.
- Outer initial non-finite/support failures have NONFINITE_EVALUATION/INFEASIBLE public detail tags, not ordinary NOT_RUN. Zero-budget outer runs can still do the existing functional coordinate update; no eager budget shortcut may skip it. Functional failures preserve their supplied residual and INNER_FAILURE tag.
- OuterOutcome must retain the last solver diagnostics and accumulated accepted/rejected, map/trial, linear-iteration and kernel-creation counts. ExactOutcome needs no such fields. A typed outcome union at the shared result owner is appropriate; a universal solver callback is not.
- Keep the current finite/support/checkpoint boundary and the static strategy selection order. Ordinary iteration is now delegated and does not move again. The primitive numerical solver remains outer.py; objective formulas remain for cff-gl89.

**2026-09-10T07:20:21Z**

Prerequisite cff-ii9t is closed and published as PR #130. Use its final published agent/cff-ii9t-iteration-control head as the work-branch base. Code/test source 20c0367 has a final 2,059-test gate (41 slow deselections, four existing warnings), recorded in bbbbccb; measured wall 1,155.815 seconds. Ordinary outcome/finalization lives in iteration_types.py/run_results.py; shared host predicates are in runtime_checks.py. Extend the result authority for exact/outer facts without changing the already-extracted ordinary policy.

**2026-09-10T07:25:58Z**

Starting from verified PR #130 head b379a28 after an ff-only pull. Applicable change paths: Runtime numerical or status invariant, plus explicit GFE chart inputs under New factor or rule family. Paired surfaces: host checks/fused JIT exact certification; flooding/colored/mixed schedule propagation; standalone/existing Colony exact and GFE fallback routes; restored State/current parameter and observation context. New IR/checkpoint schema and unsupported outer-vmap execution are N/A. Preserve schedule instrumentation aliases, dynamic chart/hard-value cache inputs, exact logical counts independent of trace lengths, and strategy-specific tolerance semantics. No mathematical or solver-policy change is intended.

**2026-09-10T07:37:18Z**

Implementation now gives exact_strategy.ExactRunStrategy and outer_strategy.OuterTrustRegionStrategy separate typed outcomes. run_results owns all public result/status construction; Engine retains selection and compatibility delegates. OuterStructure contains only edges, shared-goal plans and three segment-offset arrays. OuterKernelProvider owns the existing chart/hard-index keyed cache; current base, observations, masks, parameters and hard values are supplied as runtime context. outer.py changed only one import and two accepted-type annotations: chart/solver mathematics are untouched. Focused validation: 82 passed, five slow deselections in 37.82 seconds; 22 new boundary tests cover exact pre/post-support counts, tolerance semantics, functional zero-budget updates, accumulated outer work, structure parity, dynamic cached context and owner boundaries. New modules/tests are Ruff-clean; no independent agent review is claimed. Full old/new result characterization and the final short-suite gate are next.

**2026-09-10T07:43:00Z**

Complete-result characterization passed against PR #130 source b379a28: 110 complete results and 3,416 raw-byte-identical numeric comparisons, with enum identity preserved. Coverage: discrete/Gaussian/EF exact execution; ordinary/hybrid/functional GFE; mixed-shape shared learned goals; T-maze staged hardening; both precisions; zero budgets, tol=None, finite tolerances, initial non-finite/support failures, repeated cached runs and existing Colony exact/GFE routes. Script: /tmp/cffg-strategy-parity.py. This is compatibility evidence, not a new scientific oracle. The existing mixed-precision functional scatter FutureWarning appeared. Kernel-plan AST parity separately passed. Final code/test source is baaad19; intervening 6812590 only refines the next ticket. Final short suite starts using cached runtime 1,155.815 seconds and allowance 1,470 seconds; first output check after 20 minutes, no intermediate suite polling.
