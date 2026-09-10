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

At authoring snapshot e488df3, `src/cffg/engine.py:2511-3040` is the scope
anchor for input gathering, compiled-parameter selection, rule invocation, and
message writes across plan families; it is not a promised future line map. The
implementation starts from the exact `cff-sf29` engine baseline. Flooding and
schedule builders risk repeating this rule-call interpretation.

This ticket extracts only per-plan execution. Schedule classification,
propagation, support checks, and run control stay in later tickets.

## Outcomes / Acceptance Criteria — WHAT

1. `cff-sf29` is closed with the bounded engine characterization baseline.
2. Typed execution owners cover every plan class exercised by `cff-sf29`:
   registered discrete/Gaussian/EF, in-graph Dirichlet parameters, mixture,
   recurrence, precision, constraint, GFE/P-substitution, and functional
   evaluator plans.
3. Each local-update owner accepts static plan metadata, typed local message
   or belief blocks, and selected parameter/observation blocks. It returns an
   `ExecutionDelta` with outgoing blocks and existing inner-solver status,
   residual, and support facts. Numeric leaves are fixed-shape JAX pytrees;
   they are not Python status objects or a public `RunResult`.
4. One owner translates each plan's declared input semantics into calls to
   existing rules. Transport adapters supply local inputs from either flat
   arenas or prefetched/register-carried scan blocks and write the outputs.
   They do not reinterpret BP cavities, VMP beliefs, parameters, or constraints.
   The local-update owner does not build schedules, loop to convergence,
   classify public status, or construct a public `Engine`.
5. Registered same-arena plans dispatch from kernel metadata. Each multi-arena,
   GFE, evaluator, and structural plan keeps an explicit typed executor.
6. Flooding and existing scheduled wrappers invoke the same rule-call owner for
   equivalent updates. `cff-nyad` later moves schedule construction without
   creating a second factor interpretation.
   6.1. Registerized chains keep prefetched inputs, local-block carry, and one
        final scatter; no full-arena carry or per-step full-arena gather is
        introduced to share the rule call.
   6.2. Kernel callables, shapes, and mode selection remain static at trace
        time; parameters and observations remain numerical runtime arguments.
   6.3. Inner exhaustion/divergence and residual facts survive both transport
        paths and reach control without being replaced by successful writes.
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
    fits one context because it moves only the rule-call responsibility anchored
    by that e488df3 region.
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
  update, together with local solver facts before control classifies the run.
- **Rule-call owner:** The only runtime owner that translates one compiled plan
  into a call to its existing mathematical rule.
- **State-array view:** The smallest immutable selection of message, belief,
  observation, and parameter arrays needed by an executor.
- **Transport adapter:** The code that reads/writes arrays or scan registers
  according to a compiled plan, without choosing mathematical input meaning.
- **Pytree:** A JAX-recognized nested structure whose numerical leaves are arrays.

## Notes

**2026-09-09T03:29:39Z**

PR #108 research — question: can one state-gathering executor serve every schedule? Answer: registerized scans consume prefetched/carried blocks and scatter once; a local-block rule API lets both transport paths share mathematics. ExecutionDelta must also carry existing inner residual/status facts or a failed local solve can disappear at the boundary. Sources at 7885bf5: engine.py:2545-2658, 2924, 3036, 3191-3196, 3288-3406; Part II section 4.5, p.15 distinguishes stationary equations from iteration convergence. Confidence: high; no benchmarks run.

**2026-09-10T00:11:44Z**

Pre-extraction handoff from cff-sf29 at unchanged production baseline 76251b44fc467a976aa056556f8fb371fe5de7f4: use the compact family/owner inventory in that ticket instead of re-reading the whole engine. kernel_runtime.py already centralizes registered discrete/Gaussian local calls; retain that authority rather than introducing a second dispatch path. The extraction must cover registered EF and explicit mixture/recurrence/structured recurrence/precision/GFE/transition-GFE/hybrid-GFE/functional evaluators as applicable.

Two non-local facts constrain the local API. First, shared-goal updates collect fresh original predictions from all GFE groups before write_goal_counts at engine.py:2832 and before final damping; a family executor returns those contributions but must not finalize a shared parameter independently. Second, mu3's prepared particle state and update lifecycle (engine.py:515-695) are not a second structural factor execution. Preserve that boundary and the once-only parameter ledger at :3826-3832 while leaving lifecycle and objective extraction to their tickets.

For transport parity, inspect only the relevant callers: _make_step:2708-2947, batch_stage:2948, registerized_scan:3095 and flat_scan:3326. Registerized chains keep prefetched local blocks and one final scatter; all mathematical parameter/mode selection should have one owner regardless of transport. Existing discrete/Gaussian chain tests already check carry size, persistent bytes and trace reuse. New test_engine_contracts pins real two-instance iterative batching plus another structure and full result/cache identities.

cff-yglj is a separate small result-reporting correction that can proceed alongside this ticket after characterization closes; it blocks cff-ii9t, not per-plan extraction. Do not change run policy or fix unrelated specialized input validation (cff-34gi) in this code move.

**2026-09-10T00:13:24Z**

Plan-class completeness check: DirichletGroupPlan is an explicit local executor too, not an ordinary GroupPlan or an EF arena slot. engine.py:2744-2764 calls dirichlet_vmp_update from both-slot categorical inputs and alpha0, then publishes categorical outputs plus distinct prior-concentration/count parameter blocks. Keep those meanings and shapes unchanged. Its independent algebra and in-graph tests are now explicitly listed in cff-sf29's inventory.

RecurrenceComponentPlan, GoalParameterPlan and LearnPlan describe component, parameter or learning ownership; they are not extra factors to execute. For mu3, the local numerical calls to gfe_stationary_stats and mu3_importance_update at engine.py:636-663 belong to an explicit functional adapter if moved here. Readiness/base-posterior checks, particle preparation, host ESS policy, parameter installation and result/key lifecycle remain in Engine until cff-5xdy. Do not invent a new schedule stage for that evaluator. Existing registerized EF code at engine.py:3254-3265 bypasses _ef_group_out and directly calls the registered message function; this is a concrete duplicate local-call interpretation to unify without introducing arena gathers into the scan.
