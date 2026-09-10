---
id: cff-ny7f
status: closed
deps: [cff-sf29]
links: [cff-nyad, cff-wbhu, cff-6n9c]
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
- **Dirichlet parameter edge:** An edge representing uncertainty about a
  probability vector or conditional-probability table; it carries prior
  concentrations and fractional observation counts, not categorical log messages.

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

**2026-09-10T00:51:06Z**

Published intended implementation base: PR #123, branch agent/cff-yglj-colony-initial-failures. Update that base before branching. Its source 1308688 passed 1,843 default tests, 41 slow deselections and four existing warnings; gate recorded in e30b330. This includes the separately approved initial-failure repair. The engine characterization source baseline remains 76251b4, and its valid-state observable contracts remain unchanged. Preserve the cff-yglj correction during execution extraction; do not reintroduce the earlier Colony discrepancy.

**2026-09-10T00:51:59Z**

Starting from updated PR #123 head 8209628. The exact engine-characterization baseline remains 76251b44fc467a976aa056556f8fb371fe5de7f4, with the separate cff-yglj correction at source 1308688 retained (1,843 passing default tests, gate e30b330). Primary change paths: New factor or rule family and Runtime numerical or status invariant. Learnable/checkpoint applies to parameter/statistic consumers; New IR field and checkpoint decoding are N/A.

Move local parameter selection, input interpretation and existing rule calls behind typed per-plan outputs/facts. Keep schedule selection, flat/register transport, lifecycle checks, host run control, objective formulas and public results outside this extraction. Flooding, batch, registerized and flat scans must share the same registered local call. Preserve static kernel/mode/shape decisions, dynamic numeric inputs, prefetched scan carry and final scatter. Explicit owners include in-graph Dirichlet, mixture, structured recurrence, precision, constraints, fixed/transition/hybrid GFE and functional mu3 numerical phases; metadata-only component/goal/learning plans are not extra executable factors. Shared-goal predictions are pooled once after all contributors and before damping; parameter energy ownership is unchanged. Functional readiness, base checks, host ESS policy and State installation stay staged until cff-5xdy.

Hostile zero/vacuous/improper/nonfinite, float32 and uncomfortable-scale behavior must retain baseline values or loud failure through existing authorities. Direct owner/vmap parity tests cover new typed boundaries; existing scientific oracles remain authoritative. No new solver, schedule, factor type, schema or performance claim. The cff-sf29 inventory and execution handoff are the source map; research agents remain unavailable, so local bounded inspection and executable comparisons supply the review evidence.

**2026-09-10T01:03:41Z**

First atomic extraction: execution/contracts.py defines immutable JAX-pytree ExecutionDelta with outgoing blocks and optional existing numerical residual/status arrays. None means no local solver fact, not convergence; exact support remains in outgoing values and existing downstream checks. execution/registered.py owns local discrete/Gaussian/EF updates from selected arrays, delegating the established discrete/Gaussian mathematical authority to kernel_runtime.py. Flooding, batch, registerized and fallback scan transports now call those owners. The registerized EF equality/registered-message duplicate is removed. No local owner accepts Arena or Engine, and prefetched scan blocks/carry/final scatter are retained.

Validation before this commit: 109 existing discrete/Gaussian/EF/schedule/Colony contract tests pass in 75.47 seconds. Twelve new direct-versus-jitted-vmap cases cover BP, VMP, Gaussian, EF prior, EF observations/masks and EF equality, each in float32/float64. They check output pytree/shape/dtype, absent solver facts and distinct runtime inputs; all 12 pass in 4.75 seconds. A later lambda-binding lint fix changes no numerical behavior. New modules/test pass Ruff and format checks; git diff --check passes. Dashboard enrollment and architecture/change-path documentation are updated. No final full suite yet; explicit composite/cross-domain and functional numerical owners are next.

**2026-09-10T01:23:08Z**

New P1 cff-jby3 directly reproduces shared hard-constraint data in Colony: exactly_k, at_most_k and same-shaped table values from the first instance govern both results, including false convergence when spawn order is reversed. The current local extraction must expose selected numeric threshold/table arguments to its explicit constraint executor while preserving existing outer binding and numerical rules. Existing JAX rules already accept those values as traced inputs. cff-jby3 then moves per-instance data ownership and fixes incompatible-shape grouping across execution, ledger, support/assignment and reconstruction; it is an additional prerequisite of cff-nyad. Do not claim that introducing the local typed API alone fixes the current shared-data defect.

**2026-09-10T01:56:39Z**

Explicit-owner extraction is now connected and passes 127 focused tests (4 slow deselections) in 145.17 seconds. This run includes execution, CSP, Dirichlet parameter edges, mixture/hybrid engine, hybrid GFE, structured recurrence, precision VMP, shared goals, mu3 and transition GFE. Source changes move local parameter selection and existing rule calls only.

execution/cross_domain.py owns mixture, recurrence and precision calls; composites.py owns Dirichlet, fixed/functional-statistic GFE, transition GFE and hybrid GFE; constraints.py accepts explicit numerical threshold/table data. parameters.py selects named array records. functional.py exposes the existing stationary and importance phases separately; Engine retains readiness/base checks, host ESS policy, installation and failure staging. The JAX log-zero predicate is moved verbatim to support.py and re-exported under the existing Engine-private alias. There is no new numeric guard or status default: absent local facts stay None, and existing residual/status arrays flow unchanged into the step reductions. Categorical/Gaussian/parameter/predictive outputs have distinct named records.

Global goal predictions remain collected from every original GFE group before one write_goal_counts call and before damping. Ledger formulas and once-only shared-goal/mu3 parameter energy remain in their existing owner. The current constraint defaults are deliberately not described as per-instance data; cff-jby3 owns that separate correction.

The execution package is 586 physical lines across nine files (largest 133); Engine is 3,717 lines. New modules pass Ruff/format; unused moved rule imports were removed and git diff --check passes. Next: direct specialized-owner/JAX failure-fact tests, bounded old/new execution comparison and final full suite. No final gate yet.

**2026-09-10T02:08:02Z**

Direct explicit-owner tests now pass: 39 cases in 42.62 seconds. Thirty cover 15 plan variants in float32/float64; three preserve existing divergence codes and non-finite residual facts under JIT/vmap; six show threshold/table arrays remain dynamic for sum- and max-product. Functional fixture moments are derived inputs, not compiler fields. Numerical residuals retain their original NaN/inf values rather than being rewritten by the adapter. The discovered malformed-particle gap is separately recorded in P2 cff-6n9c; public initial-State finite checks are not claimed broken.

Bounded trusted-source comparison against 1308688 completed using /tmp/cffg-execution-parity.py: 17 model fixtures, 68 auto/flood and float32/float64 configurations, and 2,336 array comparisons, all bitwise equal (including matched NaNs). Coverage includes initial state/parameter leaves, three damped update steps and their existing residual/status facts, free energy at each step, exact propagation when present, and the prepared mu3 numerical update. The script dynamically loads the old tracked Engine source with the unchanged compiled/rule modules; this is wiring parity, not an independent mathematical oracle or performance benchmark. Existing family tests remain authorities.

New execution modules and both execution test files pass Ruff/format checks; git diff --check and tk dep cycle pass. Dashboard enrollment includes both new files. Implementation is complete pending the one final default-suite gate. Independent agent review remains unavailable due to the reported agent usage limit.

**2026-09-10T02:31:02Z**

Completed the extraction and final default gate. At final code/test source b84560e (production extraction ab17b5a), all 1,894 tests passed, with 41 opt-in slow deselections and four existing Gym deprecation warnings. Pytest reported 1,050.10 seconds; dashboard measured 1,062.672 seconds wall time, completed 2026-09-10T02:25:45.542649+00:00, and recorded observed ticket-only head fe07e88. The new suggested wait budget is 1,350 seconds. Dashboard and runtime history are refreshed. No source/test changes followed this gate; opening the PR does not require a second run.

The implementation, 51 new local-boundary cases, prior focused family runs, and 2,336 bitwise old/new array comparisons are complete. No new mathematics, schedule, convergence strategy, public result/schema field, or throughput claim is included. The new execution package remains 586 lines; Engine is 3,717 lines. User-visible HTML explains the ownership changes separately from the preserved planning snapshot.

Next in the authorized architecture chain is P1 cff-jby3: remove shared hard-constraint numeric binding, preserve all four consumers and genuine shape-compatible Colony batching, and verify reconstruction. It blocks cff-nyad schedule extraction. P2 cff-6n9c and cff-34gi remain separate numerical validation work. Agent review was unavailable due to the reported usage limit; the PR reports local review and executable evidence only.
