---
id: cff-jby3
status: closed
deps: [cff-ny7f]
links: [cff-ii9t, cff-sf29]
created: 2026-09-10T01:20:57Z
type: bug
priority: 1
assignee: coopwilliams
tags: [engine, colony, constraints, parameters]
---
# Keep hard-constraint parameters separate between Colony instances

## Background in Plain English — WHY

Colony can run the wrong hard constraint for a graph. Two instances with the same template but different exactly_k limits share one cached Engine. The Engine reads the first instance's compiled limit instead of the second instance's data. In a two-variable probe, observations (1, 0) are feasible for k=1 and infeasible for k=0. Spawning k=0 first makes both batched results infeasible, although a standalone k=1 Engine converges.

Constraint limits and allowed-tuple tables currently live in ConstraintPlan, not State.params. The extraction must not hide that shared-data defect inside a local-update adapter. Existing count and table rules already accept their values as numerical JAX arguments; a bounded probe confirmed distinct outputs under jit without changing the rules.

## Outcomes / Acceptance Criteria — WHAT

1. Two same-template instances with different exactly_k limits produce their respective standalone states and results regardless of spawn order.
2. The same independence holds for at_most_k limits.
3. Two same-template instances with different same-shaped allowed-tuple tables produce their respective standalone states and results.
4. Compatible same-shaped instances retain one cached Engine and actually enter the vectorized update path; sequential fallback is not a substitute.
5. Instances with incompatible numeric array shapes are not stacked together, and each still runs correctly without changing the public template structure hash.
6. Message updates consume the current instance's constraint values.
7. Free-energy evaluation consumes those same values.
8. Support checks and decoded-assignment validation consume those same values.
9. New execution owners accept selected numerical constraint inputs without capturing another instance's values in a shared plan.
10. Current serialization and supported legacy checkpoints reconstruct the constraint values from the template and data. No callable, unsafe object encoding, or new public model field is introduced.
11. Checkpoints carrying fixed-parameter identity metadata still reject different constraint data. Supported legacy formats without that metadata retain their documented limits; the repair does not claim they detect a data mismatch.
12. Focused regressions cover both spawn orders, healthy and infeasible peers, exact zeros, float32/float64, and different table shapes including at least ten times the smallest table's row count.
13. The existing enumeration and constraint-rule tests remain the mathematical authority; no counting algorithm, energy formula, or solver changes.
14. The affected change paths are Runtime numerical or status invariant, New factor or rule family, and Learnable or checkpoint feature. Compiler declaration/packing/resolution, initial State parameters, local execution, ledger, support/assignment consumers, Colony grouping and checkpoint reconstruction are checked explicitly before implementation.
15. Focused checks and one final dashboard-backed short suite pass, and ARCHITECTURE describes the delivered data and batching ownership.

## Methodological Suggestions — HOW

These are optional suggestions; another sound method is acceptable.

- Publish constraint values as per-instance numerical parameter arrays and select them through the local executor's typed input boundary.
- Distinguish compatible batch shapes from Engine identity. A cached numerical program can accept different values; incompatible array shapes cannot be stacked into one batch.
- Inspect every remaining read of ConstraintPlan.k and ConstraintPlan.table, including objective and assignment checks, before declaring the repair complete.
- Reuse the model-data reconstruction and static-parameter-hash checks already present in the agent checkpoint code. Do not assume a schema migration is needed merely because derived runtime parameter keys change.
- Sparse/zero support must retain exact impossibility. Vacuous inputs retain the existing rule behavior; improper continuous beliefs are N/A to these discrete constraints. Exercise both float precisions and larger table shapes. No new numerical guard is expected.

## Future work

cff-ny7f first extracts the local numerical boundary while preserving the existing data binding. This repair follows it and precedes cff-nyad schedule-construction extraction. cff-ii9t later unifies run/result policy; cff-wbhu owns compiling convergence loops. No broad CSP algorithm or performance benchmark is added here.

## Definitions

- **Colony:** The service that groups graph instances for vectorized execution and reuses compiled Engines.
- **exactly_k / at_most_k:** Constraints requiring exactly or at most k of their binary variables to equal one.
- **Allowed-tuple table:** Rows listing the joint variable assignments that a hard constraint permits.
- **ConstraintPlan:** The immutable compiled description of a hard-constraint factor.
- **State.params:** One graph instance's numerical runtime parameter arrays.
- **jit:** Just-in-time compilation of array computations.
- **Vectorized update:** One numerical program applied to multiple independently stored graph states.
- **Float32 / float64:** Floating-point numbers stored in 32 or 64 bits.
- **CSP:** Constraint satisfaction problem.



## Notes

**2026-09-10T01:23:08Z**

Confirmed for all three affected families at source 8932515: exactly_k, at_most_k and table. Each two-variable template has one hard factor and two observed DATA ports set to (0,1). Count values are 0 and 1; tables are [[0,0],[1,1]] and [[0,1],[1,0]]. Each pair uses one cached Engine and an empty runtime-parameter dictionary. With the infeasible model spawned first, Colony reports [infeasible,infeasible] while standalone Engines report [infeasible,converged]. Reversing spawn order gives Colony [converged,converged] versus standalone [converged,infeasible]. Thus this can falsely certify a model whose actual constraint is violated, not merely reject a feasible model.

Affected source is unchanged from the pre-extraction repair baseline 1308688: colony.py cache/spawn; compiled.py ConstraintPlan; csp.py numerical rules; compiler.py. The local constraint, ledger and assignment paths in engine.py still read plan.k/table. Anchor refs at 8932515: compiled.py:91-107; colony.py:39-70; engine.py:2442-2446,2698-2700,3789-3791. Confidence: high, executable full-Engine/Colony probes in both orders.

Dynamic-input feasibility probe: existing exactly_k_messages and table_messages each accept their threshold/table as a numerical jitted argument with incoming shape (2,2), returning distinct (2,2) outputs for the two values above. No rule change was needed. An earlier exploratory table call used an incorrect extra batch dimension; that shape error is not a dynamic-parameter limitation. Current agent checkpoint code already hashes fixed model-data blobs (static_parameter_hash, _agent_checkpoint.py:216-249), so preserve that identity check while deriving runtime constraint parameters from model data.

Legend: prerequisite --> dependent.
[x] cff-sf29 contracts
  --> [~] cff-ny7f local execution (ready/in progress)
        --> [ ] cff-jby3 constraint-data isolation
              --> [ ] cff-nyad schedule construction
                    --> cff-5xdy lifecycle --> cff-ii9t control
                          --> cff-jrbw --> cff-gl89 --> cff-3ola --> cff-ws3c

cff-nyad also retains its direct dependency on cff-ny7f. This tree is based on tk dep tree --full cff-nyad, tk ready and tk blocked; tk dep cycle passes.

**2026-09-10T02:12:40Z**

Bounded pre-implementation inspection at cff-ny7f source b84560e confirms four consumers of compiled constraint data: Engine._has_zero_support (:1734-1757), assignment_energy/_constraint_ok (:2432-2446), _constraint_writes via execution.parameters.constraint_defaults (:2615-2617), and _make_bfe (:3650-3656). Constraint-bearing models are excluded from _make_support_check (:3318); exact-path JIT combinatorial support is therefore N/A, not an omitted parallel implementation. Auto mixed and flood updates, host feasibility, sum-product FE, max-product assignment, solo and genuine vmap execution all remain in scope.

Compilation map: specialized/constraints.py resolves and validates threshold/table payloads into ConstraintPlan; resolution/resolve.py:149-154 currently collects those plans without adding runtime numeric data. Move the validated values into the published initial parameter map and have runtime selection read State.params. No numerical rule rewrite is needed. Colony.spawn already installs each freshly compiled arena.log_params for a reused Engine, but Colony.run groups solely by structure hash and immediately stacks every State. Group compatible State shapes/dtypes within each structure; do not put numeric contents into the public structure hash or lose same-shaped vmap reuse.

Checkpoint map: _agent_checkpoint.static_parameter_hash(:216) hashes grounded fixed blobs; _planning_inputs(:595) starts from plan.init_state before installing restored learnables/beliefs. Current and supported legacy reconstruction must be exercised with constraint data, and current fixed-data mismatch checks preserved. A change to derived State parameter keys is not by itself a schema change. Check remaining compiled-value tests explicitly when removing or redefining the static k/table fields; do not silently preserve runtime reads of those old fields. Read-only preparation only; implementation follows the completed cff-ny7f gate and updated base.

**2026-09-10T02:14:28Z**

Checkpoint acceptance clarification: strict fixed-data hashes are checked by _validate_versioned_envelope (:337-343); the old agent:version=1 and raw legacy decoders predate strict identity. Criterion 11 now preserves that actual contract rather than claiming a missing legacy hash can detect changed input data. Test current mismatch rejection and supported legacy reconstruction separately. Also update README Colony caching wording when shape-compatible batching lands: one Engine object can hold several JAX shape/dtype specializations; one cached Engine must not be described as universally one numerical compilation.

**2026-09-10T02:34:14Z**

Starting from updated published PR #124 head 8f8aa6f82363270a3b345401235d7b8fcd2fa1e9. The extraction gate passed 1,894 default tests at code/test source b84560e, recorded in a7e87c7. Primary change paths and all four consumers are mapped above. Keep ConstraintPlan structural: validated integer thresholds and allowed-tuple arrays will move to initial runtime parameters, with explicit family-prefixed keys and no runtime fallback to a captured plan value. A typed constraint-resolution result will carry the plan and its initial parameter mapping. Current numerical rule functions are unchanged. Compatible State shapes/dtypes will batch under one cached Engine per public structure hash; incompatible table lengths get separate batches, not fabricated padding or sequential fallback for compatible peers.

Tests first reproduce the existing wrong-data mechanism, then verify both spawn orders, solo/result parity, actual vmap for compatible healthy peers, sum/max consumers, both float precisions, tenfold table size, and strict versus legacy reconstruction. New IR fields are N/A; derived compiled fields and State parameter keys intentionally change to correct data ownership. The exact-path JIT support check remains N/A for constraints, as confirmed in its admission guard. No solver, CSP numerical formula, public model hash, schema or public status changes.

**2026-09-10T02:48:48Z**

Implementation checkpoint: the new regression failed before the repair with the second k=1 graph receiving k=0 messages (4 of 8 message entries wrong, including exact-zero support). ConstraintPlan now holds only structure and a family-prefixed runtime parameter key. A typed ResolvedConstraint publishes validated integer payloads into initial State parameters; collisions with another family are rejected. Local updates, free energy, Engine.objective (the actual decoded-assignment API; the earlier note called it assignment_energy), and host feasibility all select current State data. Colony partitions each structure by pytree layout, array shape and dtype while retaining one Engine. No CSP rule changed.

Current focused evidence: 168 compiler/CSP/constraint checks passed before the last extra collision case; all 71 new tests now pass, including 48 two-instance cases, real same-shaped vmap, one Colony mixing both precisions and 2/20-row tables, six sum/max consumer checks, collision rejection, and 15 bundle/analytic checkpoint checks. New/changed small modules pass Ruff/format. The 23 Engine and five Colony Ruff findings match the exact pre-repair baseline; no blanket cleanup was made. The old LIMITATIONS S10 workaround has been removed from the working docs because the new runtime paths are fixed.

One compatibility edge remains before the final gate: prepared functional-mu3 checkpoints serialize every State parameter, whereas analytic envelopes reconstruct initial parameters. _agent_mu3.restore_runtime (:280-294) and _load_state (:398-421) demand the current exact parameter inventory, so an older prepared cache will lack the newly derived constraint entries. Verify this with a fixed-particle cache regression, then migrate only the exact pre-constraint parameter layout from the supplied model data, retaining all other missing/extra-field, dtype, fixed-identity and cache checks. Current analytic, raw legacy analytic, and agent:version=1 analytic restoration already pass. No final default gate yet; no source changes committed yet.

**2026-09-10T03:02:34Z**

Implementation and focused verification are complete. All 152 tests in the final constraint/reconstruction/Colony/engine-contract/local-owner selection passed in 134.68 seconds, with three slow deselections. The two new files contribute 80 cases. They include current prepared mu3 caches plus exact older cached-parameter inventories under strict and agent:version=1 envelopes, with bitwise result/key preservation and no resampling. Regressions also retain rejection of missing declared constraint fields, missing messages and extra arrays. The older inventory was reproduced as a real rejection before the migration. Both the inner runtime inventory and the outer strict-envelope re-encoding check now recognize the same narrow legacy layout through _constraint_checkpoint.py; every other decoder check remains in force.

Thirty existing checkpoint/restore-focused cases passed in 249.54 seconds (nine deselections), before the final outer-inventory adjustment; the upcoming default gate will validate that final source against all existing cases. Earlier compiler/CSP focused evidence is 168 passing tests. CSP/rules source files are unchanged. The only compiler characterizations updated are the intentionally obsolete plan.k/table storage assertions; endpoint order, public model hashes and numerical authorities remain unchanged.

Ruff passes for the new helper, changed small modules, compiled values and new tests. Baseline findings are unchanged by code/message multiset: Engine 23, Colony 5, _agent_mu3 7, _agent_checkpoint 3. New/changed small-file formatting, git diff --check and tk dep cycle pass. Architecture, README batching/checkpoint descriptions, change-path test anchors, and dashboard enrollment are updated. LIMITATIONS S10 is removed because the defect is repaired. No final full-suite claim yet; run it once after this final source commit and use its measured timing rather than repeatedly polling.

**2026-09-10T03:22:54Z**

Final default gate passed at code/test source 84aa318: 1,974 tests passed, 41 slow tests deselected, and four existing Gym deprecation warnings; pytest reported 1,119.00 seconds and measured wall time was 1,132.293 seconds. Completion was 2026-09-10T03:21:26.867992+00:00, recorded against ea82c53. The dashboard was refreshed and includes the 80 new constraint cases. The only intervening commit ea82c53 records the next ticket handoff; no source/test change supersedes this result. The suite was checked once, after the expected completion window, and will not be rerun for publication.

All acceptance criteria are satisfied. The numerical constraint rules and public structure hash are unchanged; instance parameters, compatible vectorized batches and strict/legacy reconstruction are covered by the final gate. Close this ticket and publish the next PR above #124; cff-nyad is the next ready architecture step. Its note already maps the exact schedule boundary and preserved hostile-input/transport tests. Independent agent review remains unavailable at the reported usage limit; no such review is claimed.
