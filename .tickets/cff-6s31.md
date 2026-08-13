---
id: cff-6s31
status: open
deps: [cff-cpxj, cff-d56d, cff-j7mm]
links: [cff-9xmq, cff-ahps]
created: 2026-07-09T22:04:18Z
type: feature
priority: 3
assignee: coopwilliams
---
# Plates phase 2: index-table expansion — compile without materializing copies

## Background in Plain English

Plate specifications keep repeated models compact on disk, but the phase-1 compiler expands every copy into Python variables, factors, ports, and blobs before it builds the JAX arena. That work wastes time and host memory because each copy has the same structure. This ticket makes the compiler derive group membership, slot addresses, parameter rows, wiring, and resource counts directly from the compact plate dimensions.

The compact path must keep phase-1 names and numerical meaning. The pixel grid, mixture bank, sensor array, resize edit, homogeneous HMM chain, tied parameters, per-copy parameters, constraints, schedules, serialization, and free-energy ledger must behave as they do after `expand_plates`. Remove the phase-1 nested-plate guard after recursive namespaces, blob inventories, compiled structure, and free energy work at nesting depth two or greater.

This ticket does not redesign plate wiring. `Broadcast` continues to emit the phase-1 equality chain recorded by cff-cpxj; cff-9xmq owns the balanced-tree replacement. Unrolling also remains separate because its time slices need not be homogeneous.

## Outcomes / Acceptance Criteria

1. The prerequisite contracts are satisfied before this ticket closes.
   1.1. cff-cpxj remains closed with its four phase-1 validation models serving as the semantic reference.
   1.2. cff-d56d remains closed with deterministic schedule construction and sealed schedule index tables.
   1.3. cff-j7mm is closed with exact named byte accounting for every compiled state arena.
2. Plate compilation uses a compact index plan instead of a materialized flat `Template`.
   2.1. `compile_template` compiles a template containing plates without calling `expand_plates`.
   2.2. A test trap on the phase-1 copy-scoping helper remains untriggered during compact plate compilation.
   2.3. Plateless templates continue through their current compiler path.
   2.4. The compact plan assigns copy ordinals in NumPy row-major order.
   2.5. The compact plan derives each copied variable name from its complete plate namespace and multidimensional index.
   2.6. The compact plan derives each copied factor name from its complete plate namespace and multidimensional index.
   2.7. The compact plan derives each exposed port name from its complete plate namespace and multidimensional index.
   2.8. One tied fragment blob produces one parent-level parameter identity across its plate copies.
   2.9. One per-copy fragment blob produces one indexed parameter identity per plate copy.
   2.10. The `{index}` token in a per-copy blob path resolves to the copy's complete multidimensional index.
   2.11. A copied constraint targets the corresponding scoped variable or factor.
   2.12. A learnable over tied blobs produces one compiled posterior plan that accumulates statistics from each copy.
   2.13. A learnable over per-copy blobs produces one compiled posterior plan per copy.
   2.14. A learnable that mixes tied and per-copy blobs fails with the phase-1 diagnostic.
   2.15. A `Chain` bond derives neighbor addresses from its declared axis and offset.
   2.16. A `Chain` bond leaves its phase-1 boundary ports exposed.
   2.17. A `Broadcast` with one copy derives the phase-1 direct hub edge.
   2.18. A `Broadcast` with two copies derives the phase-1 ternary equality node.
   2.19. A `Broadcast` with three or more copies derives the phase-1 linear equality chain.
   2.20. `Expose` derives the same indexed parent ports as phase 1.
   2.21. A generated-name collision fails before arena allocation.
   2.22. The generated-name collision diagnostic names the colliding namespace.
3. The compact compiler matches a test-only phase-1 expansion oracle.
   3.1. The compact and phase-1 paths produce the same sorted edge names.
   3.2. The compact and phase-1 paths produce equal `EdgeInfo` metadata for each edge.
   3.3. The compact and phase-1 paths produce bitwise-identical discrete message prefix tables.
   3.4. The compact and phase-1 paths produce bitwise-identical Gaussian eta prefix tables.
   3.5. The compact and phase-1 paths produce bitwise-identical Gaussian precision prefix tables.
   3.6. The compact and phase-1 paths produce bitwise-identical Gaussian observation prefix tables.
   3.7. The compact and phase-1 paths produce the same factor-group keys.
   3.8. The compact and phase-1 paths produce the same ordered factor members within each group.
   3.9. The compact and phase-1 paths produce bitwise-identical gather index arrays.
   3.10. The compact and phase-1 paths produce bitwise-identical scatter index arrays.
   3.11. The compact and phase-1 paths produce the same DATA-port maps.
   3.12. The compact and phase-1 paths produce the same parameter-stack keys.
   3.13. The compact and phase-1 paths produce bitwise-identical parameter-stack values.
   3.14. The compact and phase-1 paths produce bitwise-identical hard-constraint plans.
   3.15. The compact and phase-1 paths produce bitwise-identical Dirichlet parameter-edge plans.
   3.16. The compact and phase-1 paths produce bitwise-identical GFE composite plans.
   3.17. The compact and phase-1 paths produce bitwise-identical hybrid plans.
   3.18. The compact and phase-1 paths produce equal `member_of` mappings.
   3.19. The compact and phase-1 paths produce bitwise-identical learnable plans.
   3.20. The compact and phase-1 arenas produce the same schedule description.
   3.21. The compact and phase-1 arenas produce the same schedule content fingerprint.
   3.22. Compact compilation does not mutate its input template.
   3.23. Compact compilation leaves the input template's structure hash unchanged.
4. The phase-1 validation models retain their observable results.
   4.1. The pixel-grid model accepts the phase-1 aggregated image binding keys.
   4.2. The pixel-grid model binds the phase-1 observed values to those keys.
   4.3. Each pixel-grid latent marginal agrees with its phase-1 value within the stated dtype tolerance.
   4.4. The pixel-grid BFE agrees with its phase-1 value within the stated dtype tolerance.
   4.5. The mixture-bank model resolves the same per-copy parameter values as phase 1.
   4.6. The mixture-bank hub marginal agrees with its phase-1 value within the stated dtype tolerance.
   4.7. The mixture-bank BFE agrees with its phase-1 value within the stated dtype tolerance.
   4.8. The sensor-array posterior mean agrees with its phase-1 value within the stated dtype tolerance.
   4.9. The sensor-array posterior covariance agrees with its phase-1 value within the stated dtype tolerance.
   4.10. The sensor-array BFE agrees with its phase-1 value within the stated dtype tolerance.
   4.11. `SetPlateDims` changes the compact template's structure hash.
   4.12. `SetPlateDims` rebuilds arena index tables for the new dimensions.
   4.13. The resized plate's marginals agree with phase-1 expansion within the stated dtype tolerance.
   4.14. A homogeneous HMM chain plate produces the phase-1 smoothed marginals.
   4.15. A homogeneous HMM chain plate produces the phase-1 BFE.
   4.16. Each validation model returns the same `RunStatus` under compact and phase-1 compilation.
5. Recursive and N-dimensional plates use the same compact semantics.
   5.1. Validation accepts a plate fragment that contains another valid plate.
   5.2. Validation accepts a nonempty dimensions tuple with three or more axes.
   5.3. A nested copy's namespace includes each enclosing plate name and index in outer-to-inner order.
   5.4. Nested dimensions produce the Cartesian product of their copy counts.
   5.5. A blob tied inside the inner plate retains one identity within its declared enclosing scope.
   5.6. A per-copy blob inside the inner plate receives one identity for each complete outer-and-inner index.
   5.7. Nested `Chain` bonds address neighbors on the declared nesting level.
   5.8. Nested `Broadcast` bonds expose their hub on the declared nesting level.
   5.9. A depth-two plate produces the same arena structure as recursive phase-1 materialization in a test oracle.
   5.10. A depth-two plate produces the same marginals as recursive phase-1 materialization within the stated dtype tolerance.
   5.11. A depth-two plate produces the same BFE as recursive phase-1 materialization within the stated dtype tolerance.
   5.12. JSON round-trip preserves nested plate specifications.
   5.13. JSON round-trip preserves the nested template's structure hash.
   5.14. Serialized JSON contains no generated copy name from a nested plate.
   5.15. `.cffgz` round-trip preserves nested plate specifications.
   5.16. `.cffgz` round-trip preserves the nested template's structure hash.
   5.17. A bundled `model.json` contains no generated copy name from a nested plate.
   5.18. A `.cffgz` manifest inventories each nested tied blob once at its declared scope.
   5.19. A `.cffgz` manifest inventories each nested per-copy blob at its complete indexed identity.
   5.20. A template and arrays loaded from `.cffgz` reproduce the pre-save compact arena structure.
   5.21. Saving rejects a missing nested tied blob before creating the destination.
   5.22. Saving rejects a missing nested per-copy blob before creating the destination.
   5.23. Loading rejects a missing nested blob member.
   5.24. Loading rejects a tampered nested blob member.
   5.25. A fragment-level `AgentSpec` remains invalid because the complete host template owns the runtime interface.
6. Resource accounting and compile performance scale from the compact specification.
   6.1. The resource estimator accepts a compact plate template without compiling it first.
   6.2. The resource estimator obtains plate copy counts by integer arithmetic over `dims`.
   6.3. The resource estimator obtains `Chain`-generated edge and port counts without enumerating copies.
   6.4. The resource estimator obtains `Broadcast`-generated equality, edge, and port counts without enumerating copies.
   6.5. The resource estimator reports tied parameter bytes once per tied blob.
   6.6. The resource estimator reports per-copy parameter bytes once per indexed blob.
   6.7. The estimator's named persistent byte counts equal the compiled 64-by-64 pixel-grid arena's state leaves.
   6.8. The estimator's named parameter byte counts equal the compiled mixture-bank parameter leaves.
   6.9. The estimator's named state byte counts equal the compiled depth-two plate fixture's state leaves.
   6.10. The estimator's named parameter byte counts equal the compiled depth-two plate fixture's parameter leaves.
   6.11. A benchmark reports validation, compact planning, array resolution, arena assembly, and total compile time separately.
   6.12. The median total `compile_template` time for a 64-by-64 pixel grid is at most 1.5 seconds across five runs on the recorded M2 Max CPU/x64 benchmark host.
   6.13. Instrumentation confirms that the 64-by-64 compile allocates no copied `VariableSpec`, `FactorSpec`, `PortSpec`, or `ConstraintSpec` objects.
   6.14. Repeated compilation of the same compact template produces bitwise-identical index tables.
7. Compact plate planning obeys the numerical robustness contract.
   7.1. A sparse discrete plate preserves the exact `NEG` sentinel on excluded states.
   7.2. A discrete plate with empty support reports `INFEASIBLE` under compact compilation.
   7.3. An unobserved Gaussian plate preserves the exact vacuous message `(eta, Lambda) = (0, 0)`.
   7.4. A Gaussian plate with an improper required belief returns the established improper-belief outputs.
   7.5. A non-finite plate message produces `DIVERGED` rather than `CONVERGED`.
   7.6. A non-finite plate BFE produces `DIVERGED` rather than `CONVERGED`.
   7.7. Float32 discrete messages agree between compact and phase-1 compilation within the stated float32 tolerance.
   7.8. Float32 Gaussian messages agree between compact and phase-1 compilation within the stated float32 tolerance.
   7.9. The 64-by-64 pixel grid is the uncomfortable-copy-count fixture.
   7.10. Dimensions containing zero or a negative value fail before planning.
   7.11. A `Chain` axis outside the dimensions tuple fails before planning.
   7.12. A `Chain` offset below one fails before planning.
   7.13. A copy-count product above `numpy.iinfo(numpy.int64).max` fails before allocation.
   7.14. The copy-count overflow diagnostic reports the dimensions.
   7.15. A generated slot address outside its arena prefix table fails compilation.
   7.16. The implementation introduces no numeric floor, ridge, or clamp because plate planning performs structural integer arithmetic only.
8. Every affected execution and persistence surface is checked.
   8.1. Discrete BP plate messages agree between host and JIT execution within the stated dtype tolerance.
   8.2. Gaussian BP plate messages agree between host and JIT execution within the stated dtype tolerance.
   8.3. Plate marginals agree between flooding and scheduled execution within the stated dtype tolerance.
   8.4. Plate BFE agrees between flooding and scheduled execution within the stated dtype tolerance.
   8.5. Sequential and `vmap` plate execution agree within the stated dtype tolerance.
   8.6. Sum-product plate compilation preserves the phase-1 arena.
   8.7. Sum-product plate compilation preserves the phase-1 BFE.
   8.8. Max-product plate compilation preserves the phase-1 arena.
   8.9. Max-product plate compilation preserves the phase-1 decoded assignment.
   8.10. A current-schema checkpoint restores into an arena compiled from the same compact plate template.
   8.11. Legacy checkpoint decoding remains unchanged because this ticket adds no checkpoint field or migration.
   8.12. Current JSON decoding remains unchanged because this ticket adds no serialized IR field.
   8.13. Plateless `.cffgz` loading remains unchanged.
   8.14. The implementation adds no runtime dependency.
   8.15. The implementation adds no development dependency.
9. Durable documentation and executable evidence cover the completed change.
   9.1. `docs/ARCHITECTURE.md` describes compact plate planning between validation and arena assembly.
   9.2. `docs/ARCHITECTURE.md` identifies `PlateSpec` as the source of truth and the arena index plan as derived state.
   9.3. `README.md` documents the unchanged plate modeling API.
   9.4. `README.md` documents the 64-by-64 compact-compilation benchmark result.
   9.5. `docs/LIMITATIONS.md` removes the phase-1 nested-plate restriction after recursive evidence passes.
   9.6. `docs/LIMITATIONS.md` continues to identify linear `Broadcast` expansion as cff-9xmq work.
   9.7. Focused plate, compiler, bundle, estimator, schedule, serialization, learning, runtime-status, and hostile-input tests pass.
   9.8. The dashboard-backed short suite passes after the final code change.

## Methodological Suggestions

These suggestions are optional. Another method is acceptable if it satisfies the numbered outcomes with equivalent artifacts and executable evidence.

- Use the **New IR field** row in `docs/ARCHITECTURE.md` as the closest change path for the existing `PlateSpec` authority. The schema is unchanged, but validation, serialization and hashing, plate scoping, compiler plans, bundles, learnables, estimator accounting, and runtime consumers still need review.
- Introduce a compact read-only grounding view that yields scoped specifications and contiguous index ranges without constructing one dataclass graph per copy. Keep the phase-1 `expand_plates` function as a test oracle until parity is established.
- Derive group rows in plate-major row-major blocks, then concatenate them with the host template's ordinary rows before the existing arena builders run. This keeps JAX rule code independent of whether a row came from a plate.
- Represent nested scopes as an index-product plan whose name rendering and blob rendering share one authoritative path. Reuse that path for bundle inventories and resource estimates.
- Compare the arena projection listed in criterion 3 instead of comparing dataclass object identity.
- Benchmark `compile_template` in a dedicated process on the recorded M2 Max CPU/x64 host. Report medians and each phase separately so array I/O or JAX initialization cannot masquerade as compact-planning cost.
- Check the deliberate execution pairs: host/JIT, flooding/scheduled, single-instance/`vmap`, and current/legacy checkpoint decoding. Record why a pair is structurally irrelevant if one does not apply.
- Run focused tests during implementation. After the final code change, run the dashboard-backed short suite once under the `AGENTS.md` protocol.

## Future work

- cff-9xmq owns replacement of the phase-1 linear `Broadcast` equality chain with a balanced tree after its cff-wbhu prerequisite.
- cff-ahps no longer requires compact plate lowering for closure. This ticket
  remains an optional scaling extension after its estimator prerequisite.
- Unrolling remains a separate mechanism under the decision recorded in cff-cpxj; no migration ticket is identified.
- No other future work is identified.

## Definitions

- **Arena:** Flat, shape-segmented JAX state arrays plus static group and addressing plans derived by the compiler.
- **Bethe free energy (BFE):** The factor-local objective terms plus edge-entropy overcounting corrections used for convergence and model scoring.
- **Compile benchmark:** Five timed `compile_template` calls after imports in one dedicated process, with no JIT execution included; the reported result is the median total wall time.
- **Compact grounding view:** A compiler input that represents repeated scoped graph items by a fragment and index ranges instead of copied Python graph objects.
- **Copy ordinal:** One integer position assigned to a multidimensional plate index in row-major order.
- **Equality chain:** The phase-1 sequence of ternary equality factors synthesized to connect `Broadcast` branches to one hub.
- **Grounding:** Resolving a reusable template into concrete names, connections, parameters, constraints, and compiler rows.
- **Host/JIT pair:** Direct Python-host execution compared with JAX-compiled execution of the same rule path.
- **Index table:** A static integer array that tells compiled rules which arena slots to gather or scatter.
- **Nested plate:** A plate whose fragment contains another `PlateSpec`.
- **Per-copy blob:** A parameter array whose identity is indexed separately for each plate copy.
- **Plate:** A compact declaration that repeats one homogeneous template fragment across declared dimensions.
- **Row-major order:** The last dimensions index changes fastest, matching `numpy.ndindex`.
- **Schedule content fingerprint:** A digest that covers each executable schedule stage and its complete row-index tables.
- **Structure hash:** The deterministic serialized-model identifier; it covers plate specifications and dimensions but excludes parameter values and blob paths.
- **Stated dtype tolerance:** `atol=rtol=1e-10` for float64 and `atol=rtol=1e-5` for float32 unless a retained phase-1 test sets a stricter bound.
- **Tied blob:** One parameter identity shared by each copy in its declared plate scope.
- **`vmap`:** JAX's transformation for applying one compiled function across a batch.

## Notes

**2026-07-11T18:30:45Z**

Follow-up from cff-vamy/PR #18: phase-1 validation now rejects any PlateSpec whose fragment declares plates, because expand_plates does not recurse and would otherwise ground an empty/incomplete model. When implementing phase-2 index-table grounding, deliberately lift this guard only after recursive plate namespaces, tied/per-copy blob resolution, bundle inventories, arena parity, and FE parity are covered at depth >= 2.

**2026-08-08T04:51:13Z**

Intelligible-ticket review retained 38 scan candidates because their local meanings are bounded: `preserve` means exact serialized/value identity or the named unchanged behavior; `reusable` means a repeated fragment or shared helper; `same` means value or artifact equality under the stated comparison; `expose` means the `Expose` wiring rule or a derived parent port; `implement` means the work required by this ticket or the preserved historical plan; `support` means feasible probability states; `complete` means an exhaustive namespace/index or acceptance completion; `remaining` means a named preserved or excluded behavior; `close` means ticket closure after the listed prerequisites and outcomes; `check` means the named execution-pair or artifact comparison; `converge` means the explicit runtime status; `equivalent` means acceptance artifacts that prove the numbered outcome; `estimate` means the resource estimator; `evidence` means named tests, projections, hashes, or benchmark output; `exact` means literal sentinel/value identity or exact byte accounting; `match` means equality with the phase-1 reference; `oracle` means the test-only phase-1 expansion path; `parity` means tolerance or structural equality in the preserved note; `pass` means automated test success or message-passing terminology; `reproduce` means equal pre-save arena structure; `robust` means the repository numerical-robustness contract; `test` means a named automated fixture, trap, oracle, or selection; `validate` means PlateSpec input validation; `accumulate` means statistics from distinct plate copies entering one tied posterior; `learn` refers to the focused learning test surface; `learnable` means the declared parameter artifact and its compiled plan; `performance` means measured compile wall time; `scale` means increasing plate copy count; `speed` means row-major index ordering; `explicit_owner` identifies cff-9xmq or the complete host Template as stated; `follow_up` identifies the completed cff-vamy review note; `named_ticket_handoff` assigns balanced Broadcast work to cff-9xmq; `any` is the universal quantifier in the preserved historical note; `every` distributes over the locally named copies or arena leaves; `only` marks the test-oracle scope or the guard-lifting condition; `current` means the schema, checkpoint, or base-revision compiler path active when this ticket was rewritten; `existing` means the base-revision PlateSpec or arena builder; and `read` is part of `read-only`, meaning the suggested compiler view does not mutate its source. Capability ownership was checked separately: cff-6s31 owns compact plate compilation, recursive and N-dimensional plate grounding, analytic plate resource counts, and phase-1 parity; cff-j7mm owns the prerequisite exact compiled-arena estimator; cff-9xmq owns balanced Broadcast topology; cff-ahps owns the downstream roadmap; unrolling stays separate under cff-cpxj's recorded decision.

Intelligible-ticket review: complete.
