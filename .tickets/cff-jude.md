---
id: cff-jude
status: open
deps: [cff-t6ec, cff-d56d]
links: []
created: 2026-07-08T02:57:59Z
type: task
priority: 3
assignee: coopwilliams
parent: cff-ql2n
---
# viz: message-passing schedule overlay (Figs 23/24/27)

## Background in Plain English

Part I Figures 23, 24, and 27 annotate CFFGs with arrows showing which way messages travel; Figures 24 and 27 highlight messages produced by P-substituted GFE nodes in red. The repository already renders the underlying diagram as typed paths carrying `var:` and `blob:` references, and the closed cff-d56d prerequisite supplies sealed `two_pass`, `colored`, `mixed`, and `flood` engine schedules. The remaining work is a deterministic bridge from compiled schedule writes to directed arrow primitives on those paths.

An engine schedule is a reusable execution plan rather than a record of one completed run. Exact and scanned stages name factor-group rows; colored and mixed schedules separate one-time priming from a repeatable iteration; mixed and flooding execution also contain synchronous updates represented by arena plan tables rather than `BatchStage` rows. This ticket exposes those structural events without reading message values, convergence history, or runtime state.

The default full overlay follows the papers by showing message directions, coalescing repeated writes of the same directed message while retaining their logical step identities in typed metadata. A caller can select one logical step to inspect exact ordering. Ordinary scheduled messages use the diagram stroke color, while messages whose update rule belongs to a compiled GFE plan use the paper's red highlight and remain identifiable without relying on color alone.

## Outcomes / Acceptance Criteria

1. A typed overlay is derived from one Template and one validated Engine.
   1.1. `derive_schedule_overlay(template, engine)` is exported from `cffg.viz`.
   1.2. The returned `ScheduleOverlay` records `engine.schedule.kind`.
   1.3. The returned `ScheduleOverlay` records `engine.schedule.reason`.
   1.4. The returned `ScheduleOverlay` contains an ordered tuple of `MessageEvent` values.
   1.5. Each `MessageEvent` records one `var:<name>` wire reference.
   1.6. Each `MessageEvent` records one source `factor:<name>` reference.
   1.7. Each `MessageEvent` records one target factor or terminal reference.
   1.8. Each `MessageEvent` records the source factor role.
   1.9. Each `MessageEvent` records its discrete, Gaussian, or exponential-family arena.
   1.10. Each `MessageEvent` records its phase identity.
   1.11. Each `MessageEvent` records its logical step identity.
   1.12. Each `MessageEvent` records its simultaneous batch identity.
   1.13. Each `MessageEvent` records whether its update is GFE-based.
   1.14. Overlay derivation rejects a template missing a scheduled factor name.
   1.15. Overlay derivation rejects a template whose factor-role wire disagrees with the arena scatter slot.
   1.16. Overlay derivation accepts an Engine carrying its genuine sealed schedule.
   1.17. Overlay derivation does not rebuild or mutate the Engine schedule.
   1.18. Overlay derivation does not read an Engine state.
   1.19. Repeated derivation from identical inputs returns value-equal overlay data.

2. Schedule stages expand into the message writes performed by one schedule plan unit.
   2.1. A `BatchStage` row resolves through the group member tuple to one factor name.
   2.2. A discrete `BatchStage` resolves through `Arena.groups`.
   2.3. A Gaussian `BatchStage` resolves through `Arena.gauss_groups`.
   2.4. An exponential-family `BatchStage` resolves through `Arena.ef_groups`.
   2.5. One factor firing produces one event for each message-carrying group role.
   2.6. A role's scatter slot resolves to the matching `Arena.edges` entry in its arena.
   2.7. A scatter slot equal to `EdgeInfo.slot0` identifies the edge's first compiled endpoint as the source.
   2.8. A scatter slot equal to `EdgeInfo.slot1` identifies the edge's second compiled endpoint as the source.
   2.9. A `ScanStage` expands scan steps in ascending index order.
   2.10. A `ScanStage` expands its elements in tuple order within one scan step.
   2.11. A `ScanElem` uses `rows[step]` for its logical scan step.
   2.12. Rows inside one `BatchStage` share one simultaneous batch identity.
   2.13. A two-pass overlay expands every stage in `Schedule.stages` once.
   2.14. A colored overlay labels `Schedule.prime_stages` as a one-time prime phase.
   2.15. A colored overlay labels `Schedule.stages` as one repeatable colored iteration.
   2.16. A mixed overlay labels `Schedule.prime_stages` as a one-time prime phase.
   2.17. A mixed overlay labels the implicit synchronous update as one repeatable iterative phase.
   2.18. A mixed overlay labels `Schedule.stages` as the repeatable island-refresh phase after the iterative phase.
   2.19. A flooding overlay represents one repeatable synchronous sweep.
   2.20. Discrete mean-field group scatter tables contribute events to the implicit iterative phase.
   2.21. Dirichlet group scatter tables contribute events to the implicit iterative phase.
   2.22. Constraint group scatter tables contribute events to the implicit iterative phase.
   2.23. Discrete GFE group scatter tables contribute events to the implicit iterative phase.
   2.24. Transition-GFE group scatter tables contribute events to the implicit iterative phase.
   2.25. Hybrid-GFE group scatter tables contribute events to the implicit iterative phase.
   2.26. Hybrid-mixture group scatter tables contribute events to the implicit iterative phase.
   2.27. Recurrence group scatter tables contribute events to the implicit iterative phase.
   2.28. Gaussian-precision group scatter tables contribute events to the implicit iterative phase.
   2.29. Flooding includes ordinary discrete factor scatter tables.
   2.30. Flooding includes ordinary Gaussian factor scatter tables.
   2.31. Flooding includes ordinary exponential-family factor scatter tables.
   2.32. A DATA clamp does not create a scheduled factor-write event.
   2.33. A blob-backed pointmass wire does not create a scheduled factor-write event.
   2.34. Delta hardening does not create a directed message event.
   2.35. Damping does not change the overlay event set.

3. Direction and GFE classification come from compiled ownership rather than display text.
   3.1. An event arrow points away from its source factor endpoint.
   3.2. An event arrow points toward its target endpoint.
   3.3. Reversing the compiled owning slot reverses the event direction on the same wire.
   3.4. A message emitted by an `Arena.gfe_groups` member is classified as GFE-based.
   3.5. A message emitted by an `Arena.gfe_tm_groups` member is classified as GFE-based.
   3.6. A message emitted by an `Arena.hybrid_gfe_groups` member is classified as GFE-based.
   3.7. A GFE classifier does not inspect a factor glyph.
   3.8. A GFE classifier does not inspect a factor display label.
   3.9. A GFE classifier does not infer from `Schedule.reason` alone.
   3.10. An ordinary BP event inside a mixed schedule is not classified as GFE-based.
   3.11. An ordinary VMP event inside a mixed schedule is not classified as GFE-based.
   3.12. Both structural outputs written by a discrete observation GFE plan are classified as GFE-based.
   3.13. Every structural output written by a transition-GFE plan is classified as GFE-based.
   3.14. Every structural output written by a hybrid-GFE plan is classified as GFE-based.
   3.15. Internal stochastic evaluator work that writes no message slot creates no arrow.

4. Scene and SVG layers render inspectable arrows without changing the base diagram.
   4.1. The typed scene graph adds a `MessageArrow` primitive.
   4.2. A `MessageArrow` carries its wire reference.
   4.3. A `MessageArrow` carries its source factor reference.
   4.4. A `MessageArrow` carries its target reference.
   4.5. A `MessageArrow` carries its GFE classification.
   4.6. A `MessageArrow` carries the logical step identities represented by that primitive.
   4.7. A full overlay coalesces events with the same wire, source role, target, and GFE classification into one `MessageArrow`.
   4.8. Coalescing retains every represented logical step identity.
   4.9. `ScheduleOverlay.select_step(step)` returns only events for the named logical step.
   4.10. A step-selected scene emits one arrow per selected directed event.
   4.11. Opposing arrows on one wire use distinct arrowhead positions.
   4.12. An arrowhead lies on its referenced wire path.
   4.13. An arrowhead tangent agrees with its event direction.
   4.14. An arrowhead extent is disjoint from each incident factor-box interior.
   4.15. An arrowhead extent is disjoint from a variational bead on its wire.
   4.16. An arrowhead extent is disjoint from its wire label extent.
   4.17. An arrowhead extent is disjoint from a terminal glyph.
   4.18. Arrow placement works on a straight wire path.
   4.19. Arrow placement works on a multi-segment rectilinear wire path.
   4.20. Ordinary arrows use the diagram stroke color.
   4.21. GFE arrows use the exact SVG color `#c62828`.
   4.22. GFE classification is present in the arrow's typed metadata.
   4.23. GFE classification is present in the arrow's SVG title.
   4.24. The SVG title names the source factor.
   4.25. The SVG title names the target reference.
   4.26. The SVG title names the represented logical steps.
   4.27. SVG marker identifiers are deterministic.
   4.28. SVG marker identifiers are unique within one document.
   4.29. `build_scene(template, overlay=overlay)` includes schedule arrows.
   4.30. `render_svg(template, overlay=overlay)` includes schedule arrows.
   4.31. `save_svg(template, path, overlay=overlay)` includes schedule arrows.
   4.32. `save_png(template, path, overlay=overlay)` forwards the overlay to SVG generation.
   4.33. Calling `build_scene` without an overlay returns value-equal pre-ticket scene data.
   4.34. Calling `render_svg` without an overlay returns byte-identical pre-ticket SVG.
   4.35. Base diagram paths retain their pre-ticket references when an overlay is present.
   4.36. Base diagram boxes retain their pre-ticket references when an overlay is present.

5. Schedule families, paper examples, compression, and scale have executable evidence.
   5.1. An acyclic HMM fixture produces a two-pass overlay.
   5.2. The two-pass fixture contains a message event from each scheduled factor firing.
   5.3. The two-pass fixture preserves inward logical step order.
   5.4. The two-pass fixture preserves outward logical step order.
   5.5. A long acyclic HMM fixture exercises `ScanStage` expansion.
   5.6. The expanded scan event count equals the unrolled row-table count times message-carrying roles.
   5.7. A cyclic BP fixture produces a colored overlay.
   5.8. The colored fixture distinguishes its prime phase.
   5.9. The colored fixture distinguishes its repeatable color phase.
   5.10. A P-substituted rollout fixture produces a mixed overlay.
   5.11. The mixed fixture contains ordinary black arrows.
   5.12. The mixed fixture contains GFE red arrows.
   5.13. A delta-constrained fixture produces a flooding overlay.
   5.14. The flooding fixture represents one synchronous sweep.
   5.15. The flooding fixture records its schedule reason.
   5.16. The Figure 23 companion fixture renders forward EFE message directions.
   5.17. The Figure 24 companion fixture renders backward GFE message directions in red.
   5.18. The Figure 27 T-maze fixture renders its GFE message directions in red.
   5.19. The acyclic HMM overlay renders at compression level 0.
   5.20. The acyclic HMM overlay renders at compression level 1.
   5.21. The acyclic HMM overlay renders at compression level 2.
   5.22. The acyclic HMM overlay renders at compression level 3.
   5.23. The acyclic HMM overlay renders at compression level 4.
   5.24. The mixed rollout overlay renders at compression level 0.
   5.25. The mixed rollout overlay renders at compression level 1.
   5.26. The mixed rollout overlay renders at compression level 2.
   5.27. The mixed rollout overlay renders at compression level 3.
   5.28. The mixed rollout overlay renders at compression level 4.
   5.29. A canonical Figure 23 companion golden SVG is committed at compression level 4.
   5.30. A canonical Figure 24 companion golden SVG is committed at compression level 4.
   5.31. A canonical Figure 27 companion golden SVG is committed at compression level 4.
   5.32. Each overlay golden is registered in the existing parameterized golden test.
   5.33. `CFFG_UPDATE_GOLDENS=1` is the sole path that rewrites overlay goldens.
   5.34. A normal golden-test run compares overlay SVGs without rewriting them.
   5.35. Existing HMM-slice level-0 golden output remains byte-identical.
   5.36. Existing HMM-slice level-4 golden output remains byte-identical.
   5.37. Existing T-maze level-4 base golden output remains byte-identical.
   5.38. A 256-step HMM overlay derives a finite event tuple.
   5.39. The 256-step HMM event count grows linearly with its scheduled factor-role writes.
   5.40. Full-overlay coalescing produces at most two arrow primitives per structural wire.
   5.41. Repeated rendering of one overlay produces byte-identical SVG.

6. Documentation, hostile-input scope, and layer boundaries are explicit.
   6.1. `docs/ARCHITECTURE.md` replaces the statement that schedule overlays are unimplemented.
   6.2. The Diagram rendering section documents the Template-plus-Engine overlay input.
   6.3. The same section documents ordered schedule events.
   6.4. The same section documents full-view coalescing.
   6.5. The same section documents step selection.
   6.6. The same section documents compiled GFE classification.
   6.7. `README.md` adds one schedule-overlay example after the base rendering example.
   6.8. The README example renders a full overlay.
   6.9. The README example renders one selected logical step.
   6.10. The README explains that an overlay is a schedule plan rather than a runtime trace.
   6.11. The README identifies red arrows as GFE-based messages.
   6.12. Sparse probability support is structurally irrelevant because overlay derivation reads no probabilities.
   6.13. Exact log-zeros are structurally irrelevant because overlay derivation reads no message values.
   6.14. An improper message is structurally irrelevant because overlay derivation reads no message values.
   6.15. A vacuous message is structurally irrelevant because overlay derivation reads no message values.
   6.16. Float32 is structurally irrelevant because overlay derivation performs no JAX numeric work.
   6.17. The 256-step fixture supplies uncomfortable schedule-size evidence.
   6.18. This ticket makes no serialized IR change.
   6.19. This ticket makes no schema-version change.
   6.20. This ticket makes no compiler-arena layout change.
   6.21. This ticket makes no engine execution change.
   6.22. This ticket makes no convergence-status change.
   6.23. This ticket makes no free-energy-term change.
   6.24. This ticket makes no bundle change.
   6.25. This ticket makes no checkpoint change.
   6.26. This ticket makes no schedule serialization change.
   6.27. This ticket creates no runtime-state trace recorder.

7. Validation and handoff provide one decision per artifact.
   7.1. Focused overlay-derivation tests pass after the final code change.
   7.2. Focused scene tests pass after the final code change.
   7.3. Focused SVG tests pass after the final code change.
   7.4. Focused schedule tests pass after the final code change.
   7.5. Focused rollout rendering tests pass after the final code change.
   7.6. Focused T-maze rendering tests pass after the final code change.
   7.7. `tools/test-suite-runtimes.json` supplies the wait interval for final validation.
   7.8. The required short suite passes once through `uv run python tools/dashboard.py --run-suite` after the final code change.
   7.9. The final short-suite run refreshes `~/agent-share/cffg-tests.html`.
   7.10. `tk dep cycle` reports no dependency cycle before handoff.
   7.11. `git diff --check` reports no whitespace error before handoff.
   7.12. One atomic implementation commit contains `cff-jude` in its message.
   7.13. The ticket handoff note names the implementation commit.
   7.14. The ticket handoff note records the overlay-derivation-test result.
   7.15. The ticket handoff note records the scene-test result.
   7.16. The ticket handoff note records the SVG-test result.
   7.17. The ticket handoff note records the schedule-test result.
   7.18. The ticket handoff note records the rollout-rendering-test result.
   7.19. The ticket handoff note records the T-maze-rendering-test result.
   7.20. The ticket handoff note records the short-suite result.
   7.21. The ticket handoff note records the dashboard refresh.
   7.22. The ticket handoff note records the golden-refresh command.

## Methodological Suggestions

- Add a focused `cffg.viz.overlay` module rather than teaching `diagram.derive` about compiled execution. Keep semantic model notation independent from an optional schedule presentation.
- Accept a validated `Engine` so the overlay uses its sealed `Arena` and `Schedule`. Validate the Template mapping through factor names, group roles, scatter slots, and `Arena.edges`; do not compile a second arena or compare names alone.
- Define frozen `MessageEvent` and `ScheduleOverlay` values. Keep phase, logical-step, batch, arena, source, target, role, wire, and GFE identity typed so SVG rendering does not parse tooltip strings.
- Centralize stage expansion in a pure iterator. Expand `ScanStage` row tables logically without changing the compiled scan or engine dispatch.
- Build the implicit iterative event phase from the same arena scatter tables consumed by `_make_step`. Keep its writes simultaneous because the engine reads the pre-phase arrays for synchronous structures.
- Map a factor-owned scatter slot back to `EdgeInfo`, then confirm the factor-role-to-variable relationship against the Template. This makes direction a compiled ownership fact rather than an SVG-coordinate guess.
- Classify red messages from membership in `gfe_groups`, `gfe_tm_groups`, and `hybrid_gfe_groups`. Do not color every message in a mixed schedule red.
- Add `MessageArrow` to `scene.py`. Place arrowheads by arclength on a straight segment of the referenced `Path`, using the semantic source endpoint to orient traversal. Reserve separate fractions for opposing directions and avoid the midpoint bead.
- Keep exact events in `ScheduleOverlay`; coalesce only at scene construction. `select_step` should remain a pure filter so GUI and test consumers can inspect execution order without reparsing SVG.
- Extend `build_scene`, `render_svg`, `save_svg`, and `save_png` with one optional overlay argument. Preserve their output exactly when it is omitted.
- Reuse the existing SVG golden update and `viz_gallery` mechanisms. Do not introduce animation, browser scripting, or a second image-comparison system.
- The change-path index has no row for a read-only visualization of derived execution structure. Host/JIT, flooding/scheduled, single-instance/`vmap`, and current/legacy-checkpoint execution pairs are N/A because the overlay neither executes nor persists a schedule.

## Future work

- Runtime animation, elapsed-time traces, residual/FE plots, and per-iteration message values require a trace recorder and are outside this ticket.
- Interactive stepping controls can consume `ScheduleOverlay.select_step` in a later GUI ticket.
- Wire-to-wire routing and label placement remain owned by their visualization tickets; this overlay consumes final `Path` geometry.
- User-defined arrow colors, themes, and export animation are outside the paper-faithful default style.
- Comparing two schedules on one diagram is outside this single-overlay contract.

## Definitions

- **Schedule plan**: the deterministic structural order in which an Engine can update messages; it contains no observed runtime history.
- **Overlay**: typed message-direction metadata plus arrow primitives drawn over an unchanged base diagram.
- **Message event**: one compiled factor-role write to one message slot during one logical schedule step.
- **Logical step**: one ordered schedule position after expanding batch and scan structure without executing it.
- **Simultaneous batch**: factor rows whose writes use the same pre-batch inputs and share one ordering position.
- **Prime phase**: stages executed once before repeated colored or mixed iterations.
- **Repeatable phase**: the schedule-plan unit executed on each applicable engine iteration.
- **Implicit iterative phase**: synchronous VMP, GFE, constraint, or hybrid writes executed by the engine outside `Schedule.stages`.
- **Island refresh**: the scheduled forest or colored BP stages following a mixed iterative phase.
- **Flooding sweep**: one synchronous update of the factor groups selected by flooding.
- **Scatter slot**: the compiled message-arena address written by one factor role.
- **Source endpoint**: the factor role that owns the scatter slot for a message event.
- **Target endpoint**: the other factor or external terminal on the same variable edge.
- **GFE-based message**: a message written by a member of a compiled discrete, transition, or hybrid GFE plan.
- **Coalesced direction view**: one visible arrow for repeated events sharing wire, source, target, and GFE identity.
- **Step-selected view**: an overlay filtered to one logical step before scene construction.
- **Message arrow**: a typed scene primitive positioned and oriented along one existing wire path.
- **Arena family**: the discrete, Gaussian, or exponential-family message storage addressed by an event.
- **Schedule reason**: the deterministic explanation stored on a colored, mixed, or flooding Schedule.
- **Golden SVG**: a committed exact SVG string used as a deterministic regression artifact.

## Notes

### 2026-08-07 — Rewrite audit

Preserved historical handoff: cff-t6ec supplied the scene/SVG layer, and cff-d56d supplied the schedule and arena inputs needed for schedule-to-wire mapping. The original paper-figure target remains Figs. 23, 24, and 27, with GFE messages in red.

Capability owners: `cffg.viz.overlay` owns structural event derivation; `cffg.viz.scene` owns arrow primitives and geometry; `cffg.viz.svg` owns serialization and exported rendering arguments; `Schedule` and `Arena` remain the authorities for compiled execution structure; Template factor roles and scene `Path.ref` values own semantic wire identity. Runtime tracing, GUI stepping, routing, and label placement retain the owners named under Future work.

Sources checked: Part I text surrounding Figs. 23, 24, and 27; closed prerequisites cff-t6ec and cff-d56d; `src/cffg/schedule.py`; `src/cffg/compiler.py`; `src/cffg/engine.py`; `src/cffg/viz/diagram.py`; `src/cffg/viz/layout.py`; `src/cffg/viz/scene.py`; `src/cffg/viz/svg.py`; `src/cffg/models/rollout.py`; `tests/test_schedule.py`; `README.md`; and the Diagram rendering change-path row in `docs/ARCHITECTURE.md`.

Historical slice: 2 candidates, 2 occurrences, 0 problems. `land -> component_landed`; `pass -> data_flow`.

Final current-scan candidate map (candidate -> selected meaning):

- `canonical -> project_authority`.
- `n_a -> structurally_irrelevant`.
- `preserve -> semantic_invariant | byte_or_value_identity`, selected by clause.
- `register -> metadata_declaration`.
- `reusable -> artifact_reuse | code_reuse`, selected by clause.
- `same -> object_identity | value_equality | shared_category`, selected by clause.
- `expose -> public_api`.
- `implement -> planned_work`.
- `provide -> ticket_promise`.
- `support -> probability_support`.
- `work -> produces_correct_result`.
- `complete -> execution_finished | ticket_scope_complete`, selected by clause.
- `remaining -> preserved_state | unfinished_required_work`, selected by clause.
- `close -> work_item_completion`.
- `check -> automated_check | manual_inspection`, selected by clause.
- `converge -> runtime_status_criterion`.
- `evidence -> automated_test_result | empirical_support`, selected by clause.
- `exact -> bitwise_or_value_identity | literal_zero_or_support | identifier_precision`, selected by clause.
- `match -> identity_alignment`.
- `pass -> data_flow | algorithmic_pass | tests_succeed`, selected by clause.
- `test -> test_artifact | automated_execution`, selected by clause.
- `validate -> input_validation | acceptance_validation | component_validation`, selected by clause.
- `infer -> metadata_derivation`.
- `scale -> domain_size`.
- `can -> demonstrated_capability | theoretical_capability`, selected by clause.
- `should -> design_intent`.
- `explicit_owner -> named_ticket_owner | component_or_actor_owner`, selected by clause.
- `every -> named_class_distributive | enumerated_set_distributive`, selected by clause.
- `only -> hard_exclusion | ticket_scope_boundary`, selected by clause.
- `current -> repository_at_reference | latest_evidence`, selected by clause.
- `existing -> predates_change`.
- `later -> future_scope`.
- `read -> machine_access`.
- `land -> component_landed`.

Current-scan totals: 34 candidates, 164 occurrences, 0 problems.

Intelligible-ticket review: complete.
