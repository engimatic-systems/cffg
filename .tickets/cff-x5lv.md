---
id: cff-x5lv
status: open
deps: [cff-9bex]
links: [cff-bxk4]
created: 2026-07-09T02:46:53Z
type: feature
priority: 2
assignee: coopwilliams
---
# viz: label placement — terminal-end parameter labels + staggered overhang labels

## Background in Plain English

At ticket-creation commit `a4b0dab`, high-degree factors place unrelated text in the same space. The LGSSM gallery shows parameter label `zn` over variable label `x0c`, parameter label `zm` over `x2`, and data-terminal caption `y0` over structural wire label `y_0`. At that repository reference, the semantic diagram gives a parameter wire both its blob name and a point-mass terminal carrying that name; `layout._emit` renders the wire copy and clears the terminal copy.

Cooper selected two layout rules on 2026-07-09. First, a parameter blob name belongs beside its point-mass terminal, following the `d`, `A`, and `c` convention in Part I Figure 22, while a structural variable name keeps its mid-wire position and receives placement priority. Second, terminal wires sharing one factor side have successively longer endpoints; each additional length provides an exclusive overhang zone for that wire's text.

Ticket cff-9bex is the prerequisite because it assigns structural wires first, fans elastic parameter wires onto free sides, and prevents routes from crossing factor interiors. This ticket starts from those routed sides and owns label position, label extent, terminal-wire staggering, and collision evidence. It does not own Gaussian dimension notation or distinct continuous-terminal glyphs from cff-bxk4.

## Outcomes / Acceptance Criteria

1. Each rendered label has one semantic owner.
   1.1. A point-mass parameter wire produces a scene `Path` with an empty `label` field.
   1.2. The point-mass `Terminal` for that wire carries the blob name as its caption.
   1.3. The SVG output contains the blob name in exactly one `<text>` element for that parameter wire.
   1.4. A structural variable wire retains its variable name in the scene `Path.label` field.
   1.5. The SVG output contains the structural variable name in exactly one `<text>` element for that wire.
   1.6. A data or open terminal retains its port caption in the scene `Terminal.label` field.
   1.7. The `factor:`, `var:`, `blob:`, and `port:` refs of affected scene primitives remain unchanged.
2. Same-side terminal wires use deterministic exclusive overhang zones.
   2.1. Wires are grouped for staggering by factor ref and routed side after cff-9bex side assignment completes.
   2.2. The order within one stagger group is a deterministic function of structural priority, attachment position, and natural wire key.
   2.3. A structural wire keeps priority over terminal wires when their label space conflicts.
   2.4. The layout computes each caption width from the same deterministic text-extent rule used by the collision oracle.
   2.5. Each terminal wire after the first extends beyond its predecessor by at least its own caption width plus the configured label clearance.
   2.6. Each terminal caption is centered in the extension between its predecessor's endpoint and its own endpoint.
   2.7. Each terminal caption's ownership metadata names its own wire or terminal ref.
   2.8. Staggering changes no structural factor-to-factor route point.
   2.9. A variable wire label and the caption on that wire's terminal occupy disjoint label boxes.
3. A deterministic collision oracle covers representative and hostile layouts.
   3.1. The typed scene exposes the position, anchor, font role, and conservative bounding box for every rendered path or terminal caption.
   3.2. Positive-area intersection between two exposed label boxes is the collision predicate.
   3.3. The HMM-slice fixture has zero label collisions at compression levels 0, 1, 2, 3, and 4.
   3.4. The T-maze fixture from `build_tmaze` has zero label collisions at compression levels 0, 1, 2, 3, and 4.
   3.5. The three-step LGSSM fixture used by `test_lgssm_renders_at_every_level` has zero label collisions at compression levels 0, 1, 2, 3, and 4.
   3.6. The five-degree factor fixture delivered by cff-9bex has zero label collisions at compression levels 0, 1, 2, 3, and 4.
   3.7. The six-degree factor fixture delivered by cff-9bex has zero label collisions at compression levels 0, 1, 2, 3, and 4.
   3.8. A same-side terminal group containing a 32-character blob name has zero label collisions.
   3.9. Every exposed label box lies inside the final scene canvas.
4. Label geometry preserves deterministic layout and ticket boundaries.
   4.1. Repeated `build_scene` calls for one template, level, and style return equal scenes.
   4.2. Growing a rollout preserves the factor-box coordinates already checked by `TestStability.test_growing_the_rollout_moves_nothing`.
   4.3. Compression still changes presentation directives without changing semantic diagram labels.
   4.4. The label change introduces no Gaussian dimension annotation.
   4.5. The label change introduces no new continuous-terminal glyph.
   4.6. The label change introduces no semantic IR, serialization, compiler, runtime, or checkpoint field.
5. Automated and visual evidence make the geometry reviewable.
   5.1. The golden-regeneration command runs once after the final geometry change.
   5.2. The regenerated artifact set includes the existing HMM-slice and T-maze golden SVGs.
   5.3. The regenerated artifact set includes the five-degree and six-degree golden SVGs from cff-9bex.
   5.4. A normal test run after regeneration reproduces every affected golden byte for byte.
   5.5. Dashboard gallery artifacts show the HMM slice, T-maze, LGSSM, five-degree, six-degree, and long-label fixtures at compression levels 0 and 4.
   5.6. Focused diagram, compression, layout, SVG, and Kalman visualization tests pass.
   5.7. The dashboard-backed default short suite passes once after the final code change.
   5.8. `docs/ARCHITECTURE.md` is updated in place if the typed scene or deterministic layout contract changes.
   5.9. `README.md` is updated in place if consumers receive a new public scene-label field or primitive.

## Methodological Suggestions

These methods are suggestions. An implementation may choose a different design if it satisfies each numbered outcome with equivalent evidence.

Land cff-9bex first and treat its routed sides and anchor points as input. In `src/cffg/viz/layout.py:_emit`, detect a `POINTMASS` terminal before assigning `Path.label_at`: leave the path text empty and carry `TerminalEnd.label` to the terminal caption. Preserve structural `WireView.label` text on the path. Do not duplicate Gaussian notation work from cff-bxk4.

Make the typed scene, rather than SVG parsing, authoritative for label geometry. One maintainable option is a typed label primitive or shared label-geometry record containing text, owner ref, anchor, position, font role, and conservative bounds. Both `layout._framed` and `svg.to_svg` can consume that record, ensuring the collision test measures the positions that the renderer uses. Keep the text-extent calculation deterministic and conservative for the supported serif math style, numeric subscripts produced by `_mathify`, `SvgStyle.label_size`, and `SvgStyle.tiny_size`; do not depend on whatever fonts happen to be installed on the test machine.

Compute stagger groups after `_assign_sides` and `_anchor_points`. Collect structural label bounds first, then assign terminal offsets in deterministic order. For each later terminal, add its caption width and a named clearance to the previous terminal length, and place the caption in that new interval. Extend elastic terminal paths before considering any structural-route movement. Include terminal captions such as `y0` in collision allocation so they cannot cover the variable label `y_0` on the same path.

The change-path index in `docs/ARCHITECTURE.md` has no visualization-specific row. Treat `src/cffg/viz/scene.py` as the label-geometry contract, `src/cffg/viz/layout.py` as the position authority, and `src/cffg/viz/svg.py` as the serialization consumer. The host/JIT, flooding/scheduled, single-instance/`vmap`, and current/legacy checkpoint pairs are structurally irrelevant because visualization executes on the host from a `Template` and does not consume runtime state. The duplicated surfaces that do apply are semantic diagram versus scene, scene versus SVG, compression levels 0 through 4, and original versus extended rollout layout.

Place scene-level collision and ownership checks in `tests/test_viz_layout.py`, renderer and golden checks in `tests/test_viz_svg.py`, and LGSSM coverage beside `test_lgssm_renders_at_every_level`. Reuse the five-degree and six-degree fixtures introduced by cff-9bex rather than creating geometrically different substitutes. Regenerate affected goldens once after geometry is final, inspect the dashboard gallery, and then run the required default short suite once.

## Future work

- cff-9bex is the direct prerequisite and owns structural-first side selection, free-side fan-out, same-side anchor spreading, and the rule that a route cannot cross a factor interior.
- cff-bxk4 owns Gaussian edge-dimension labels, distinct prior and observed continuous-terminal glyphs, and LGSSM-specific golden SVG coverage.
- Interactive GUI font measurement is outside this ticket. A later GUI may replace the conservative scene bounds for display, but the deterministic headless bounds remain the golden and collision-test contract.

## Definitions

- **Caption:** Text placed next to a terminal mark, including a parameter blob name or port name.
- **Collision:** A positive-area intersection between two conservative label bounding boxes.
- **Compression level:** One of the integer presentation levels 0 through 4 implemented by `compress.present`.
- **Conservative bounding box:** A deterministic rectangle at least as large as the supported SVG text estimate for the selected font role.
- **Elastic terminal wire:** A factor-to-terminal path whose length may increase without moving factor boxes or structural routes.
- **Exclusive overhang zone:** The segment between one terminal endpoint and the preceding endpoint in the same stagger group.
- **Golden SVG:** A committed expected SVG file compared byte for byte with renderer output.
- **Label clearance:** The named minimum empty distance added around a caption when calculating a terminal extension.
- **Natural wire key:** The repository's numeric-aware ordering from `layout.natural_key`.
- **Parameter wire:** A path between a factor parameter role and a `POINTMASS` terminal representing a blob.
- **Point-mass terminal:** The filled square denoting a fixed model parameter in the paper notation.
- **Representative fixtures:** The HMM slice, T-maze, three-step LGSSM, and cff-9bex five-degree and six-degree diagrams named in criterion 3.
- **Stagger group:** Terminal wires attached to one factor side and ordered for successively longer endpoints.
- **Structural variable wire:** A variable path whose label identifies the model variable rather than a parameter blob.
- **Typed scene:** The backend-independent collection of positioned visualization primitives returned by `build_scene`.

## Notes

**2026-08-08T04:53:17Z**

Retained wiggle candidates after the bounded current scan (candidate -> selected sense): fixed -> a model parameter supplied by a blob; preserve -> ref, semantic-label, or factor-coordinate invariance; public -> documented build_scene consumer contract; reusable -> reuse of routed fixtures or scene geometry; same -> spatial identity, shared-side membership, or one text-extent rule as named; expose -> typed-scene geometry callable by consumers and tests; implement -> named repository function or planned code change; provide -> an exclusive geometric interval created by added wire length; support -> the explicitly supported headless SVG text style; complete -> side-assignment execution_finished; deliver -> cff-9bex fixture artifact supplied by its prerequisite work; land -> cff-9bex work-item completion before this implementation; remaining -> preserved scene refs or headless collision contract; check -> automated ownership, collision, golden, or focused test check as named; equivalent -> equal acceptance artifacts and evidence; estimate -> deterministic conservative text-width model; evidence -> named scene property, golden, dashboard artifact, or automated check; exact -> one SVG text-element count for a named owner; oracle -> automated scene-level collision calculation; pass -> automated tests_succeed; reproduce -> byte identity between a golden and a normal render; test -> automated test artifact or execution; can -> demonstrated consumer capability or theoretical implementation capability as named; expected -> committed comparison artifact in the Golden SVG definition; may -> implementation permission or future GUI option as named; explicit_owner -> cff-x5lv owns label geometry while named prerequisite and linked tickets own routing and Gaussian notation; named_ticket_handoff -> cff-9bex and cff-bxk4 are concrete owners; scope_exclusion -> Gaussian notation, GUI font measurement, semantic IR, compiler, runtime, persistence, and checkpoint changes are outside this ticket; any -> universal negative scope over structural-route movement or installed-font dependence; at_least -> inclusive terminal-extension or conservative-bound lower limit; every -> each member of the locally named label, golden, level, or test set; current -> the named current/legacy checkpoint review pair; existing -> HMM-slice and T-maze goldens that predate this change; later -> a subsequent terminal in deterministic order or a future interactive GUI. Each retained occurrence is bounded by its sentence, symbol, fixture, metric, commit, or named artifact.

Intelligible-ticket review: complete.
