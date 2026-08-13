---
id: cff-9bex
status: open
deps: []
links: [cff-bxk4]
created: 2026-07-09T02:46:53Z
type: bug
priority: 2
assignee: coopwilliams
---
# viz: wire routing at high-degree factors — no interior crossings, spread to free sides

## Background in Plain English

The rectilinear layout assigns each factor role to a box side and then turns those assignments into SVG paths. In commit `a59bbfa`, `_assign_sides` already processes structural wires before ports and blob-backed parameters, but `_route` chooses its bend from the first endpoint's side alone. A structural path can therefore reach its second endpoint from a direction that disagrees with the assigned side, cross that factor's open interior, or pass through another factor box.

Cooper's 2026-07-09 design decision makes terminal and parameter wires elastic. Structural factor-to-factor wires keep the straightest topology-driven sides; ports and parameters consume unused sides before sharing a side; a shared side receives deterministically ordered, evenly spaced anchors; and external wire paths stay outside box interiors. Internal bead and dome leaders remain allowed inside their owning box because they depict the node marginal rather than an external wire route.

The concrete five-connection case is `gaussian_linear`, with two structural roles and three parameter roles. `gaussian_mixture_linear` supplies a six-connection case with three structural roles and three parameter roles. This ticket provides the routed sides, boundary anchors, and exterior path geometry consumed by cff-bxk4's Gaussian captions/terminals and cff-x5lv's terminal-label placement; it does not implement either consumer's notation or label policy.

## Outcomes / Acceptance Criteria

1. Side assignment preserves structural priority and uses free sides for elastic endpoints.
   1.1. `_assign_sides` classifies a wire with two factor endpoints as structural.
   1.2. `_assign_sides` classifies a one-factor wire ending in POINTMASS as a blob-backed parameter wire.
   1.3. `_assign_sides` classifies every other one-factor wire as a port wire.
   1.4. Structural wires claim sides before port wires.
   1.5. Port wires claim sides before blob-backed parameter wires.
   1.6. A horizontally dominant structural displacement assigns horizontal endpoint sides.
   1.7. A vertically dominant structural displacement assigns vertical endpoint sides.
   1.8. Equal horizontal and vertical displacement retains the pre-ticket horizontal tie break.
   1.9. The two sides assigned to one structural wire face one another along the selected axis.
   1.10. A rail GLUE port continues away from its rail endpoint when that continuation side is unused.
   1.11. A port attached above the rail prefers an unused upper side.
   1.12. A port attached below the rail prefers an unused lower side.
   1.13. A blob-backed parameter uses the declared `left, up, down, right` preference among unused sides.
   1.14. No elastic endpoint receives an occupied side while its factor has an unused side.
   1.15. An elastic endpoint shares a side after all four sides have been claimed.
   1.16. Adding an elastic endpoint does not change a structural role's assigned side.
   1.17. Repeated side assignment for identical diagram topology returns value-equal role-to-side mappings.

2. Boundary anchors make crowded sides deterministic and reviewable.
   2.1. Every factor role receives one anchor.
   2.2. A left-side anchor lies on the factor's left boundary.
   2.3. A right-side anchor lies on the factor's right boundary.
   2.4. An upper-side anchor lies on the factor's upper boundary.
   2.5. A lower-side anchor lies on the factor's lower boundary.
   2.6. A side with `n` assigned roles places role `i` at fraction `(i + 1) / (n + 1)` of that side.
   2.7. Structural endpoints on one side are ordered by the far factor's orthogonal coordinate.
   2.8. Terminal endpoints tied on the far-coordinate key are ordered by `natural_key(wire.id)`.
   2.9. Distinct roles sharing one side receive distinct anchor coordinates.
   2.10. Repeated anchor calculation from identical inputs returns value-equal coordinates.
   2.11. Internal bead leaders terminate at the anchor for their role.
   2.12. Internal dome leaders terminate at the anchors for their roles.

3. Every external wire route respects both endpoint sides and every factor obstacle.
   3.1. Every external wire path consists of horizontal or vertical segments.
   3.2. The first nonzero segment of a structural path leaves its first anchor in that anchor side's outward direction.
   3.3. The final nonzero segment of a structural path reaches its second anchor from that anchor side's outward half-plane.
   3.4. The open portion of a structural path is disjoint from its first factor's open box interior.
   3.5. The open portion of a structural path is disjoint from its second factor's open box interior.
   3.6. A structural path is disjoint from every non-incident factor's open box interior.
   3.7. A terminal path leaves its factor anchor in the assigned side's outward direction.
   3.8. The open portion of a terminal path is disjoint from its factor's open box interior.
   3.9. A terminal path is disjoint from every non-incident factor's open box interior.
   3.10. Touching a factor boundary at the assigned incident anchor is allowed by the intersection oracle.
   3.11. Crossing a factor boundary away from the assigned incident anchor fails the intersection oracle.
   3.12. A `Path(kind="internal")` is excluded from the external-wire intersection prohibition.
   3.13. A zero-length segment is removed before a path enters the typed scene.
   3.14. Route selection is deterministic when two collision-free candidates have equal Manhattan length.
   3.15. Route selection prefers the collision-free candidate with lower Manhattan length.
   3.16. A rendered rounded SVG wire is disjoint from every factor's open box interior except at its assigned incident anchors.
   3.17. Every emitted wire keeps its pre-ticket `var:` or `blob:` scene reference.

4. Five-connection, six-connection, obstacle, compression, and stability fixtures pin the behavior.
   4.1. A minimal `gaussian_linear` fixture contains exactly five factor connections.
   4.2. The five-connection fixture includes two structural wires.
   4.3. The five-connection fixture includes three blob-backed parameter wires.
   4.4. A minimal `gaussian_mixture_linear` fixture contains exactly six factor connections.
   4.5. The six-connection fixture includes three structural wires.
   4.6. The six-connection fixture includes three blob-backed parameter wires.
   4.7. The five-connection fixture satisfies criteria 1 through 3 at compression level 0.
   4.8. The five-connection fixture satisfies criteria 1 through 3 at compression level 1.
   4.9. The five-connection fixture satisfies criteria 1 through 3 at compression level 2.
   4.10. The five-connection fixture satisfies criteria 1 through 3 at compression level 3.
   4.11. The five-connection fixture satisfies criteria 1 through 3 at compression level 4.
   4.12. The six-connection fixture satisfies criteria 1 through 3 at compression level 0.
   4.13. The six-connection fixture satisfies criteria 1 through 3 at compression level 1.
   4.14. The six-connection fixture satisfies criteria 1 through 3 at compression level 2.
   4.15. The six-connection fixture satisfies criteria 1 through 3 at compression level 3.
   4.16. The six-connection fixture satisfies criteria 1 through 3 at compression level 4.
   4.17. A three-factor obstacle fixture exercises a route whose direct Manhattan candidates meet the middle box.
   4.18. The obstacle fixture produces a path disjoint from the middle box's open interior.
   4.19. A factor with 32 endpoint roles produces finite anchor coordinates.
   4.20. A factor with 32 endpoint roles produces pairwise-distinct same-side anchors.
   4.21. A factor with 32 endpoint roles satisfies the external-wire intersection oracle.
   4.22. A golden SVG for the five-connection fixture is committed at compression level 0.
   4.23. A golden SVG for the five-connection fixture is committed at compression level 4.
   4.24. A golden SVG for the six-connection fixture is committed at compression level 0.
   4.25. A golden SVG for the six-connection fixture is committed at compression level 4.
   4.26. Each new golden is registered in the existing parameterized golden test.
   4.27. `CFFG_UPDATE_GOLDENS=1` is the sole path that rewrites the four new goldens.
   4.28. A normal golden-test run compares the four new SVGs without rewriting them.
   4.29. Existing HMM-slice level-0 golden output remains byte-identical.
   4.30. Existing HMM-slice level-4 golden output remains byte-identical.
   4.31. Existing T-maze level-4 golden output remains byte-identical.
   4.32. Growing the existing rollout from two to three steps leaves the pre-existing factor coordinates value-equal.
   4.33. Repeated rendering of one high-degree fixture produces byte-identical SVG.

5. Documentation, downstream boundaries, and hostile-input scope are explicit.
   5.1. `docs/ARCHITECTURE.md`'s Diagram rendering section records structural-first side assignment.
   5.2. The same section records free-side fan-out for elastic endpoints.
   5.3. The same section records evenly spaced shared-side anchors.
   5.4. The same section records the no-open-box-interior-crossing invariant.
   5.5. `README.md`'s visualization guidance states that exported diagrams use deterministic exterior routing.
   5.6. cff-9bex owns side assignment.
   5.7. cff-9bex owns anchor placement.
   5.8. cff-9bex owns external wire paths.
   5.9. cff-bxk4 consumes the routed path when placing Gaussian domain captions.
   5.10. cff-x5lv consumes the routed path when placing terminal-end labels.
   5.11. This ticket does not define Gaussian domain notation.
   5.12. This ticket does not define continuous-terminal glyphs.
   5.13. This ticket does not define parameter-label text.
   5.14. This ticket does not define label collision bounds.
   5.15. Sparse probability support is structurally irrelevant because routing reads no probability values.
   5.16. Exact log-zeros are structurally irrelevant because routing reads no probability values.
   5.17. An improper message is structurally irrelevant because routing reads no messages.
   5.18. A vacuous message is structurally irrelevant because routing reads no messages.
   5.19. Float32 is structurally irrelevant because routing performs no JAX numeric work.
   5.20. Variable cardinality does not affect routing geometry for fixed topology.
   5.21. Gaussian dimension does not affect routing geometry for fixed topology.
   5.22. The 32-endpoint fixture supplies uncomfortable routing-scale evidence.
   5.23. This ticket makes no serialized IR change.
   5.24. This ticket makes no schema-version change.
   5.25. This ticket makes no compiler-arena change.
   5.26. This ticket makes no message-update-rule change.
   5.27. This ticket makes no free-energy-term change.
   5.28. This ticket makes no runtime-status change.
   5.29. This ticket makes no bundle change.
   5.30. This ticket makes no checkpoint change.

6. Validation and handoff provide one decision per artifact.
   6.1. Focused layout tests pass after the final code change.
   6.2. Focused SVG tests pass after the final code change.
   6.3. Focused Kalman rendering tests pass after the final code change.
   6.4. Focused hybrid rendering tests pass after the final code change.
   6.5. `tools/test-suite-runtimes.json` supplies the wait interval for final validation.
   6.6. The required short suite passes once through `uv run python tools/dashboard.py --run-suite` after the final code change.
   6.7. The final short-suite run refreshes `~/agent-share/cffg-tests.html`.
   6.8. `tk dep cycle` reports no dependency cycle before handoff.
   6.9. `git diff --check` reports no whitespace error before handoff.
   6.10. One atomic implementation commit contains `cff-9bex` in its message.
   6.11. The ticket handoff note names the implementation commit.
   6.12. The ticket handoff note records the focused layout-test result.
   6.13. The ticket handoff note records the focused SVG-test result.
   6.14. The ticket handoff note records the focused Kalman-rendering-test result.
   6.15. The ticket handoff note records the focused hybrid-rendering-test result.
   6.16. The ticket handoff note records the short-suite result.
   6.17. The ticket handoff note records the dashboard refresh.
   6.18. The ticket handoff note records the golden-refresh command.

## Methodological Suggestions

- Start from the Diagram rendering section of `docs/ARCHITECTURE.md`. `src/cffg/viz/layout.py:_assign_sides` owns role-to-side decisions, `_anchor_points` owns boundary coordinates, `_wire_geometry` owns endpoint handling, and `_route` owns structural polylines. `scene.Path` transports the result and `svg._rounded_path` renders it. The change-path index has no compiler/runtime row for this work; host/JIT, flooding/scheduled, single-instance/`vmap`, and current/legacy-checkpoint review pairs are N/A.
- Preserve the structural-first classification and side preferences in `_assign_sides`. Treat the side map as input to routing rather than letting routing silently reinterpret an anchor's side.
- Give a structural router both endpoint sides. One maintainable approach is to create an outward stub from each anchor, connect the two stub endpoints with rectilinear candidates, reject candidates that meet an open box interior, and choose among survivors by `(Manhattan length, bend count, natural coordinate key)`.
- Derive stub and detour clearance from `LayoutStyle`, box extent, and rendered stroke/bead extents. Do not tune an absolute distance to the default 36- or 58-pixel box.
- When both direct elbow candidates meet an obstacle, enumerate deterministic perimeter channels outside the obstacle rectangles. Keep the router topology-based and bounded; this ticket does not call for a global graph-drawing optimizer.
- Implement one reusable segment-versus-open-rectangle predicate. Treat boundary contact as non-intersection, then add the stricter incident-anchor check separately so a path cannot exit through the wrong point.
- Apply the geometric oracle to `Path(kind="wire")`. Do not reject `Path(kind="internal")`, beads, or domes for entering their owning box.
- Exercise `_assign_sides` and `_anchor_points` directly for priority/order decisions. Exercise `build_scene` for the end-to-end intersection, compression, reference, and stability decisions.
- Build the five-connection fixture from the registered `gaussian_linear` role signature. Build the six-connection fixture from `gaussian_mixture_linear`; keep both fixtures minimal so their goldens isolate routing.
- Reuse the existing golden update mechanism and `viz_gallery` evidence path. Do not introduce another image-comparison harness.
- The numerical hostile-input contract is N/A for probability support, messages, and dtype. The degree-32 fixture is the applicable scale pressure for this combinatorial geometry rule.

## Future work

- cff-bxk4 owns Gaussian edge-domain captions and continuous DATA/OPEN terminal marks. Do not duplicate that notation here.
- cff-x5lv owns terminal-end parameter labels, text extents, overhang staggering, and label-collision checks. Do not move those policies into the router.
- Preventing external wire-to-wire crossings is outside this ticket. A future planarization ticket can own that broader graph-drawing problem.
- Edge bundling, spline routing, and interactive drag constraints are outside this deterministic monochrome SVG contract.
- A global optimizer is outside this ticket unless a later fixture proves the bounded rectilinear candidate set cannot find an available route.

## Definitions

- **High-degree factor**: a factor box with five or more structural, port, or parameter wire endpoints.
- **Structural wire**: a wire joining two factor roles.
- **Port wire**: a one-factor wire ending at a DATA, PRIOR, GOAL, or GLUE terminal.
- **Blob-backed parameter wire**: a one-factor wire ending at a POINTMASS terminal for a named blob.
- **Elastic endpoint**: a port or blob-backed parameter endpoint whose exterior route may lengthen or bend to use an available side.
- **Claimed side**: the left, right, upper, or lower box boundary assigned to one factor role.
- **Unused side**: a box side not yet claimed during the ordered assignment pass.
- **Anchor**: the exact boundary coordinate at which one role's wire meets its factor box.
- **Outward direction**: the half-line from an anchor away from the owning box, perpendicular to the claimed side.
- **Open box interior**: the rectangle strictly inside a factor boundary; boundary points are excluded.
- **Incident factor**: a factor named by one of a wire's `NodeEnd` values.
- **Non-incident factor**: a factor with no endpoint on the wire being checked.
- **Rectilinear path**: a polyline whose nonzero segments are horizontal or vertical.
- **Outward stub**: the first exterior segment from an anchor before a route turns.
- **Obstacle**: a factor's open box interior, which an external wire route must not enter.
- **Internal leader**: a `Path(kind="internal")` joining a factor boundary to an internal bead or dome.
- **Intersection oracle**: the automated segment/rectangle predicate that decides the external routing invariant.
- **Compression level**: one of presentation levels 0 through 4, which changes box dimensions and variational detail.
- **Golden SVG**: a committed exact SVG string used as a deterministic regression artifact.
- **Layout stability**: value equality of named scene geometry before and after an unrelated graph extension.

## Notes

**2026-08-08T05:18:55Z**

Ticket rewrite audit, 2026-08-07.

The bounded historical slice reported 9 candidates, 11 occurrences, and 0 problems. Historical mapping: same -> shared_category; stable -> unchanged_state; test -> test_artifact; may -> possibility; all -> enumerated_subset_exhaustive; any -> universal; every -> named_class_distributive; only -> temporal_gate; existing -> unanchored_presence.

Sources checked: cff-9bex metadata and dependency tree; commit a59bbfa; FactorType signatures for gaussian_linear and gaussian_mixture_linear; layout side-assignment, anchor, route, scene-framing, and SVG corner code; layout, SVG, Kalman, hybrid, and gallery fixtures; the Diagram rendering architecture section; and the retained cff-bxk4/cff-x5lv routing-consumer requirements.

Final current-scan totals after this note: 38 candidates and 169 occurrences.

Complete candidate mapping: end_to_end -> component_chain; fixed -> pinned_artifact; n_a -> structurally_irrelevant; preserve -> semantic_invariant; register -> metadata_declaration; reusable -> code_reuse or evidence_reuse; same -> object_identity or shared_category; available -> theoretically_obtainable; implement -> planned_work or primitive_exists; provide -> ticket_promise or evidence_supply; support -> probability_support; complete -> ticket_scope_complete; deliver -> primitive_delivered; land -> repository_change_present; remaining -> preserved_state; close -> work_item_completion; check -> automated_check; evidence -> automated_test_result or empirical_support; exact -> identifier_precision, literal_zero_or_support, mathematical_no_approximation, or bitwise_or_value_identity; oracle -> test_artifact; pass -> tests_succeed or non_test_check_succeeds; robust -> hostile_numeric_survival; stable -> unchanged_state; test -> test_artifact or automated_execution; validate -> acceptance_validation; scale -> domain_size; can -> possibility or theoretical_capability; may -> permission or possibility; explicit_owner -> current_ticket_owner, named_ticket_owner, or component_or_actor_owner; named_ticket_handoff -> concrete_current_owner; all -> enumerated_subset_exhaustive; any -> universal; every -> named_class_distributive; only -> temporal_gate; current -> repository_at_reference; existing -> predates_change or unanchored_presence; later -> future_scope; read -> machine_access.

Each retained occurrence is bounded by a named side, endpoint class, role, factor signature, coordinate, rectangle predicate, compression level, fixture, golden, file, command, commit, ticket, or scope exclusion. Capability owners: cff-9bex owns side assignment, anchors, and external paths; cff-bxk4 owns Gaussian notation and continuous terminal marks; cff-x5lv owns parameter-label text, extents, staggering, and label-collision bounds; core IR owns model structure; SVG owns final path serialization.

Intelligible-ticket review: complete.
