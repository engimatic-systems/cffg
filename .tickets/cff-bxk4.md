---
id: cff-bxk4
status: open
deps: [cff-4e5z, cff-9bex]
links: [cff-9bex, cff-x5lv, cff-ahps]
created: 2026-07-08T16:50:10Z
type: task
priority: 3
assignee: coopwilliams
---
# viz: Gaussian/continuous domain rendering

## Background in Plain English

Commit `bae4d87` proves that Gaussian templates reach the SVG backend and gives Gaussian and linear-Gaussian factor boxes the short `𝒩` and `𝒩A` glyphs. At that commit their variable wires still look categorical: a reader cannot see the vector dimension, and a Gaussian observation or open prior boundary uses the same terminal mark as its categorical counterpart. This ticket carries the existing `VariableSpec.dim` fact through the semantic diagram, typed scene, layout, and SVG stages so continuous wires display their domain and continuous boundary terminals are visibly distinct.

Part I §5.2 and Figure 8 put `N` inside a bead to denote a Gaussian *form constraint*. They do not prescribe a numeric dimension caption. This ticket therefore keeps form constraints separate and adopts `ℝᵈ` as the repository's dimension caption, with the positive integer `d` taken from `VariableSpec.dim`; it does not infer a domain from a neighbouring factor or parameter array.

The closed prerequisite cff-4e5z supplies validated Gaussian variables and the LGSSM builder. The open prerequisite cff-9bex supplies routes that keep high-degree structural and parameter wires out of factor interiors. The linked cff-x5lv owns general label extent, terminal overhang staggering, and collision bounds; this ticket owns the Gaussian domain text and continuous-terminal semantics that those facilities place. This optional visualization work is linked from cff-ahps but is not a closure requirement.

## Outcomes / Acceptance Criteria

1. The semantic diagram has an explicit Gaussian-domain contract.
   1.1. For every variable whose `VariableSpec.family` is `"gaussian"`, `derive` records `"gaussian"` on its `WireView`.
   1.2. For every variable whose `VariableSpec.family` is `"gaussian"`, `derive` records the positive integer `VariableSpec.dim` on its `WireView`.
   1.3. A Gaussian `WireView` exposes its family without requiring endpoint-factor inspection.
   1.4. A Gaussian `WireView` exposes its dimension without requiring endpoint-factor inspection.
   1.5. The wire's variable-name label remains unchanged.
   1.6. The wire's `var:<name>` reference remains unchanged.
   1.7. Dimension metadata is a separate semantic property rather than text spliced into the variable name.
   1.8. Discrete, scalar exponential-family, and blob-backed pointmass wires do not acquire Gaussian-domain metadata.
   1.9. DATA, PRIOR, GOAL, and GLUE retain their existing `PortRole` values.
   1.10. DATA, PRIOR, GOAL, and GLUE retain their existing `TerminalKind` meanings.
   1.11. The wire domain does not create a model port role.
   1.12. Gaussian visualization does not change port validation.
   1.13. `VariableSpec.dim` is the sole dimension authority.
   1.14. Visualization does not inspect blob shapes to determine a wire domain.
   1.15. Visualization does not execute array code to determine a wire domain.
   1.16. Visualization does not infer a wire domain from a `gaussian*` factor name.

2. Every Gaussian variable wire displays one numeric real-vector domain caption.
   2.1. The visible domain caption has the form `ℝᵈ`.
   2.2. The superscript is the complete base-10 value of `d`.
   2.3. Dimension 1 renders as `ℝ¹`.
   2.4. Dimension 128 renders as `ℝ¹²⁸`.
   2.5. Each rendered Gaussian variable wire contains exactly one domain caption at compression levels 0, 1, 2, 3, and 4.
   2.6. The variable-name label remains visible exactly once on each Gaussian wire.
   2.7. The domain caption does not replace the variable-name label.
   2.8. The domain caption carries the wire's `var:<name>` scene reference.
   2.9. Each domain-caption extent lies outside every factor-box interior.
   2.10. Each domain-caption extent is disjoint from its wire stroke.
   2.11. Each domain-caption extent is disjoint from its terminal mark.
   2.12. Each domain-caption extent is contained by the SVG view box.
   2.13. Scene-level geometry checks decide criteria 2.9 through 2.12 without installed-font rasterization.
   2.14. Categorical, scalar exponential-family, and parameter wires contain no `ℝ` caption.
   2.15. Existing categorical SVG goldens remain byte-identical to the baseline committed by prerequisite cff-9bex.

3. Continuous boundary terminals are distinct while their boundary semantics remain recognizable.
   3.1. Gaussian domain identity reaches each typed DATA or OPEN `Terminal` scene item.
   3.2. SVG renderer logic does not derive terminal domain identity from caption text.
   3.3. SVG renderer logic does not derive terminal domain identity from serialized SVG text.
   3.4. A Gaussian DATA terminal keeps the filled observation bead.
   3.5. A Gaussian DATA terminal keeps the white `∂` mark.
   3.6. A Gaussian DATA terminal adds one concentric continuous-domain ring.
   3.7. Gaussian DATA geometry differs from categorical DATA geometry.
   3.8. A Gaussian DATA terminal keeps its `port:<name>` reference.
   3.9. A Gaussian DATA terminal retains observation-clamp semantics.
   3.10. A Gaussian PRIOR, GOAL, or defensive unported degree-one endpoint keeps the open bead.
   3.11. A Gaussian PRIOR, GOAL, or defensive unported degree-one endpoint adds one concentric continuous-domain ring.
   3.12. Gaussian OPEN geometry differs from categorical OPEN geometry.
   3.13. A Gaussian GLUE port remains an ellipsis.
   3.14. A POINTMASS parameter terminal remains a black square when attached to a Gaussian factor.
   3.15. A Gaussian prior factor remains a factor box.
   3.16. A Gaussian prior factor box retains the already-landed `𝒩` glyph.
   3.17. A Gaussian prior factor is not converted into an OPEN terminal.
   3.18. A `gaussian_linear` factor box retains `𝒩A`.
   3.19. The registered-glyph length check remains satisfied.

4. Compression and GFE presentation preserve their pre-ticket meanings.
   4.1. Compression continues to alter variational-bead presentation.
   4.2. Compression continues to alter node-marginal placement.
   4.3. Gaussian domain captions survive compression levels 0 through 4.
   4.4. Continuous DATA terminal variants survive compression levels 0 through 4.
   4.5. Continuous OPEN terminal variants survive compression levels 0 through 4.
   4.6. The existing T-maze fixture retains its set of `gfe:<name>` composite references.
   4.7. Each T-maze GFE composite retains its member-box set.
   4.8. Each T-maze GFE composite retains its dashed SVG rectangle.
   4.9. P-substitution squares keep their pre-ticket typed-scene and SVG behavior.
   4.10. Delta bead symbols keep their pre-ticket typed-scene and SVG behavior.
   4.11. Expectation bead symbols keep their pre-ticket typed-scene and SVG behavior.
   4.12. Joint domes keep their pre-ticket typed-scene and SVG behavior.
   4.13. Plates keep their pre-ticket typed-scene and SVG behavior.
   4.14. GLUE ellipses keep their pre-ticket typed-scene and SVG behavior.
   4.15. Repeated rendering of the same Gaussian template, compression level, and styles produces byte-identical SVG.

5. Focused fixtures and committed visual artifacts cover the contract.
   5.1. An LGSSM fixture with `n=4` and `m=2` renders at compression levels 0 through 4.
   5.2. The LGSSM fixture's state wires carry `ℝ⁴` captions.
   5.3. The LGSSM fixture's observation wires carry `ℝ²` captions.
   5.4. Every `y` port in the LGSSM fixture has a continuous DATA terminal.
   5.5. The LGSSM scene's reference set equals the named prior, transition, emission, equality, blob, variable, and port reference set derived from that fixture.
   5.6. A canonical LGSSM SVG golden is committed for compression level 0.
   5.7. A canonical LGSSM SVG golden is committed for compression level 4.
   5.8. The LGSSM goldens are registered in the existing parameterized golden test.
   5.9. `CFFG_UPDATE_GOLDENS=1` is the sole path that rewrites the LGSSM goldens.
   5.10. A normal golden-test run compares the LGSSM SVG without rewriting either file.
   5.11. A minimal Gaussian PRIOR-port fixture covers the continuous OPEN terminal.
   5.12. The existing Kalman-slice GLUE fixture renders its continuous GLUE boundary as an ellipsis.
   5.13. A focused scale fixture renders dimension 1 with its complete superscript.
   5.14. A focused scale fixture renders dimension 128 with its complete superscript.
   5.15. Every coordinate in the focused scale fixture is finite.
   5.16. The focused scale fixture satisfies the geometry criteria 2.9 through 2.12.
   5.17. Growing an LGSSM from three to four steps leaves the factor coordinates for `prior`, `eq0`, `trans1`, `emis0`, `eq1`, `trans2`, and `emis1` unchanged.
   5.18. The same growth comparison leaves path geometry unchanged for `x0`, `x0c`, `x0o`, `y0`, `x1`, `x1c`, `x1o`, and `y1`.
   5.19. The same growth comparison leaves domain-caption coordinates unchanged for the variables named in 5.18.
   5.20. The same growth comparison leaves terminal coordinates unchanged for `y0`, `y1`, and blob wires attached to the factors named in 5.17.
   5.21. The former final state and `emis2` are excluded from 5.17 through 5.20 because their graph structure changes when the fourth slice is appended.
   5.22. Existing HMM-slice golden checks pass.
   5.23. Existing T-maze golden checks pass.
   5.24. Existing notation-inventory checks pass.
   5.25. Existing layout-stability checks pass.
   5.26. New diagram-level Gaussian-domain checks pass.
   5.27. New scene-level Gaussian-domain checks pass.
   5.28. New SVG-level Gaussian-domain checks pass.

6. Documentation states what the new marks mean and where the data comes from.
   6.1. `docs/ARCHITECTURE.md`'s Diagram rendering section records that Gaussian domain metadata comes from `VariableSpec.dim`.
   6.2. The same section records how domain metadata crosses the typed visualization pipeline.
   6.3. The same section distinguishes `ℝᵈ` from Part I's Gaussian form-constraint bead.
   6.4. `README.md`'s visualization guidance defines `ℝᵈ`.
   6.5. The same README guidance defines the double-ring continuous DATA terminal.
   6.6. The same README guidance defines the double-ring continuous OPEN terminal.
   6.7. The same README guidance identifies the unchanged GLUE mark.
   6.8. The same README guidance identifies the unchanged POINTMASS mark.
   6.9. This ticket makes no serialized IR change.
   6.10. This ticket makes no schema-version change.
   6.11. This ticket makes no compiler-arena change.
   6.12. This ticket makes no message-update-rule change.
   6.13. This ticket makes no free-energy-term change.
   6.14. This ticket makes no runtime-status change.
   6.15. This ticket makes no bundle change.
   6.16. This ticket makes no checkpoint change.

7. The implementation is validated and handed off as one visualization change.
   7.1. Focused diagram tests pass after the final code change.
   7.2. Focused layout tests pass after the final code change.
   7.3. Focused SVG tests pass after the final code change.
   7.4. Focused Kalman rendering tests pass after the final code change.
   7.5. `tools/test-suite-runtimes.json` supplies the wait interval for final validation.
   7.6. The required short suite passes once through `uv run python tools/dashboard.py --run-suite` after the final code change.
   7.7. The final short-suite run refreshes `~/agent-share/cffg-tests.html`.
   7.8. `tk dep cycle` reports no dependency cycle before handoff.
   7.9. `git diff --check` reports no whitespace error before handoff.
   7.10. One atomic implementation commit contains `cff-bxk4` in its message.
   7.11. The ticket handoff note names the implementation commit.
   7.12. The ticket handoff note records the focused-test result.
   7.13. The ticket handoff note records the short-suite result.
   7.14. The ticket handoff note records the dashboard refresh.
   7.15. The ticket handoff note records the golden-refresh command.

## Methodological Suggestions

- Start from the `docs/ARCHITECTURE.md` Diagram rendering pipeline: `src/cffg/viz/diagram.py:derive` owns domain semantics; `compress.py` transports them without interpretation; `layout.py` and `scene.py` own typed geometry; `svg.py` owns the final marks. The change-path index has no matching compiler/runtime row. Host/JIT, flooding/scheduled, single-instance/`vmap`, and current/legacy-checkpoint pairs are all N/A because this work does not execute or persist the graph.
- Preserve the public `VariableSpec` and `Template` schema. A small domain value object or `family`/`dimension` fields on `WireView`, followed by a domain field on scene primitives, keeps renderers declarative and leaves serialization as the source of truth.
- Represent the domain caption as its own typed scene datum with the same `var:<name>` back-reference. Lay it out beside the variable-name label on the route delivered by cff-9bex; do not infer semantics by parsing a combined display string.
- Reuse the pre-ticket inner terminal mark and add an outer ring for Gaussian DATA/OPEN variants. This preserves the paper-inspired observation/open-bead vocabulary while making continuous boundaries distinguishable in monochrome output.
- Keep cff-x5lv's ownership boundary: reuse its routed label/terminal space and collision oracles where present, but do not move structural routes or redefine terminal-end parameter labels here.
- Extend the existing `lgssm` visualization tests and `tests/test_viz_svg.py` golden parameterization instead of building a second harness. Use a tiny explicit PRIOR-port template only for the boundary case that `lgssm` does not contain.
- Preserve commit `bae4d87`'s Gaussian box glyphs and registered-factor coverage. A regression should fail if the new terminal treatment accidentally replaces a factor box or lengthens a registered box glyph.
- The numerical-robustness hostile-input checklist is bounded here. Sparse support, exact log-zeros, improper/vacuous messages, and float32 are N/A because visualization reads validated structural metadata and performs no message or free-energy arithmetic. Dimension 128 and a longer LGSSM provide the required domain/layout scale pressure; invalid non-positive dimensions remain the core IR validator's responsibility.
- The original ticket author explicitly allowed delegating this complete implementation against the golden-SVG pattern. If delegated, the top-level reviewer still inspects the typed semantic contract, both canonical goldens, and the focused geometry assertions before accepting it.

## Future work

- cff-9bex owns high-degree side assignment, fan-out, and the no-factor-interior-crossing guarantee. Do not duplicate that routing algorithm here.
- cff-x5lv owns general label extent, terminal-wire staggering, and collision bounds. This ticket supplies the Gaussian caption and terminal domain data that its layout policy consumes.
- Scalar exponential-family domain notation or family-specific terminal marks are outside this ticket. Create a separate ticket if those variables need exported domain captions.
- Interactive styling, hover text, and renderer-specific color encodings are outside this monochrome SVG contract. The typed scene data should permit those future consumers without SVG parsing.
- Closing cff-bxk4 completes its optional Gaussian-visualization scope and does
  not change the dependency state of cff-ahps.

## Definitions

- **Gaussian variable**: a `VariableSpec` whose `family` is `"gaussian"` because its positive `dim` field is set.
- **Dimension**: the positive integer `VariableSpec.dim`, the number of real coordinates in one value carried by a Gaussian edge.
- **Domain caption**: the visible `ℝᵈ` annotation on a Gaussian variable wire, with `d` rendered as complete Unicode superscript digits.
- **Form constraint**: a restriction on a variational marginal's distributional family; Part I §5.2 Figure 8 writes `N` in a bead for this purpose. It is not the dimension caption.
- **DATA terminal**: the endpoint of a DATA port, represented before this ticket by a filled bead containing `∂`, which denotes an observation clamp.
- **OPEN terminal**: the open-bead endpoint used for PRIOR/GOAL ports and the renderer's defensive dangling-edge fallback.
- **Continuous-domain ring**: the outer concentric outline added around the existing DATA or OPEN bead for a Gaussian wire.
- **GLUE port**: a boundary that composes with another template instance and renders as an ellipsis.
- **POINTMASS terminal**: the black-square endpoint of a blob-backed fixed model parameter.
- **LGSSM**: the linear-Gaussian state-space model built by `cffg.models.kalman.lgssm`, with Gaussian state and observation edges.
- **Compression level**: one of levels 0 through 4 in the paper-inspired presentation pipeline; it changes variational-detail presentation without changing graph semantics.
- **Typed scene**: the renderer-independent collection of positioned paths, boxes, beads, terminals, and related primitives returned by `build_scene`.
- **Golden SVG**: a committed exact SVG string used as a deterministic regression artifact.
- **GFE composite**: the dashed scene box around the paired observation and goal factors detected for a P-substituted GFE node.

## Notes

**2026-08-08T05:03:52Z**

Ticket rewrite audit, 2026-08-07. The bounded historical slice reported 7 candidates, 8 occurrences, and 0 problems. Its mapping was: correct -> appropriate_choice; stable -> unchanged_state; test -> automated_execution; small -> model_size; can -> possibility; all -> named_class_exhaustive; only -> hard_exclusion.

Sources checked: cff-bxk4 dependency and link metadata; closed cff-4e5z Gaussian-domain handoff; open prerequisite cff-9bex routing boundary; commit bae4d87 and its Gaussian-glyph tests; VariableSpec.family/dim; lgssm state, observation, DATA, and GLUE structure; diagram, layout, typed-scene, and SVG terminal code; visualization goldens and layout/Kalman tests; docs/ARCHITECTURE.md Diagram rendering pipeline; and Part I section 5.2/Figure 8.

Final current-scan totals after consolidation: 41 candidates and 209 occurrences. Complete candidate mapping: canonical -> project_authority; fixed -> constant_parameter; n_a -> structurally_irrelevant; preserve -> semantic_invariant, compatibility, or byte_or_value_identity; public -> documented_contract; register -> metadata_declaration; reusable -> code_reuse or evidence_reuse; same -> equivalent_configuration or value_equality; expose -> internal_access; implement -> planned_work; provide -> evidence_supply; support -> probability_support; complete -> exhaustive or ticket_scope_complete; deliver -> primitive_delivered; land -> repository_change_present; remaining -> preserved_state; close -> work_item_completion; check -> automated_check; correct -> appropriate_choice; evidence -> provenance_artifact; exact -> identifier_precision, literal_zero_or_support, or bitwise_or_value_identity; match -> suitability; oracle -> test_artifact; pass -> tests_succeed or non_test_check_succeeds; robust -> hostile_numeric_survival; stable -> unchanged_state; test -> test_artifact or automated_execution; validate -> component_validation, input_validation, or acceptance_validation; infer -> metadata_derivation; scale -> domain_size; small -> code_scope; can -> demonstrated_capability; should -> obligation or design_intent; explicit_owner -> named_ticket_owner, current_ticket_owner, or component_or_actor_owner; named_ticket_handoff -> concrete_current_owner; all -> enumerated_subset_exhaustive; every -> named_class_distributive or enumerated_set_distributive; only -> hard_exclusion; current -> repository_at_reference; existing -> predates_change or repository_at_reference; read -> machine_access.

Each retained occurrence is bounded by a named type, field, role, dimension, level, fixture, file, command, commit, coordinate projection, ticket, or scope exclusion. Capability owners: cff-bxk4 owns Gaussian captions and continuous terminal semantics; cff-9bex owns routing; cff-x5lv owns general label extent, terminal staggering, and collision bounds; core IR owns validation; SVG owns final marks.

Intelligible-ticket review: complete.
