---
id: cff-cpxj
status: closed
deps: [cff-wx28]
links: []
created: 2026-07-09T03:56:19Z
type: feature
priority: 2
assignee: coopwilliams
parent: cff-ahps
---
# Plates: compact replication of subgraphs (grids, banks, layers)

Cooper 2026-07-09: factor-graph plates — draw a box around a fragment and declare it duplicated N times — so CFFG specs for repeated structure (pixel grids, mixture banks, sensor arrays, scaled layers) stay compact instead of ridiculously over-detailed. ARCHITECTURE (designed, ready to implement):

(1) IR: Template grows plates: dict[name, PlateSpec]; PlateSpec = (fragment: nested Template, dims: tuple[int, ...] for 1-D/2-D/N-D counts, bonds: wiring rules, per_copy_blobs: list[str]). Parameters are TIED by default — copies share the fragment's blob names (already natural: blobs are named refs); per_copy_blobs opts specific blobs out (indexed names), cheap to initialize via init specs (cff-wx28).

(2) WIRING VOCABULARY (the design core — patterns, not arbitrary index arithmetic): 'chain(port_a, port_b, axis, offset=1)' bonds copy i's port to copy i+offset's (1-D chain plate = unrolling's pattern); 'grid' is chain per axis (pixel neighborhoods); 'broadcast(port, hub)' bonds every copy's port toward one shared edge — NOTE: FFG edges have max degree 2, so broadcast expansion must SYNTHESIZE an equality tree (this is where plates earn their keep; the expansion owns that plumbing); 'expose' turns remaining fragment ports into indexed parent ports plate/i/port, with an aggregated bind path so a whole image tensor binds to a plate's DATA ports in one call.

(3) GROUNDING: expansion happens in the compiler's grounding phase. Phase 1 implementation may materialize copies naively (correctness; reuses compose/scoped). Phase 2 target: expand INDEX TABLES, not Python objects — copies are structurally identical, so they drop into the existing [M, ...] group tensors by arithmetic; a 64x64 pixel plate should compile in about a second and its arena is exactly as vectorized as PGMax would make it. The arena design was built for this.

(4) SEMANTICS: structure hash covers the plate SPEC (fragment + dims + bonds) — compact, and resizing dims IS a structural change (recompile; correct for colony batching). Edit action set_plate_dims(name, dims) is the GUI-ready resize. Viz draws the literature's rectangle-with-count badge using the fragment's own layout, never the expansion (ties into the compression passes). Resource estimator reports cost as an analytic function of dims without expanding.

(5) OPEN QUESTION (resolve during implementation, note in ticket): does unroll migrate onto 1-D chain plates (one mechanism) or stay separate? Receding-horizon slices are NOT homogeneous (past slices carry data beads, future slices goal/P-sub beads); plates v1 should require homogeneous copies and leave unroll alone unless index-dependent constraint templating proves cheap.

VALIDATION USE CASES: pixel-grid observation layer (2-D plate, tied A, aggregated image bind); mixture bank broadcasting into a selector (equality-tree synthesis); N-sensor array into one state; plate resize edit stability. DELEGATION: design above is Fable-reviewed and final unless implementation contradicts it; phase-1 implementation (materializing expansion + wiring vocabulary + tests) is a good Opus ticket with the use cases as acceptance; phase-2 index-arithmetic and any FE-ledger subtleties come back to Fable.

## Acceptance Criteria

Phase 1: all four validation use cases compile, run, and serialize compactly (spec size independent of dims); equality-tree synthesis correct (marginals match a hand-expanded reference); structure hash covers spec not expansion; viz shows plate boxes. Phase 2 (may split): index-table expansion with the 64x64 compile-time target.


## Notes

**2026-07-09T22:04:33Z**

CLOSED (branch plates, chained on gauss-vmp PR #11). Phase 1 landed as designed: PlateSpec IR (fragment/dims/bonds/per_copy_blobs) + Chain/Broadcast/Expose wiring; expand_plates at the top of grounding (pure, reuses compose scoping); equality-tree synthesis for broadcast (N=1 hub==branch, N=2 one ternary eq, N>=3 chain); serialization additive with recursive blob-path erasure in structure_hash (spec-level: dims change the hash, fragment blob moves don't; spec JSON size independent of copy count — 4x4 vs 9x9 identical length); SetPlateDims edit + codec; viz plate boxes with xN badges drawn from the SPEC. All four validation use cases pass against hand-built references (pixel grid == single-pixel model 1e-9; broadcast hub == hand-expanded equality tree AND analytic product 1e-10; sensor array == information addition 1e-9; resize edit rehashes+recompiles). Design refinements (reported, consistent with intent): broadcast hub is SYNTHESIZED and exposed as a GLUE parent port <plate>/<hub>; leftover fragment ports always exposed (Expose() is the explicit marker); estimate.py untouched (analytic-in-dims is phase 2). OPEN QUESTION ANSWERED: unroll stays separate — rollout slices are not homogeneous (past=data beads, future=goal/P-sub beads); the 1-D chain plate reproduces an unrolled HMM's smoothed marginals exactly where copies ARE homogeneous (test included). Phase 2 (index-table expansion, ~1s 64x64 target, analytic estimator): cff-6s31. Suite: 1067 green.
