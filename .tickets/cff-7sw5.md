---
id: cff-7sw5
status: open
deps: [cff-ocxl, cff-k6g5, cff-a3wz, cdd-ziiw]
links: []
created: 2026-09-14T01:11:24Z
type: feature
priority: 2
assignee: coopwilliams
parent: cff-zov5
tags: [rslds, inference, hybrid]
---
# Compile bounded hybrid region and adapter plans

## Background in Plain English — WHY

Serialized D12 ownership needs a reconstructible component plan. A whole-rSLDS topology recognizer would reproduce the rejected historical integration and cannot preserve composition semantics. The complete numerical solver must first pass cff-a3wz so compiler integration starts from measured feasibility.

## Outcomes / Acceptance Criteria — WHAT

1. Compilation derives internal factor/variable membership, declared boundary ports, array indexes, approximation identity, objective ownership, and schedule stage from serialized semantics and graph incidence.
2. The plan is bounded and reconstructible from template plus data, includes hybrid/ordinary identity adapters, and leaves exactly one runtime array authority for beliefs and learned statistics.
3. Compiler grouping, slot addressing, parameter packing, Arena/member_of, topology, specialized/registered lowering, resolution, and assembly agree. Renaming, factor reordering, extra Gaussian/categorical attachments, and a compatible non-rSLDS boundary factor preserve membership.
4. Unsupported association cuts, duplicate entropy owners, malformed shapes, empty supported regions, and stale support declarations fail with named boundaries. No D9 or D10 fallback is selected.
5. The Derived component coordinate, New IR field, and New factor or rule family handoff classifies scheduling/readout/ledger consumers for the execution ticket. Declaration identity is preserved in the compiled executable and a successful fixed-parameter, empty-learning public CFFGZ save/load round-trip reconstructs the declared plan. Numerical host/JIT/single-vmap and learning checkpoint payload changes are N/A; cff-czdq owns full learned-state bundle/resume qualification.
6. Estimator inventory includes component arrays and linear-horizon/polynomial local storage. ARCHITECTURE records plan ownership, focused structural tests pass, and one final short-suite gate passes.

## Methodological Suggestions — HOW

These suggestions are optional; another sound method meeting the outcomes is welcome.

Use a model-independent plan protocol, and keep semantic eligibility out of names, learned-owner counts, and model helpers.

## Future work

cff-f3jz consumes plans; cff-czdq owns learning/checkpoint migration after execution.

## Definitions

- D12: the hybrid-regime-ec-v1 mathematical contract in docs/DERIVATIONS.md.
- EC: expectation consistency, matching declared moments between regional beliefs.
- Hybrid separator: one boundary carrying regime probabilities and regime-conditioned state moments.
- Adapter: an explicit equality region connecting a hybrid boundary to an ordinary categorical or Gaussian copy.
- JJ / xi: the Jaakkola–Jordan logistic quadratic bound and its squared auxiliary coordinate.
- FE / ledger: free energy and the single assembly of its owned energy/entropy terms.
- GFE: generalised free energy with P-substitution, excluded from D12.
- IR: serialized intermediate representation of model semantics.
- Arena / member_of: compiled array storage and the map from factors to its members.
- JIT / vmap: JAX compiled and batched execution surfaces.
- CFFGZ: the project bundle format containing template, data, and optional learning state.
- Gauge: redundant numerical coordinate that leaves a belief unchanged.
- Krylov: a matrix-free iterative linear solve using bounded stored vectors.
- D9 / D10: the current factor-local recurrence approximation and the retired whole-model candidate.
- Moment / covariance: an expected feature or product, and its centered second moment.
- Semantic identity: the versioned meaning of a declaration, executable, or persisted statistic.


## Notes

**2026-09-14T11:02:30Z**

Declaration boundary handoff from independent k6g5 research: compiled identity transport and the successful fixed-parameter, empty-learning public CFFGZ save/load check are explicitly owned here. K6G5 can preserve codec/hash/scoped references and bundle blob rewriting but cannot demonstrate successful public loading while D12 compilation is intentionally rejected. save_bundle validates through load_bundle; _load_archive compiles the template without executing Engine inference. Sources bundle.py:181/288/336 and learn.py:666; confidence high. Full learned-state bundles/resume remain cff-czdq after F3JZ. This allocates the existing guarantee across dependencies without reducing scope or changing the early numerical gate.

**2026-09-14T20:26:29Z**

Cooper requests an explicit opt-in boundary for D12 compilation and evidence of the feature/cost trade-off. Plans must derive D12 activation from its declared model semantics; ordinary existing models must not invoke the hybrid compilation path. The current unbound solver already performs fixed-parameter inference, with learning-array validity in auxiliary/status dependencies. cdc-ogj9 will measure scaling and output selection. Do not silently substitute D9, freeze xi, omit required certificates, or remove failure checks under a performance flag; such changes require their appropriate contract/scope decision.

**2026-09-14T23:24:33Z**

Independent read-only research for Cooper's compilation/functionality trade-off, using published PR147 source d22a43a. Question: which inputs specialize the measured unbound solver and which may vary while retaining an executable? Answer: the wrapper lowers dynamic (TransitionParameters, Natural sites, scalar initial_xi) descriptors while capturing Coordinate and Budgets. Changing numerical weights/offsets/dynamics/noise and compatible site constants/linear/precision values does not require a new argument signature. New observations and repeated learning parameter updates can therefore use a retained executable at fixed layout, precision/backend and solver policy, subject to all validity checks. This is a structural interface finding, not measured reuse speed or a promise of persistent-cache hits. Every call in the published wrapper still performs feasible initialization; executable reuse is distinct from warm-start reuse.

T/K/d, exact active/gauge/auxiliary index inventory, captured origins, budgets/line-search policy and wrapper output selection specialize the program. Active support is data-derived from exact -inf unary constants: a changed support pattern can change the coordinate dimension even when site shapes remain [T,K,d], and equal active counts do not imply equal maps. Old executable use with changed support reports SUPPORT_CHANGED. Dynamic site origins must match captured origins or fail. Optional PyTree leaves (None versus an array), scalar versus array initial_xi, dtype and device/sharding/layout compatibility also matter. Initial xi comes from the separate scalar argument; parameters.xi values are replaced by the initialized/iterated coordinate. Full-State and projected-output wrappers are different programs. Rebuilding equivalent Python closures does not guarantee JIT cache reuse because JAX includes function identity.

Sources: tools/d12_compile_scaling.py:144; src/cffg/hybrid_joint_storage.py:57; hybrid_joint.py:39,186,204,260,268; installed JAX stages.py:814, interpreters/pxla.py:1986 and api.py:209 in /Users/ravd/projects/cffg-d12-scaling. Confidence high for published argument/capture and validation boundaries. Numerical behavior across changed inputs, timed amortization and future integrated-plan cache ownership remain to be qualified. PR147 performed exactly three lower/compile-only calls, zero executable calls, and terminated each child; no live executable or cross-process reuse follows. No files changed, tests, compilation or numerical solves ran during this research.

Implementation implication for compiler-plan ownership: preserve an explicit specialization/validity boundary and retain reusable executables or stable callables when valid, rather than accidentally rebuilding data-specific closures for every parameter update. Keep D12 activation explicit and D9 paths free of D12 lowering as already required. cdc-ztuf/A3WZ will measure actual retained-executable use on authorized fixed starts; X0IF records timing using only its three frozen fits. Neither this research nor projected output trimming establishes a cheap no-learning mode or changes any numerical/approximation/scope/recovery contract.

**2026-09-15T05:08:21Z**

K6G5 handoff: explicit Template.hybrid_components and immutable separator/region/adapter/component specs now preserve all declaration identity through JSON/hash/plates/supported composition/edits/topology/bundle rewriting. core/hybrid.py owns structural admission and mapping. Reviewed admission enforces destination-selected R/H/D pairing and per-coordinate consistency through owned equalities, with negative and positive mechanism regressions. Compiler and bundle guards currently reject declared D12 before discovery/array resolution; undeclared templates retain the ordinary path. This ticket now owns replacing the compilation/publication guard with reconstructible plans while keeping a named execution guard until IDHK/F3JZ, so successful fixed-parameter empty-learning bundle reconstruction cannot accidentally enable D9 fallback. Existing plan/grouping/ownership acceptance remains authoritative. A3WZ is accepted as PR150; the serialization code passed independent review/re-review and the final 2867-test short suite at c2af46e in1145.056 seconds. No registered fits have run. Root is obtaining a bounded read-only compiler-boundary handoff before assigning the next fresh sequential author; no compiler implementation has begun.

**2026-09-15T05:13:00Z**

Independent bounded research question -> how can compiler plans and fixed-parameter bundles become available while runtime state/execution remain deferred, without D9 fallback? Answer -> snapshot explicit component ownership before ordinary discovery, publish immutable plans and resolved addresses, certify exclusive execution/objective ownership, and retain named execution guards until the runtime tickets. TopologyLayout already snapshots hybrid_components; specialized discovery/claims must consume owned factors before ordinary recurrence/mixture/equality classification. Registered lowering honors claims.consumed, while registered source_requirements still visits original consumed factors, preserving blob sources. Resolution owns actual support/shape checks and addresses; assembly must exclude internal component factors and entropy corrections from ordinary recurrence/mix groups and interior-slot tables while preserving exterior adapters.

The existing compiler guard cannot simply be removed: Engine construction and independently callable StateLifecycle, synchronous sweep, schedule, schedule-program, exact/propagation and ledger factories need a shared compiled-identity execution check before tracing or supplied flooding schedules. Fixed-parameter empty-learning CFFGZ reconstruction can still succeed: loading compiles but _restore_learning uses an Arena-only namespace and empty inventories. D12 learning inventories remain explicitly unsupported until their downstream semantics land. The current estimate.estimate constructs state and attempts cost-analysis compilation even with calibrate=False; a separate analytic Arena/plan inventory is needed. hybrid_joint_storage.buffer_values evaluates/differentiates and is not that API. Coordinate.from_sites accepts qualified chain-shaped sites; this ticket must make its mapping from declared scopes reviewable rather than treating it as general graph discovery.

Sources -> topology.TopologyLayout; specialized.discovery.discover_specialized/complete_specialized and SpecializedClaims; registered.lowering._group_keys and registered.requirements.source_requirements; resolution.resolve_compile_data; ArrayRequirement/MemberRequirement/ParameterRuntime/AddressBuilder.add; assembly_checks._execution_owners/_objective_declarations and assembly.assemble_arena; compiled.Arena interior-slot and ordinary family tables; Engine.__init__, StateLifecycle.init_state, execution.synchronous.make_sweep, schedule.build_schedule, schedule_program.build_schedule_program, make_exact_unit/make_propagate, ledger.make_belief_energy; bundle._load_archive/_restore_learning; estimate.estimate/_cost_analysis. Regression anchors are existing topology/specialized/registered/resolution/assembly/compiled/bundle/estimate suites plus direct-factory rejection and ordinary-model positive controls. Confidence -> high for current interfaces/bypasses, medium for the proposed context boundary. Read-only research at c2af46e performed no edits, tests, numerical calls or compilation.

Before implementation, split cdd-ziiw as a bounded prerequisite owning immutable plan/availability contracts, shared execution guards and analytic inventory. It depends on closed K6G5 and A3WZ, is ready, and leaves the existing compiler/bundle rejection intact. 7SW5 retains real discovery, lowering, resolved parameter/support packing, exclusive assembly and successful fixed-parameter bundles; all original outcomes remain required, with the prerequisite supplying its contract. IDHK/BRCF/F3JZ retain accepted state, ledger/readouts and numerical activation. No mathematical, supported-model, approximation, numerical or recovery threshold change is made.

Legend: prerequisite --> dependent; TK is authoritative, with additional direct dependencies retained. Companion derived from tk dep tree --full cff-7sw5, tk ready and tk blocked.

[x] A3WZ ----------------+
[x] K6G5 ----------------+--> [ ] cdd-ziiw (ready; next fresh author)
[x] OCXL ------------------------------------+       |
                                               +--> [ ] 7SW5 (blocked by cdd-ziiw)
                                                       |
                                                   [ ] IDHK --> [ ] BRCF --> [ ] F3JZ
                                                                              /   \
                                                                        [ ] CZDQ [ ] ULBZ
                                                                              \   /
                                                                            [ ] KUL2
                                                                                |
                                                                            [ ] X0IF --> [ ] ZOV5

The closed ready roots supplied independent prerequisites; implementation stays sequential. The three registered recovery fits remain unused. cdh-kvy5 remains a separate nonblocking audit.
