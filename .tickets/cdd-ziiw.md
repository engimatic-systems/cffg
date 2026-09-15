---
id: cdd-ziiw
status: closed
deps: [cff-k6g5, cff-a3wz, cdpf-0p7z]
links: []
created: 2026-09-15T05:11:58Z
type: task
priority: 2
assignee: coopwilliams
parent: cff-7sw5
tags: [rslds, compiler, hybrid]
---
# Define immutable hybrid plans, execution guards, and analytic inventory

## Background in Plain English — WHY

The D12 numerical checkpoint and serialized declarations now pass. Compiler integration must next publish a reconstructible description of each hybrid component. That description must remain safe to inspect and store before the engine can execute it. Removing the current compilation guard without explicit ownership and execution guards could let ordinary D9 code omit or misinterpret a D12 component.

This bounded prerequisite extracts immutable plan contracts, execution availability, and inspection without execution from cff-7sw5. It leaves the current D12 compiler and bundle rejection in place. The following compiler ticket will construct real plans and resolve their arrays; cff-idhk and cff-f3jz still own runtime state and execution.

## Outcomes / Acceptance Criteria — WHAT

1. Immutable compiled-plan records represent explicit D12 declaration identity, component/region/separator/adapter ownership, structural indexes and parameter/address metadata needed by the compiler. The contract states which values are derived and preserves one future authority for accepted beliefs and learning statistics. Plans do not store duplicate mutable belief/state arrays or assert execution support. Ordinary Arenas retain an empty/default component inventory and their existing behavior.
2. One named availability check rejects unsupported D12 execution before JIT wrapping or ordinary fallback. It covers Engine construction and independently callable lifecycle, synchronous sweep, schedule, schedule-program, exact/propagation and free-energy factory entry points. A supplied flooding schedule cannot bypass the check. Errors identify the component and approximation. Ordinary models remain executable and do not enter D12 lowering.
3. An analytic inventory API accepts compiled Arena/plan metadata and explicit dtype without constructing an Engine, initializing state, evaluating numerical kernels, differentiating, lowering or compiling a program. It reports stored plan/parameter arrays separately from declared future coordinate/scratch requirements and from unavailable physical/compiled peak memory. It does not count future State allocations as already owned persistent arrays or use hybrid_joint_storage.buffer_values to obtain estimates.
4. Malformed plan identity and unknown approximation versions fail by component name. Immutability, ownership metadata and byte/element accounting have meaningful regressions; guard tests exercise direct factory bypasses and an ordinary-model positive control. Empty plans, sparse fixed-support metadata, float32/float64 dtype accounting and larger shape-only inventories are handled explicitly. Exact-zero numerical arithmetic, improper cavities, convergence and free energy values are N/A because no solver executes here; actual resolved support/array consistency remains cff-7sw5.
5. The existing named compiler and bundle guards remain intact. This ticket performs no real D12 plan lowering, numerical solve, registered recovery fit or activation. Successful declared fixed-parameter bundles remain cff-7sw5; learned-state identity/resume remains cff-czdq. No mathematical contract, model scope, approximation semantics or numerical/recovery threshold changes.
6. The pre-implementation handoff applies the Derived component coordinate, New IR field and runtime availability aspects of Runtime numerical or status invariant change paths in ARCHITECTURE. It classifies immutable compiled/Arena publication, Engine versus standalone factories, synchronous/scheduled/exact paths and analytic versus runtime estimation. Host/JIT guards apply before tracing; numerical host/JIT and single/vmap parity, actual State transport, ledger terms, readouts and learning persistence are deferred to their existing tickets, not claimed as implemented.
7. Canonical documentation explains the safe plan/availability/inventory boundary. Focused checks pass after final edits, new tests are registered in dashboard EPIC_FILES, independent review is accepted, and root's one final short-suite gate passes under the existing 1,200-second cap before closure/publication.

## Methodological Suggestions — HOW

Use existing ImmutableRecord/compiled plan conventions and an explicit unavailable/deferred execution marker. Coordinate.from_sites is a qualified solver-layout builder for [T,K,d] sites, not a general declaration-discovery interface; preserve a reviewable mapping contract without invoking it during analytic inventory. Reuse existing ownership and address types where their meaning fits, and avoid speculative generic frameworks.

Inspect topology.TopologyLayout, compiled.Arena, specialized claims, assembly checks and the current estimator before choosing the record boundary. The independent research in cff-7sw5 names concrete insertion points. Guard public construction and independently callable execution factories rather than duplicating checks in every numerical helper. Keep scalar/shape inventory distinct from physical allocation measurements. Another sound method meeting the outcomes is welcome.

## Future work

cff-7sw5 derives real ownership, resolves shapes/support/parameters, packs addresses and certifies assembly, then enables compiled-plan and fixed-parameter empty-learning bundle reconstruction while preserving execution rejection. cff-idhk transports one accepted runtime coordinate; cff-brcf implements ledger/readout ownership; cff-f3jz activates the numerical solver. cff-czdq, cff-ulbz and cff-kul2 complete learning, exterior mixtures and public composition before cff-x0if's three registered fits and cff-zov5.

## Definitions

- D12: the accepted hybrid-regime-ec-v1 approximation in DERIVATIONS; D9 is the existing ordinary recurrence approximation.
- Plan: immutable derived metadata explaining component ownership, indexes and parameter addresses.
- Arena: the compiler's array container and ownership/address maps.
- JIT: compilation of array code into an executable program; no such compilation occurs during analytic inventory.
- CFFGZ: the portable template/data/optional-learning bundle format.
- Coordinate: the numerical unknowns and fixed indexing metadata for one accepted component state.
- Scratch: temporary storage required during a numerical iteration, distinct from persistent State and compiler process memory.
- Analytic inventory: counts derived from known arrays, shapes and dtypes without executing the model.
- Float32/float64: floating-point formats occupying four/eight bytes per scalar.
- Host/JIT, synchronous/scheduled, and single/vmap: eager/compiled, full-sweep/planned-sweep, and individual/batched execution surfaces.



## Notes

**2026-09-15T05:13:00Z**

Root assigned this bounded prerequisite before any 7SW5 implementation, following the independent compiler-boundary research distilled into cff-7sw5. K6G5 passed at c2af46e: 2867 tests, 1145.056 seconds under the 1200-second cap. A3WZ is accepted and published as PR150. Root will provide a fresh worktree from the completed PR151 stack and a fresh sequential author. Preserve the current compiler/bundle rejection until 7SW5; no registered fit or numerical benchmark is part of this ticket. Every original 7SW5 outcome remains required through this prerequisite plus its own implementation.

**2026-09-15T05:16:30Z**

Pre-implementation handoff: applies ARCHITECTURE change-path rows Derived component coordinate, New IR field, and Runtime numerical or status invariant (availability only). Immutable compiled publication will add an empty-by-default Arena component tuple and explicit plan records for serialized identity, named region/separator/adapter ownership, structural indexes, existing member_of parameter-address references, and optional declared coordinate/scratch shapes. Compilation/discovery/resolution/assembly remains unchanged and its named D12 rejection stays active.

Affected duplicated surfaces: Engine versus independently callable StateLifecycle, synchronous make_sweep, build_schedule, build_schedule_program (including supplied flooding schedules), make_support_check, make_exact_unit, make_propagate, ObjectiveLedger and make_belief_energy all use one host availability check before callbacks or JIT wrappers. Availability comes from implemented runtime support, never a caller-supplied enabling flag. Analytic Arena/plan inventory is separate from the existing execution/calibration estimator and traverses only stored immutable array metadata; future coordinate/scratch requirements remain declared logical sizes, with unknown requirements and physical/compiled peak memory explicitly unavailable.

Checked N/A/deferred pairs: numerical host/JIT and single/vmap parity, synchronous/scheduled D12 numerical behavior, runtime State transport, convergence, ledger values, readouts, learning persistence and checkpoint migration remain IDHK/BRCF/F3JZ/CZDQ; no numerical kernel changes. New IR codec/scoping/edit behavior is N/A because K6G5 already owns the serialized declarations. Actual array/support/address consistency and exclusive assembly certification remain 7SW5.

Hostile inputs: empty component inventory preserves ordinary execution and reports no hybrid requirements; malformed component identity/unknown approximation fail with component names; sparse fixed support remains explicit immutable metadata; float32/float64 accounting uses explicit dtype sizes independent of JAX configuration; large shapes are counted with Python integers without arrays/state allocation. Exact-log-zero arithmetic, improper/vacuous cavities, FE/convergence and solver scale behavior are provably N/A because this ticket executes no solver. Test direct-factory bypass rejection before callback/JIT activity and an ordinary-model execution control. Register the new focused test in dashboard EPIC_FILES. Root owns independent review, one final short-suite gate under 1200 seconds, ticket closure and publication; author pauses only after committed focused evidence.

**2026-09-15T05:42:59Z**

Implementation handoff for independent review: immutable hybrid_plans records preserve complete declaration identity, named separator/region/adapter ownership, ordered separator indexes, ordinary adapter slots and references to existing Arena.member_of parameter addresses. Arena.hybrid_components is an empty-by-default final field, preserving existing positional constructor arguments. The shared require_execution_available check rejects all published D12 components before Engine/lifecycle, synchronous/schedule/program (including supplied flooding), support/exact/propagation and objective-ledger construction. No caller flag can enable execution. Current compiler and bundle rejection remain unchanged.

The separate inventory.analytic_inventory entry (also exported through estimate) reads stored array metadata and explicit float32/float64 dtype only. It distinguishes stored plan/default-parameter arrays, ordinary future State requirements and optional declared hybrid coordinate/scratch requirements. Repeated published-array references are counted once with aliases listed. Unknown future requirements and physical/compiled peak memory remain unavailable; no coordinate/state is constructed and no evaluation, differentiation, lowering or compilation is performed by inspection. ARCHITECTURE, README and LIMITATIONS describe this boundary; test_hybrid_plans.py is registered in dashboard EPIC_FILES.

Final focused validation after the last code edit: uv run pytest -q tests/test_hybrid_plans.py tests/test_hybrid_declarations.py tests/test_compiled.py tests/test_engine_contracts.py tests/test_lifecycle.py tests/test_schedule.py tests/test_schedule_program.py tests/test_ledger.py tests/test_estimate.py -> 199 passed in 30.46 seconds. Regressions cover malformed identity/unknown approximation, immutable input snapshots, sparse fixed support, explicit float32/float64 accounting, unknown/empty/large shape-only requirements, aliases, named rejection of every independent execution factory before JIT, ordinary execution controls, and import/inspection with numerical-runtime modules forbidden. git diff --check passed; tk dep cycle found no cycles. No D12 solve, registered recovery fit or full short suite ran.

Author pauses at this committed implementation for root's independent review. Next: accept review/fix any findings, then root runs the single final short-suite gate under the 1200-second cap before closure/publication. Remaining gotcha: these manually constructible records certify only identity/local index metadata, not resolved support/parameter arrays or graph assembly. 7SW5 still owns real compiler derivation/resolution/exclusive assembly and successful fixed-parameter bundles. IDHK/BRCF/F3JZ/CZDQ retain State, objective/readouts, execution and learning/resume; their work is not activated or claimed here. Ticket stays in_progress until root completes review and final validation.

**2026-09-15T05:47:31Z**

Independent review complete at f3be4b9ad93a82d5652344172005db0e95af8ae9, after the author's clean committed pause. A fresh reviewer inspected execution guards; root inspected plan identity, immutability, analytic inventory and documentation. Three bounded P2 corrections are required before the final gate:

1. Host support-check bypass: runtime_checks.StateChecks.__post_init__ lacks the shared execution-availability guard. For the committed valid hand-published D12 Arena, StateChecks(arena) succeeds and has_zero_support(SimpleNamespace(messages=np.empty(0), params={})) returns False, while make_support_check raises the named ExecutionUnavailableError. Guard the host factory before support metadata is constructed and cover it in the direct-factory regression. This restores the host/JIT boundary; it does not add D12 support arithmetic.
2. Explicit dtype: inventory.analytic_inventory currently applies np.dtype before validation, so dtype=None silently returns a float64 inventory. Reject an unspecified dtype before NumPy can select its default; cover this alongside the existing float32/float64 and unsupported-dtype cases. Explicit future precision is an acceptance requirement, independent of JAX defaults.
3. Inventory labeling: ordinary_state_requirements lists eight nonparameter State buffers but omits State.params. An Arena with only a 100x100 float64 default parameter reports 80,000 stored-default bytes and zero future ordinary State bytes. Stored defaults and runtime parameter arrays are distinct, so this label and the current prose can mislead readers into treating a partial buffer list as a complete future State estimate. Either provide a separately justified future-parameter requirement or keep this ticket bounded by explicitly naming/documenting the eight buffers as the nonparameter subset and marking runtime-parameter/complete-State requirements unavailable. Do not silently count stored defaults as future State, promise complete/physical memory, or construct runtime state to estimate it.

No other findings in the reviewed scope. The independent guard reviewer ran 18 focused tests, all passed; root's metadata-only repros confirmed the two inventory findings. Immutable declaration snapshots, ordered indexes, parameter address references, deferred runtime authority and named compiler/bundle rejection remain sound within this ticket's stated boundary. Runtime availability is not caller-controlled. No numerical solve, registered recovery fit or full suite ran during review. Root notifies the original author only now that review is complete. Repair these findings, update focused evidence after final edits, commit and pause; root will re-review before the one final short-suite gate under the unchanged 1200-second cap.

**2026-09-15T05:49:35Z**

Repair handoff after independent review: the original author is paused. Root could not resume that author or start a fresh worker because agent capacity checks rejected three attempts. Root therefore reassigned the completed guard reviewer to implement only the three recorded corrections; the original review is preserved in ae0955e and root will independently review the repairs after the new committed pause. There is no concurrent implementation and the repair author will not accept their own changes.

The existing Derived component coordinate, New IR field and Runtime numerical/status availability change paths still apply. This repair adds runtime_checks.StateChecks to the explicitly guarded host/JIT support pair, alongside schedule_program.make_support_check, before host support metadata construction. Inventory changes reject dtype=None and name the eight nonparameter State buffers as a subset, with runtime-parameter and complete-State counts unavailable. Existing compiler/bundle rejection and all numerical/execution/learning deferrals remain intact. Regression inputs are a valid hand-published D12 Arena at direct host construction, omitted/None/unsupported dtype, and a large stored-default array with zero nonparameter buffers. Root retains the single final short-suite gate, closure and publication; this worker runs focused validation only.

**2026-09-15T05:50:04Z**

Host support correction: StateChecks now calls the shared named availability guard before constructing edge groups, matching make_support_check. The direct-factory rejection table includes StateChecks. Focused validation: uv run pytest -q tests/test_hybrid_plans.py tests/test_runtime_status.py tests/test_engine_contracts.py -> 51 passed in 2.96 seconds. No D12 numerical operation or full suite ran. The two inventory corrections remain in progress for the next atomic commit.

**2026-09-15T05:52:57Z**

Review corrections completed. Commit 79a4ed0 guards StateChecks before host support metadata construction and tests that direct entry. The inventory correction rejects explicit dtype=None before NumPy can choose float64. Its eight ordinary nonparameter State buffers are now exposed as ordinary_nonparameter_state_requirements; runtime_parameter_bytes and complete_state_bytes explicitly return None. Stored default parameters retain their separate actual-dtype inventory. README, ARCHITECTURE and LIMITATIONS explain this bounded scope. A regression stores a 100x100 float64 default (80,000 bytes), requests float32 future buffers, observes zero nonparameter-buffer bytes, and requires unavailable runtime-parameter and complete-State counts. The existing inspection-without-numerical-imports regression is preserved.

Final focused validation after the last code edit: uv run pytest -q tests/test_hybrid_plans.py tests/test_hybrid_declarations.py tests/test_compiled.py tests/test_engine_contracts.py tests/test_lifecycle.py tests/test_schedule.py tests/test_schedule_program.py tests/test_ledger.py tests/test_estimate.py tests/test_runtime_status.py -> 210 passed in 30.63 seconds. This includes the host/JIT direct guards, ordinary auto/flood execution, explicit float32/float64 and None rejection, no-runtime inspection and stored-default/subset boundary. git diff --check passed and tk dep cycle found no cycles. No full short suite, D12 lowering/solve or registered recovery fit ran.

The repair author pauses after committing with a clean tree. Root independently re-reviews the full correction range from ae0955e; the repair author does not accept their own corrections. Root still owns the one final short-suite gate under the unchanged 1,200-second cap, ticket closure and publication. No new ticket or downstream implementation began. Runtime State.params and complete-State estimation remain unavailable here; all original 7SW5/IDHK/BRCF/F3JZ/CZDQ deferrals and compiler/bundle guards remain unchanged.

**2026-09-15T05:53:51Z**

Root independently re-reviewed the completed repair range ae0955e..edafd3be35fab02bcc0d0be05200ab9f850c4c83 after the repair author committed and paused. All three review findings are resolved; no further findings. StateChecks uses the shared named guard before host support metadata construction, matching make_support_check. None is rejected before NumPy dtype normalization. The renamed ordinary_nonparameter_state_requirements lists exactly eight documented buffers; runtime_parameter_bytes and complete_state_bytes are explicitly unavailable, with stored defaults kept distinct. Canonical docs and regressions match those limits. Compiler/bundle rejection and numerical code are unchanged.

Independent focused command: uv run pytest -q tests/test_hybrid_plans.py -k 'standalone_execution_factories or explicit_supported_dtype or stored_defaults or ordinary_models or imports_and_runs' --junitxml=/tmp/cdd-ziiw-root-rereview.xml. All 22 selected tests passed in 0.23 seconds; log /tmp/cdd-ziiw-root-rereview.log. No stale ordinary_state_requirements references remain in source/tests/canonical docs. git diff --check and tk dep cycle passed. Root notifies the repair author after this completed re-review; the original author thread could not be resumed/notified because the service rejected its activation at the thread limit, so the durable TK note preserves the same handoff. Root's independent re-review maintains separation from the repair author.

No D12 lower/compile/solve or registered recovery fit ran. Next: root's single final short-suite gate after this final code state, under the existing 1200-second cap, followed by closure and stacked PR publication if it passes. This remains a foundation only; cff-7sw5 retains real discovery, array/support resolution, ownership-certified assembly and fixed-parameter bundle reconstruction, with runtime integration deferred to its existing dependencies.

**2026-09-15T06:20:11Z**

The final short suite at cfd3c19c2388125808a1e46f537763c25c3cc775 failed within the cap: 2893 passed, 16 failed, 1146.079 seconds, 4 workers, unchanged not-slow selection and an empty initial executable cache. Phase audit records 8727 phases and 2909 distinct calls, zero duplicates. Log /tmp/cdd-ziiw-short-suite.log; audit /tmp/cdd-ziiw-first-gate-summary.json. The dashboard and runtime history truthfully show failure. This is not an accepted final gate.

Diagnosis: all 16 failures trace to the new compiler-only hybrid_plans.py being included by the qualification tools' hybrid_*.py source-family scan. Comparison with the registered tiny and descriptor maps shows exactly one added file, no removed files and no changed existing hashes. The native abstract-storage subprocess ends at the same descriptor producer mismatch; tracing-cache messages in its stderr are not the failing assertion. No numerical qualification result or registered recovery fit changed.

Root created cdpf-0p7z as a bounded required repair, under Cooper's instruction that root handle short-suite work personally. Move the new metadata module to compiled_hybrid.py and update imports; preserve the numerical source discovery/checks, frozen specifications, descriptors and accepted evidence byte-for-byte. Independent review follows root's committed pause, then one fresh final suite after this relevant edit. The original failed result remains recorded.

Legend: prerequisite --> dependent. Based on tk dep tree --full cff-7sw5, tk ready, tk blocked; no dependency cycles.

[x] A3WZ + [x] K6G5 ----+
[ ] cdpf-0p7z (ready) ---+--> [ ] cdd-ziiw --> [ ] 7SW5 --> [ ] IDHK
                                                              |
                                       remaining TK chain --> [ ] X0IF --> [ ] ZOV5

cdpf-0p7z is the immediate landing blocker; cdh-kvy5 remains an independent nonblocking audit. Existing direct dependencies and the downstream runtime/learning/composition chain remain authoritative in TK. Implementation stays sequential. All three registered recovery fits remain unused.

**2026-09-15T06:45:46Z**

Final gate passed at 2bfba267709d47605369a773491692783f60b0b8 after all implementation, review repairs and the source-family module move: 2909 passed, zero failures, 16 warnings; measured wall 1126.198 seconds (18m46.198s), 73.802 seconds below the unchanged 1200-second cap. Four workers, unchanged not-slow selection, empty initial executable cache. Phase audit verifies 8727 records, 2909 distinct calls, zero duplicates, every phase passed. Log /tmp/cdd-ziiw-final-short-suite.log; audit /tmp/cdd-ziiw-final-gate-summary.json. Dashboard and runtime history are refreshed. Root announced once and waited quietly for the full 1200-second suggested budget without polling. The earlier failed source-inventory gate remains recorded; this passing result follows a relevant import-name edit and is not an unchanged rerun.

cdd-ziiw and its bounded suite repair cdpf-0p7z are complete. Final foundation: immutable compiled_hybrid plan records, empty-by-default Arena component inventory, shared named execution guards including host/JIT support factories, and no-runtime analytic inventory with explicit dtype and honest partial/unavailable storage categories. Current compiler/bundle D12 rejection remains intact. Independent initial review, repair re-review and separate root-authored source-boundary repair review are accepted. The renamed metadata module is byte-identical; all 12 tiny, 17 descriptor and 32 moderate accepted source hashes remain exact. No qualification artifact, numerical requirement or recovery threshold changed.

Next: publish the stacked foundation PR on cff-k6g5-hybrid-declarations, then assign a fresh sequential author to cff-7sw5 from the published head. 7SW5 owns real semantic discovery, resolved shapes/support/parameter addresses, exclusive assembly, declared plan publication and successful fixed-parameter empty-learning bundles. IDHK/BRCF/F3JZ retain accepted State, objective/readout ownership and runtime activation; CZDQ/ULBZ/KUL2 retain learning/resume and composition before X0IF. Keep the new compiled_hybrid module outside the numerical hybrid_*.py source family; do not regenerate accepted evidence for metadata-only integration. All three registered recovery fits remain unused. No merge is authorized or performed.
