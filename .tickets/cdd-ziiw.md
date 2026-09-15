---
id: cdd-ziiw
status: open
deps: [cff-k6g5, cff-a3wz]
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

Root assigned this bounded prerequisite before any 7SW5 implementation, following the independent compiler-boundary research distilled into cff-7sw5. K6G5 passed at c2af46e: 2867 tests, 1145.056 seconds under the1200-second cap. A3WZ is accepted and published as PR150. Root will provide a fresh worktree from the completed PR151 stack and a fresh sequential author. Preserve the current compiler/bundle rejection until 7SW5; no registered fit or numerical benchmark is part of this ticket. Every original 7SW5 outcome remains required through this prerequisite plus its own implementation.
