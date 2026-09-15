---
id: cdpf-0p7z
status: closed
deps: []
links: []
created: 2026-09-15T06:17:31Z
type: bug
priority: 2
assignee: coopwilliams
parent: cdd-ziiw
tags: [testing, hybrid, compiler]
---
# Keep compiler metadata outside frozen solver source inventories

## Background in Plain English — WHY

The final cdd-ziiw short suite completed in 1146.079 seconds, within its 1200-second cap, but failed 16 source-integrity checks. Its new hybrid_plans.py file contains compiler metadata, while the frozen tiny and moderate qualification tools use hybrid_*.py to identify the numerical solver source family. Both live inventories therefore gained one unrelated file. Direct comparison confirms no previously registered producer or tiny-checkpoint source changed or disappeared. Numerical qualification evidence must not be rewritten merely because compiler integration adds a metadata module.

This is a bounded short-suite repair owned by root under Cooper's instruction that root handle suite work personally. Implementation remains sequential while the cdd-ziiw author is paused. The parent foundation cannot land until this repair and its independent review pass.

## Outcomes / Acceptance Criteria — WHAT

1. The new immutable plan module lives under a compiler-specific name, outside the numerical hybrid_*.py source family. Its imports, no-runtime test and canonical architecture reference remain correct. Its contents and behavior remain unchanged.
2. Existing qualification source discovery, hash checks, fixtures, accepted reports, compressed archives, prospective descriptor and specification remain byte-identical. Live tiny and descriptor source maps again exactly match their registered maps, including every existing hash. Do not drop numerical producers or rewrite evidence to make a check pass.
3. All 16 failed checks and the new plan/guard/inventory regressions pass after the rename. Checks still reject changed producers and corrupted source/runtime/storage evidence. Record the diagnosis, source-set/hash comparison, commands and results.
4. An independent reviewer reviews only after root commits and pauses. One fresh final short suite after this relevant edit passes under the unchanged 1200-second cap; preserve the failed first result and do not describe it as passing. No complete numerical qualification run or registered recovery fit is performed for this repair.
5. The architecture handoff applies the Derived component coordinate publication path and qualification-tool source-identity consumers. New IR transport, resolved arrays, runtime State, numerical host/JIT/single-vmap/FE/support/convergence behavior and learning persistence are N/A: only the newly added module's import name changes. The existing compiler/bundle/runtime guards remain intact.

## Methodological Suggestions — HOW

Rename src/cffg/hybrid_plans.py to src/cffg/compiled_hybrid.py with git mv and update its direct imports and architecture module reference. Keep tests/test_hybrid_plans.py as the behavioral test file. Use the existing frozen-source and descriptor-integrity tests to exercise the source-family boundary, plus a byte comparison against the pre-repair head. Another sound method is welcome if it preserves all acceptance criteria.

Hostile inputs: sparse support, improper cavities, float32 arithmetic and larger numerical domains are N/A to the import-only rename; existing plan f32/f64 and shape-only tests must remain green. Changed producer bytes, omitted source entries and corrupt inventory/runtime bindings remain rejected. Keep stored versus future versus unavailable memory semantics unchanged.

## Future work

cdd-ziiw retains foundation closure/publication; cff-7sw5 will derive real compiled plans. Subsequent runtime and recovery tickets remain unchanged. A general redesign of qualification provenance is not part of this repair.

## Definitions

- Source inventory: a list of files and their content hashes used to identify the implementation behind qualification evidence.
- Producer: a source file that constructs the measured numerical program or its abstract storage descriptors.
- Descriptor: array shapes, dtypes and logical storage counts obtained without executing the complete solver.
- D12: the accepted hybrid-regime-ec-v1 approximation; this repair changes none of its mathematical or numerical requirements.
- Short suite: the default not-slow pytest gate, with an enforced 1200-second wall-clock limit.



## Notes

**2026-09-15T06:20:41Z**

Root completed the bounded repair and now pauses for independent review. The new metadata module is renamed from hybrid_plans.py to compiled_hybrid.py, with its direct imports, no-runtime test and canonical architecture reference updated. The module content is byte-identical to the original at cfd3c19. No numerical producer, source-discovery function, source hash check, threshold, specification, descriptor, fixture or accepted evidence artifact changed.

Post-repair source audit: all 12 tiny-checkpoint hashes, all 17 descriptor-producer hashes and all 32 registered moderate-source hashes exactly match the accepted maps. No registered source was omitted. Evidence: /tmp/cdpf-0p7z-source-audit.json. Failed-gate evidence remains in the parent ticket, runtime history and /tmp/cdd-ziiw-first-gate-summary.json.

Focused validation after final relevant edits: uv run pytest -q tests/test_hybrid_plans.py tests/test_d12_moderate.py tests/test_d12_moderate_integrity.py tests/test_hybrid_joint_solver.py::test_frozen_checkpoint_sources --junitxml=/tmp/cdpf-0p7z-focused.xml -> 146 passed, 2 deselected in 56.69 seconds. This covers all 16 previously failed checks, the metadata/guard/inventory contract, native-float32 abstract storage, and source/runtime/storage corruption rejection. Log /tmp/cdpf-0p7z-focused.log. git diff --check and tk dep cycle pass. No complete D12 qualification solve, recovery fit or full suite has run since the rename.

Root pauses after this committed repair; the independent reviewer must verify the import-only diff, exact preserved qualification sources/artifacts and focused evidence. After acceptance, root will run one fresh final short suite under the unchanged 1200-second cap. Both this ticket and cdd-ziiw stay in_progress until that gate passes; publication remains root-owned.

**2026-09-15T06:23:33Z**

Independent review at 9c4a8f0712ba6b75e50777fc6cbca65ba482161e is accepted with no findings. The reviewer started only after root's committed clean pause and independently recomputed the live tiny (12), descriptor (17) and moderate (32) source maps. All accepted names and hashes match exactly; no additions, omissions or changed existing producers. The renamed compiled_hybrid.py is byte-identical to the original at cfd3c19. No source-discovery tool, numerical producer, specification, descriptor, fixture or accepted measurement changed.

The reviewer confirmed passing focused XML outcomes for all 16 prior failed node IDs and independently ran 43 bounded metadata/import/guard/ordinary-path/no-runtime/frozen-source tests, all passed in 0.33 seconds. git diff --check passed and the tree was clean. No full suite, complete D12 program or recovery data was used during review. Root has received the completed review and now owns the single fresh final short-suite gate after the rename, under the unchanged 1200-second cap. The earlier failed gate remains recorded. This ticket and cdd-ziiw remain in_progress pending that gate and closure/publication.

**2026-09-15T06:45:46Z**

Final gate passed at 2bfba267709d47605369a773491692783f60b0b8 after all implementation, review repairs and the source-family module move: 2909 passed, zero failures, 16 warnings; measured wall 1126.198 seconds (18m46.198s), 73.802 seconds below the unchanged 1200-second cap. Four workers, unchanged not-slow selection, empty initial executable cache. Phase audit verifies 8727 records, 2909 distinct calls, zero duplicates, every phase passed. Log /tmp/cdd-ziiw-final-short-suite.log; audit /tmp/cdd-ziiw-final-gate-summary.json. Dashboard and runtime history are refreshed. Root announced once and waited quietly for the full 1200-second suggested budget without polling. The earlier failed source-inventory gate remains recorded; this passing result follows a relevant import-name edit and is not an unchanged rerun.

cdd-ziiw and its bounded suite repair cdpf-0p7z are complete. Final foundation: immutable compiled_hybrid plan records, empty-by-default Arena component inventory, shared named execution guards including host/JIT support factories, and no-runtime analytic inventory with explicit dtype and honest partial/unavailable storage categories. Current compiler/bundle D12 rejection remains intact. Independent initial review, repair re-review and separate root-authored source-boundary repair review are accepted. The renamed metadata module is byte-identical; all 12 tiny, 17 descriptor and 32 moderate accepted source hashes remain exact. No qualification artifact, numerical requirement or recovery threshold changed.

Next: publish the stacked foundation PR on cff-k6g5-hybrid-declarations, then assign a fresh sequential author to cff-7sw5 from the published head. 7SW5 owns real semantic discovery, resolved shapes/support/parameter addresses, exclusive assembly, declared plan publication and successful fixed-parameter empty-learning bundles. IDHK/BRCF/F3JZ retain accepted State, objective/readout ownership and runtime activation; CZDQ/ULBZ/KUL2 retain learning/resume and composition before X0IF. Keep the new compiled_hybrid module outside the numerical hybrid_*.py source family; do not regenerate accepted evidence for metadata-only integration. All three registered recovery fits remain unused. No merge is authorized or performed.
