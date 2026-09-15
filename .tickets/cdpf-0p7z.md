---
id: cdpf-0p7z
status: in_progress
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
