---
id: cdhd-0ios
status: closed
deps: [cdd-ziiw]
links: []
created: 2026-09-15T07:58:45Z
type: bug
priority: 2
assignee: coopwilliams
parent: cdpf-gzix
tags: [hybrid, compiler, testing]
---
# Use declared region kinds in hybrid recurrence requirements

## Background in Plain English — WHY

The combined cdpf-gzix/cdhd-fyq6 short suite found that hybrid requirement lowering selects recurrence roles by testing a factor's type again. The ownership check requires ordinary recurrence discovery to have one classification owner. D12 already has a validated explicit region kind, so requirement lowering should consume that declaration instead of rediscovering a recurrence factor. The failed gate at 407299d completed in 1152.414 seconds with 2945 passes and this one ownership failure, below the unchanged 1200-second cap.

## Outcomes / Acceptance Criteria — WHAT

1. Hybrid requirement lowering selects recurrence parameter roles from the validated explicit recurrence region kind. It preserves original topology/source traversal, names, parameter transformations, ownership and all other admitted region requirements.
2. The unchanged test_recurrence_classification_has_one_discovery_owner passes. No allow-list expansion, test exclusion or pass/fail reinterpretation hides the duplicate classification.
3. Existing actual declaration-to-draft checks cover recurrence, dynamics, ordinary unary, selected Gaussian unary and one-label unary requirements. Focused extension ownership and discovery tests pass after the repair.
4. Public compiler/bundle/runtime guards, numerical sources, qualification artifacts and all recovery attempts remain unchanged.
5. Root personally implements this bounded short-suite repair. Independent review follows a committed clean pause; one fresh final combined short suite under the unchanged 1200-second cap then gates this ticket and its discovery parent.

## Methodological Suggestions — HOW

Pass the validated declaration region through the per-factor requirement boundary, preserving factor traversal order. Reuse the existing ownership and requirement regressions. Another sound method meeting the outcomes is welcome. Applicable architecture row: Derived component coordinate, limited to declaration/claim-to-parameter-draft interpretation. Checked duplicated surfaces: no serialization change; no numerical host/JIT, synchronous/scheduled, single/vmap, state/ledger/readout, checkpoint or bundle activation. Sparse support and K=1 metadata remain covered; improper cavities, numerical float32 arithmetic, convergence and FE are N/A because this stage does not execute. Existing larger structural cases remain applicable.

## Future work

cdpf-gzix retains combined closure and publication. cff-7sw5 and its numerical prerequisites retain actual resolution, assembly and numerical binding. No qualification or supported-model change is proposed.

## Definitions

- D12: the accepted hybrid-regime-ec-v1 approximation and its explicit graph declaration.
- Region kind: the validated recurrence, dynamics or unary ownership declared for a factor.
- Requirement: an unresolved intention identifying a parameter source, destination and transformation.
- JIT/vmap: JAX compiled and batched execution.
- FE: free energy.



## Notes

**2026-09-15T07:59:52Z**

Root completed the bounded repair after recording its pre-edit architecture and hostile-input handoff. Requirement lowering now receives each already validated region and selects the recurrence recipe from region.kind. Factor/source traversal and all parameter keys, transforms and member roles are preserved. The canonical ownership documentation names this boundary. No test or test-selection change was needed; the unchanged single-classification-owner assertion now passes.

After the final source edit, uv run pytest -q tests/test_extension_ownership.py tests/test_hybrid_discovery.py tests/test_hybrid_declarations.py tests/test_hybrid_plans.py tests/test_topology.py tests/test_specialized.py tests/test_registered.py --junitxml=/tmp/cdhd-0ios-focused.xml passed 203 tests in 3.36 seconds. Log: /tmp/cdhd-0ios-focused.log. No numerical source, qualification artifact, execution guard or recovery attempt changed. Root commits and pauses for independent review before any new full suite.

Preserved failed combined gate: at 407299d, 2945 passed and one unchanged architectural ownership assertion failed; wall 1152.414 seconds under the 1200-second cap. Runtime history records the failure; complete log /tmp/cdpf-gzix-final-short-suite.log and phase summary /tmp/cdpf-gzix-first-gate-summary.json. No duplicate test calls. This was not a source-recovery failure. All three registered fits remain unused. After review, one fresh passing final gate is required because the source changed.

**2026-09-15T08:03:56Z**

Independent review of root-authored cdhd-0ios completed at clean source commit 7317bbe after root paused implementation. The previous unary metadata author served only as reviewer for this separate root repair after its own author/review cycle ended; a fresh reviewer spawn had hit thread capacity. No self-review is claimed. Accepted with no findings. Core validation already enforces declared region/factor pairing, so selecting the recurrence recipe from region.kind preserves unique ownership, traversal order and all parameter roles/keys/transforms. The unchanged ownership test now passes. Independent focused review: 24 passed in 0.29s; no edits or numerical program. Root notified the reviewer after completion.

The failed gate remains recorded; no interpretation or selection changed. Root will now run one fresh combined final short suite after this source repair under the unchanged 1200-second cap. Later ticket-only dependency refinement does not change the reviewed code. Close cdhd-0ios, cdhd-fyq6 and cdpf-gzix only after that gate passes. All three registered recovery fits remain unused.

**2026-09-15T08:26:36Z**

Combined final code gate passed at 8d71a976281fb94ffd290f3d7e722bd9dcca8ce2: uv run python tools/dashboard.py --run-suite completed with 2946 passed, zero failures and 16 warnings. Pytest time 1127.49 seconds; measured wall 1128.324 seconds (18m48.324s), leaving 71.676 seconds beneath the unchanged 1200-second cap. Four workers, unchanged not-slow selection and initially empty compilation cache. Root announced once and waited quietly for the full suggested 1200 seconds without polling. All 8838 phase records passed, 2946 distinct test calls, zero duplicates. Runtime history and ~/agent-share/cffg-tests.html are refreshed. Log /tmp/cdpf-gzix-repaired-short-suite.log; summary /tmp/cdpf-gzix-final-gate-summary.json. The earlier failed gate remains truthfully retained in runtime history and /tmp/cdpf-gzix-first-gate-summary.json.

Independent graph review/re-review and the separate unary/ownership repairs are accepted. The final gate includes the unchanged numerical-source integrity checks; no qualified numerical producer, frozen source inventory/specification or recovery artifact changed. No complete new qualification program or registered recovery fit was executed by this metadata work; all three registered fits remain unused. Root closes cdhd-fyq6, cdhd-0ios and cdpf-gzix in this resolving session. Follow-up ticket/evidence publication alone does not require another unchanged suite.

Next: publish this stack increment on PR152. cdhd-mfc9 is the next fresh sequential implementation ticket, defining general graph incidence/evaluator inputs; parent cdhd-hzw0 blocks real cff-7sw5 publication until corresponding numerical qualification is complete. The source-backed interface audit and dependency companion are in TK. Public compiler/bundle/runtime guards remain active; actual support/value resolution, final assembly, accepted runtime state and recovery remain downstream. No mathematical, declaration scope, approximation or threshold revision.
