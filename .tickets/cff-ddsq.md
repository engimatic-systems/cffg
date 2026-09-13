---
id: cff-ddsq
status: closed
deps: [cff-bfr9]
links: [cff-edoz, cff-ws3c, cff-sf29, cff-34gi]
created: 2026-09-02T04:17:42Z
type: chore
priority: 1
assignee: coopwilliams
parent: cff-edoz
tags: [compiler, architecture, refactor]
---
# Assemble immutable arenas through slim compiler orchestration

## Background in Plain English — WHY

At authoring snapshot e488df3, `compile_template` computes final support groups,
slot prefix offsets, delta/interior edge tables, semiring exclusions, and the
`Arena` constructor across `src/cffg/compiler.py:1819-1936`. That range is a
scope anchor, not a promised future line map. Earlier parts of the same
function also leave intermediate state for that constructor. The semiring
exclusions move to public option checking and the registered/specialized
compatibility owners; they are not an assembly responsibility.

The previous compiler tickets replace those intermediates with complete typed
values. This ticket gives final invariant checking and immutable assembly one
owner, leaving `compile_template` as readable sequencing only.

## Outcomes / Acceptance Criteria — WHAT

1. `cff-bfr9` is closed with `ResolvedCompileData` passing.
2. One assembly owner accepts `TopologyLayout`, `RegisteredPlanDrafts`,
   `SpecializedPlanDrafts`, `ResolvedCompileData`, and semiring; it returns one
   complete immutable `Arena` or a structured `CompileError`.
3. The assembler consumes resolved slot prefix arrays and specialized delta
   pairs, and constructs interior-edge ledger groups, final plan collections,
   and structure/fingerprint inputs without reclassifying a factor or
   reevaluating family-specific semiring compatibility. Prefix arrays are
   resolved by `cff-bfr9` before learning validation to preserve failure order.
   3.1. It mechanically joins raw topology, specialized semantic edge tags,
        drafts, and validated numeric payloads. No plan requires a missing
        blob-derived field before resolution.
   3.2. It verifies exactly one execution owner for each factor or absorbed
        member and the declared objective ownership/exclusions. A shared
        parameter contribution is accounted once, independent of its users.
4. `compile_template` performs only accepted-semiring option checking, plate
   expansion, the existing core `validate` call, and sequential calls to
   topology, specialized discovery, registered lowering, specialized
   completion, data resolution, and assembly. Discovery and completion use
   one specialized owner. Family-specific semiring compatibility stays with
   its registered or specialized owner.
5. All four architecture change-path rows are checked because the final value
   serves compiler, runtime, serialized-model, and learning consumers. The
   assembler introduces no new semantic interpretation for any row.
6. Flooding/scheduled, direct/`vmap`, and latest-schema/supported-legacy paths
   consume the same `Arena`. Host/JIT execution receives this one host-built
   value.
7. Sparse/vacuous/improper messages are runtime N/A. Exact support masks,
   float32 dtypes, and the 32-group fixture preserve the baseline offsets and
   plan order needed downstream.
8. Every `cff-ecw0` compiler characterization remains unchanged. Existing
   family oracle tests remain authoritative, while old/new compiler
   characterization protects this structural refactor. A newly noticed
   scientific evidence gap becomes a separate ticket and does not enlarge this
   ticket unless it proves assembly unsafe.
9. No factor, rule, approximation, objective term, schema, checkpoint field,
   status, or public compiler call form changes.
10. `compile_template` and its public compatibility surface are below 250
    physical lines; the whole public compiler orchestration module is below
    1,000 physical lines; no extracted file exceeds 1,000 lines.
11. This ticket fits one implementation context because all earlier stages are
    already complete and it owns only invariant validation plus construction of
    one immutable output.
12. Focused compiler/integration tests and the final short suite pass.
13. `docs/ARCHITECTURE.md` records the final sequence, inputs, outputs, and
    one-way imports in the assembly commit.

## Methodological Suggestions — HOW

These suggestions are optional. An implementer may choose another sound method.

- Make the assembler pure over complete typed values.
- Keep compatibility exports separate from assembly logic.
- Use field-by-field compiler characterization instead of a new opaque golden
  arena blob.

## Future work

Closing this ticket closes compiler track `cff-edoz`. Engine characterization
in `cff-sf29` then records the runtime baseline against these compiled owners.

## Definitions

- **Assembler:** The owner that checks cross-stage invariants and constructs
  the complete immutable arena.
- **Compiler orchestration:** Ordered calls to focused compiler stages without
  duplicating their interpretations.
- **Prefix array:** Cumulative offsets that map logical slots into flat arrays.

## Notes

**2026-09-09T21:53:23Z**

cff-bfr9 now publishes immutable slot/observation prefixes at the original final resolution checkpoint before learning validation. This small ownership correction preserves Python constructor-argument evaluation order from compiler.py:606-640 at701c10f; otherwise a later invalid learnable could hide an earlier prefix-overflow error. Assembly consumes these arrays without republishing numeric storage. Delta pairs already belong to specialized drafts. Acceptance criterion3 updated; no extra scientific or public contract change. Prerequisite --> dependent: cff-ejib (final gate running) --> cff-bfr9 (immediate blocker) --> cff-ddsq --> cff-edoz --> engine track --> cff-cfqg. No parallel prerequisite within this chain; tk records are authoritative.

**2026-09-09T22:57:05Z**

Preparation during bfr9 final gate at339c46a, not an implementation start. Remaining assembly is compiler.py:129-134 edge tagging plus142-199 Gaussian/EF interior tables and Arena construction. Inputs now include ResolvedCompileData with immutable prefixes, final ordinary plans, validated constraints, learning/member/observation data, and unchanged mu3/global-goal plans; do not load arrays, repack, recompute prefixes, or repeat learning. schedule._arena_signature already consumes Arena, so fingerprint inputs means preserve its fields/order, not add a second fingerprint schema. CompositeOwnership distinguishes executed and absorbed factors from parameter-owner references. Equality plans appear in resolved ordinary collections but belong to specialized ownership; count their execution once. mu3 and goal-parameter plans are overlays, not additional executed factors. Validate declared ownership/exclusion/reference consistency using one-pass indexes; keep family-specific interpretation in specialized owners. Negative tests should remove or duplicate an execution owner, mismatch an exclusion/parameter reference, and duplicate a shared-parameter definition, while positive mixed-goal and mu3 fixtures protect valid shared references. Reuse all published maps/arrays/plans; assemble only new edge tags and interior tables. Source: bounded local inspection of compiler.py, specialized/contracts.py and __init__.py, resolution/contracts.py, compiled.py and schedule.py at339c46a; no new tests or benchmarks. Extra preparation agents were unavailable due to usage limits; do not treat them as reviewers of this handoff.

**2026-09-09T22:57:05Z**

Dependency companion from tk dep tree --full cff-ddsq, tk ready, tk blocked; prerequisite --> dependent:
[x] cff-ejib (#119)
  --> [in_progress: final gate] cff-bfr9
    --> [blocked: next] cff-ddsq
      --> [blocked] cff-edoz
        --> [blocked] cff-sf29 --> engine track cff-ws3c --> cff-ofv1 --> cff-cfqg
No parallel prerequisite within this compiler chain; unrelated ready work remains outside this epic. Start from the published bfr9 branch only after its gate and closure, following the normal base update.

**2026-09-09T23:05:14Z**

Ready source baseline: cff-bfr9 is closed. Source 339c46a passed 1,777 default tests with 41 slow deselections and four existing warnings in 999.044 seconds measured wall; runtime record 5b3606f. All 22 new resolution cases and 125 final focused boundary tests pass; 14 complete arenas match 0bcda82. No source changed after the gate. Begin only after bfr9 PR publication, using agent/cff-bfr9-array-resolution as the intended base and updating it before branching. This ticket requires its own fresh final suite after implementation, not another run of the unchanged base.

**2026-09-09T23:07:24Z**

Published base: PR #120, https://github.com/engimatic-systems/cffg/pull/120, branch agent/cff-bfr9-array-resolution. It is stacked on #119 and has the passing final gate described above. Update this branch before creating the final-assembly work branch.

**2026-09-09T23:09:38Z**

Starting from updated PR #120 branch at 2779e5d; validated source is 339c46a (1,777 passing default tests). Change paths: New IR field, New factor/rule family, Learnable/checkpoint, and Runtime numerical/status all consume the assembled value; assembly adds no mathematical interpretation or status. Flooding/scheduled, direct/vmap, host/JIT runtime, and current/legacy reconstruction pairs share this Arena. Sparse/vacuous/improper runtime messages are N/A; preserve exact support/offsets, source float32 behavior, group order, and published storage reuse. Scope: move edge tagging and interior ledger construction into a pure typed assembler; verify complete execution/absorption ownership and declared objective/exclusion references without loading sources, calling kernels, redoing prefixes, or changing valid-model admission. Group names are not globally unique across family namespaces, so ownership checks must join original member identities rather than reject harmless cross-family name collisions. One fresh default suite after final implementation; focused contracts and field-by-field baseline comparison first.

**2026-09-09T23:13:19Z**

Mechanical extraction passes 141 focused assembly/resolution/registered/specialized/topology/compiler tests. assemble_arena now consumes only complete typed stage values, tags edges, forms Gaussian/EF interior tables, and constructs Arena. The facade delegates once and specialized.validate_admission preserves the existing early error checkpoint. Sixteen new assembly cases include all 14 characterization models with caller input disposal and blocked source/packer/prefix callbacks, delegation identity, and an accepted cross-family plan-name collision. The latter confirms that a constraint may share a name with a discrete group; upcoming ownership checks must use original-member identity. Architecture updated in the same extraction commit. Cross-stage ownership guards are the remaining implementation; no final default suite yet.

**2026-09-09T23:23:17Z**

Ownership checks are complete in assembly_checks.py. Registered/resolved plan counts, member/role/type order and published addresses agree; specialized execution/absorption records and ordinary plans cover each original factor exactly once. Checks join original members, preserving accepted plan-name collisions across families. Entropy exclusions, parameter-edge tags and constraint attachments agree with declarations. Dirichlet, functional mu3 and global-goal parameter-owner references match their contributors; learned-goal rows and carrier addresses are covered once, while valid shared references across shape groups remain accepted. Existing complete declarations suffice; no new family discovery or ownership metadata schema was needed. Prefix lengths are checked without recalculation. All 36 assembly cases pass; final broad focused run passed 182 tests with three slow deselections. Fourteen complete arenas and FrozenArray publication counts match pre-assembly source 339c46a. Public facade 138 lines; assembler 88; checks 294. Architecture updated. A separate documentation-only commit corrected the inherited discrete-only description of learning statistic sources. No inference mathematics, parameter data, schema or public call form changed.

**2026-09-09T23:23:17Z**

Bounded local performance check, no competing tests: 100 retained alternating repetitions after five warmups, assembly only over pre-resolved HMM inputs. Before/after ownership checks: 1 slice/4 factors, 0.032 -> 0.049 ms median; 32 slices/97 factors, 0.080 -> 0.133 ms; 128 slices/385 factors, 0.229 -> 0.389 ms. The baseline assembler came from 4e03323; current checks use one-time indexes and member lookups, with no per-owner whole-template copy. These are host setup costs on this machine, not inference timing or a speedup claim. Review evidence is local source inspection, 14-model parity/publication-count comparison, and explicit mismatch regressions. The additional review agents were unavailable due to usage limits; no independent agent review of this ticket is claimed. Final default gate follows at the completed source commit; no more implementation edits are planned.

**2026-09-09T23:42:33Z**

A final bounded source review confirmed an inherited malformed-input gap: fixed GFE A/c arrays with [-0.2, 1.2] can pass the sum-only check and pack to mass 1.2. This also reproduces in pre-extraction compiler 0bcda82. Separate P2 cff-34gi owns its fix; LIMITATIONS S9 records the current scope. It does not change valid-model parity or make the structural move unsafe, so it is not folded into this refactor or made an epic prerequisite. No numerical source was changed for this finding.

**2026-09-09T23:42:33Z**

Final gate passed: 1,813 tests, zero failures, 41 opt-in slow deselections, and four existing Gym deprecation warnings. Pytest: 1,003.86 seconds; measured wall: 1,016.282 seconds. Source: 707c31b; runtime record at documentation/ticket descendant 6364053, completed 2026-09-09T23:40:14.531582+00:00. Dashboard refreshed with 36 new cases and 30 diagrams in 23 galleries. No source/test change after the gate began. All assembly criteria are complete: typed pure join, exactly-once factor/objective checks, immutable publication reuse, slim facade, preserved public values and scientific behavior, one-way imports, explicit execution pairs, focused/full verification and updated architecture. Source work is finished. Publish on PR #120, close compiler track cff-edoz, and record its exact closing commit for engine characterization. No second unchanged suite is required for publication or epic closure.

**2026-09-09T23:45:38Z**

Published as PR #121: https://github.com/engimatic-systems/cffg/pull/121, stacked on #120. Compiler track cff-edoz is closed at 76251b44fc467a976aa056556f8fb371fe5de7f4, which is the exact engine-characterization baseline. Shared HTML now records the completed stages, 1,813-test final gate, host assembly cost, and separate P2 cff-34gi. Source is unchanged after 707c31b; later commits are documentation, ticket and publication metadata only. No second unchanged suite was run.
