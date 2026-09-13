---
id: cff-ejib
status: closed
deps: [cff-kfo7]
links: [cff-kfo7]
created: 2026-09-02T04:17:42Z
type: chore
priority: 1
assignee: coopwilliams
parent: cff-edoz
tags: [compiler, architecture, refactor]
---
# Extract registered same-arena compiler lowering

## Background in Plain English — WHY

At authoring snapshot e488df3, `src/cffg/compiler.py:1370-1742` is the source-
region anchor where ordinary discrete, Gaussian, and exponential-family group
lowering is interleaved with specialized GFE, hybrid, recurrence, precision,
and constraint handling. The implementation starts from the exact `cff-ecw0`
baseline, after `cff-209l` makes ordinary families use registered same-arena
kernel declarations; the old line range is not a promised future map.

After `cff-8t4d` produces topology and `cff-kfo7` identifies specialized
ownership, this ticket gives only registered
same-arena structural lowering a focused owner. It does not load blobs or
construct an `Arena`.

## Outcomes / Acceptance Criteria — WHAT

1. `cff-kfo7` is closed with specialized discovery and completion extracted.
2. One registered-lowering owner accepts `Template`, `TopologyLayout`,
   `SpecializedClaims`, semiring, and the kernel registry and returns immutable
   `RegisteredPlanDrafts` plus typed `ArrayRequirement` values.
3. The output records group identity, members, roles, family, message mode,
   gather/scatter addresses, observation intent, parameter roles, energy and
   readout identities, and serialized implementation identity. It contains no
   loaded parameter array and no partially initialized `Arena`.
4. Ordinary discrete, Gaussian, and exponential-family factors select this
   path from registered metadata. Their type names are not reclassified in a
   second shared compiler block.
   4.1. Neither `TopologyLayout` nor final assembly chooses registered family
        meaning; this ticket is the sole registered classification/lowering
        owner.
   4.2. Factors claimed by a composite, including absorbed goals and priors,
        never enter ordinary groups. This owner consumes claims without
        rediscovering their GFE or parameter-edge meaning.
   4.3. Completed registered membership is available to the specialized owner
        for recurrence-component construction; it does not change prior claims.
5. The affected change-path row is **New factor or rule family**. **New IR
   field** applies only to roles copied from `TopologyLayout`; **Learnable or
   checkpoint feature** applies only to emitted address requirements;
   **Runtime numerical or status invariant** is N/A for host-side lowering.
6. Flooding/scheduled and direct/`vmap` paths consume the same drafts after
   assembly. Host/JIT is N/A. Latest-schema/supported-legacy reconstruction
   reaches this one lowering owner.
7. Sparse/vacuous/improper messages are N/A. Float32 does not alter plan
   membership or addresses. The 32-group fixture retains deterministic plan
   ordering.
8. Existing registered-kernel, HMM, Gaussian, EF, observation, serialization,
   and compile-failure tests protect old/new observable values. The independent
   cff-209l test-factor oracles remain authoritative; no other oracle is added.
9. Unsupported registration metadata fails with the factor and incompatible
   declaration. No rule, objective term, public API, schema, or factor family
   changes.
10. The registered lowerer is below 1,000 physical lines and contains only the
    same-arena registered path, fitting one implementation context.
11. Focused compiler/kernel/family tests and the final short suite pass.
12. `docs/ARCHITECTURE.md` names the registered-lowering input/output boundary
    in the extraction commit.

## Methodological Suggestions — HOW

These suggestions are optional. An implementer may choose another sound method.

- Move structural grouping and address construction before parameter packing.
- Express parameter needs as typed requirements consumed by `cff-bfr9`.
- Keep compiler adapters thin while internal consumers move to the draft type.

## Future work

`cff-bfr9` resolves declared data requirements after both kinds of draft exist.
The specialized completion call remains owned by `cff-kfo7`.

## Definitions

- **ArrayRequirement:** A typed request for a named, shaped parameter or
  observation value that a later stage resolves.
- **Plan draft:** Complete structural plan metadata before numeric arrays are
  loaded and the final arena is assembled.
- **SpecializedClaims:** Explicit ownership of factors and absorbed members
  by specialized plans, plus their parameter-edge semantic tags.
- **Registered same-arena:** A kernel whose message-bearing edges share one
  discrete, Gaussian, or exponential-family representation.

## Notes

**2026-09-09T21:02:15Z**

Preparation during cff-kfo7 final gate at f281a29; no registered-stage implementation started. Smallest output: immutable discrete/Gaussian/EF drafts, ordered group references (including opaque equality references), BP membership, conditioning diagnostics and deferred grouping/per-group diagnostics. Drafts retain frozen kernel metadata, structural group fields, parameter requirements and explicit observation intent. Gaussian/EF observation intent carries role/index and member-ordered factor/port/variable/dimension-or-EF-tag references, not obs_rows, gauss_data/ef_data IDs or prefixes; preserve observed-with-unresolved-rows versus unobserved. Those row/member addresses stay compatibility-resolver-owned until cff-bfr9. Extend the common array requirement contract for registered kernel metadata; do not eagerly evaluate external ParameterSpec.shape callbacks, which can raise before the existing initializer/packing gate. Retain the provider plus structural role sizes. Registered lowering replaces _ordinary_grouping and ordinary gather/scatter construction, feeds bp_members to specialized completion, then existing numeric/checkpoint order remains. Equality routes from SpecializedClaims are authoritative: registered output carries opaque group-key references and counts discrete equality membership for recurrence; specialized completion supplies the actual equality plans. Sort original group keys, not displayed names (numeric dimensions sort differently). Preserve non-equality family=any domain routing.

**2026-09-09T21:02:15Z**

Question: which registered-lowering observables require deferred checks? Answer: conditioning problems remain before parameter-prior problems but after blob loading; remaining-grouping errors remain after GFE/hybrid packing. Gaussian name collisions, parameter paths, callback/shape checks and observation-row allocation retain sorted group order. EF keeps its legacy callback/key-check behavior and independent observation counter; do not upgrade it to discrete/Gaussian packing semantics during extraction. Discrete missing bindings, semiring restrictions, packed-shape checks and writable-member decisions remain at their current packing gate. Sources: bounded modularity-agent inspection of compiler.py:113,295,462-652 at f281a29, using prior extraction review context. Confidence high. No numeric packer, observation/member resolver, learning resolver or public compiled-record redesign is required. Correctness and ownership baseline includes cff-n183/cff-gvch and new specialized tests; cff-kfo7 must pass its final gate before this ticket starts.

**2026-09-09T21:18:43Z**

Additional boundary clarification: registered binding-capability interpretation (constraint_modes, EF capability, conditioned-output eligibility) should live in small registered-owned helpers, even when specialized discovery invokes them. Keep discovery original per-factor traversal and exact diagnostic append points; do not delay all ordinary failures until registered grouping or collect them in a separate pass. Capture one immutable registry mapping before discovery, then share it with admission and registered lowering instead of consulting the live registry repeatedly. Helpers depend only on declarations/bindings, not SpecializedClaims, avoiding an import cycle; later lowering separately consumes specialized exclusions. Add a regression combining an earlier ordinary-capability failure with a later specialized failure and pin the entire ordered diagnostic list. Question -> answer from bounded modularity-agent review of the existing admission traversal at 9a130c0; confidence high. This refines ownership without a generalized diagnostics framework or semantic change.

**2026-09-09T21:26:45Z**

Ready baseline update: cff-kfo7 is closed; final code tree 9a130c0 (implementation f281a29 plus callback-timing correction dba0429) passed 1736 default-suite tests, 41 slow deselections, four existing Gym warnings, 998.32 seconds wall. Preserve the now explicit specialized ownership and all 27 new stage-boundary tests, especially same-FactorType replacement shape-callback timing. No user decision is pending before registered extraction. Start from the published specialized-lowering branch after its PR publication and normal base update.

**2026-09-09T21:28:21Z**

Starting implementation on agent/cff-ejib-registered-lowering from updated agent/cff-kfo7-specialized-lowering at bbf3036 (PR #118); validated source baseline 9a130c0. Change-path rows: New factor/rule is primary; New IR field only copies existing topology role declarations; Learnable/checkpoint only emits existing address requirements. No new schema or runtime numerical/status change. Flood/scheduled and direct/vmap consume the same assembled plans; host/JIT compiler pair N/A; JSON/current/legacy bundles reconstruct through one owner, agent runtime checkpoint data remains unchanged. Capture a shared immutable binding snapshot, preserve per-factor admission ordering through registered-owned capability helpers, then produce registered structural drafts/observation intent/typed array requirements before specialized completion. Equality remains specialized: carry opaque ordered references and discrete BP membership only. Do not invoke shape, packing or observation-validation callbacks early; no observation/member IDs or partially initialized Arena are published. Numeric packing, observation/member address resolution, learning and assembly remain compatibility-owned until their tickets. Sparse/improper messages N/A; float32 metadata and 32-group ordering parity required. Preserve all existing gates, error classes, binding fallback and legacy EF packing behavior. Focused tests throughout; one final dashboard-backed default suite after the final source change.

**2026-09-09T21:47:49Z**

Registered drafts implemented; final validation pending. Registry capability interpretation now lives in registered/capabilities.py, shared snapshot passed to original per-factor admission and lower_registered. Draft streams preserve original group-key ordering, opaque specialized equality references and recurrence BP membership. Gaussian/EF observation intent is separate from row allocation; immutable gather/scatter storage is reused. Common ArrayRequirement types live in array_requirements.py; external shape providers are unevaluated, with one immutable role-size map shared across a member’s parameter requirements. Numeric packer mathematics, row/member/learning resolution and assembly remain compiler-owned. Tests: 18 new boundary cases pass and 14 complete arenas match an isolated pre-extraction compiler from 753d287; broad focused run and final gate pending. Review question -> answer: narrowing registered packer inputs breaks existing callbacks that read other declared factor blobs, so all three registered paths retain the full resolved mapping. Eager port lookup for an invalid any-family EF observation preempts missing-blob/conditioning diagnostics, so an absent port remains explicit pending intent until the original diagnostic gate. Both reproduced against baseline by modularity agent and protected by regressions. Performance-agent source review confirms immutable array identity preservation; shared role-size maps avoid repeated O(parameter-count × role-count) storage. Sources: compiler.py and registered/ versus 753d287 (same source as 9a130c0); confidence high. No rule/schema or supported-case expansion.

**2026-09-09T21:49:11Z**

Final review refinement: a malformed family=any EF observation whose output is discrete has no earlier conditioning diagnostic. Preserve its original late KeyError(variable) at observation-row resolution instead of publishing ef_data[None]. The optional source-port contract now covers both early conditioning rejection and late missing-port rejection; the regression is parameterized over Gaussian/discrete malformed output domains. Broad focused run before this final guard: 224 passed in 79.76 seconds. New stage file now has 19 tests; final focused check/gate follows.

**2026-09-09T22:07:02Z**

Final gate passed: 1,755 tests, zero failures, 41 opt-in slow deselections, four existing Gym deprecation warnings. Pytest took 992.95 seconds; measured wall time was 1,005.304 seconds. The dashboard includes 19 new cases, 30 diagrams and 23 galleries.

Source commit: 701c10f. The run began at 758d43e; runtime history records 2c29d5c after ticket-only handoffs. Source, tests, dashboard code and ARCHITECTURE are identical across these heads. The final focused check passed 90 tests in 3.03 seconds after the optional-port guard; the earlier broad check passed 224 in 79.76 seconds. All 14 complete baseline arenas match. Changed source/tests pass Ruff; diff/dependency checks pass.

Closing cff-ejib: the registered lowerer is 335 lines; compiler.py shrank from 1,122 to 1,024 lines. Next, cff-bfr9 owns source, numeric, address and learning resolution. Its handoff preserves full-map callbacks, original failure gates, named family numeric owners and prefix conversion before learning. No significant user decision remains before that extraction. PR publication follows; an unchanged second suite is not required.

**2026-09-09T22:08:44Z**

Published PR #119: https://github.com/engimatic-systems/cffg/pull/119, based on specialized-lowering PR #118. The shared architecture HTML and test dashboard show the completed 1,755-test gate. Later commits contain only runtime evidence, ticket handoffs and ticket readability improvements; implementation source remains 701c10f. cff-bfr9 is ready, with the agreed validation-order boundary correction already recorded.
