---
id: cff-ejib
status: open
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
