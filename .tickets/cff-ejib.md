---
id: cff-ejib
status: open
deps: [cff-8t4d]
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

After `cff-8t4d` produces topology once, this ticket gives only registered
same-arena structural lowering a focused owner. It does not load blobs or
construct an `Arena`.

## Outcomes / Acceptance Criteria — WHAT

1. `cff-8t4d` is closed with `TopologyLayout` characterization passing.
2. One registered-lowering owner accepts `Template`, `TopologyLayout`,
   semiring, and the kernel registry and returns immutable
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

`cff-kfo7` extracts explicit specialized plan lowering. It must not route
multi-arena or structural behavior through this same-arena interface.

## Definitions

- **ArrayRequirement:** A typed request for a named, shaped parameter or
  observation value that a later stage resolves.
- **Plan draft:** Complete structural plan metadata before numeric arrays are
  loaded and the final arena is assembled.
- **Registered same-arena:** A kernel whose message-bearing edges share one
  discrete, Gaussian, or exponential-family representation.
