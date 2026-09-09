---
id: cff-ecw0
status: open
deps: [cff-209l]
links: [cff-544x]
created: 2026-09-02T04:02:00Z
type: task
priority: 1
assignee: coopwilliams
parent: cff-edoz
tags: [compiler, architecture, characterization]
---
# Pin compiler contracts before extraction

## Background in Plain English — WHY

`cff-209l` moves ordinary discrete and Gaussian factors onto the registered
same-arena kernel contract. The next work will move compiler records and
lowering code out of a 2,368-line module. Before that move, reviewers need an
executable record of what the compiler produces and which failures it reports.

This ticket adds characterization evidence without moving production compiler
code. The evidence becomes the comparison base for `cff-544x` and
`cff-8t4d`. It records observable contracts, not incidental local variable
names.

## Outcomes / Acceptance Criteria — WHAT

1. `cff-209l` is closed with its discrete and Gaussian kernel migration and
   extension tests satisfied.
2. The characterization record names the exact commit that closes `cff-209l`
   and uses that commit as its compiler baseline. Where `cff-209l` promises
   public, serialized, or numerical parity, the record may also compare its
   outputs with authoring snapshot e488df3 as an external reference.
3. Compact fixtures cover these compiled paths:
   3.1. discrete HMM and fixed-A Figure 5 GFE;
   3.2. Gaussian linear and observed Gaussian models;
   3.3. Gaussian mixture and hybrid GFE;
   3.4. decision-list and structured recurrence;
   3.5. functional `mu3` metadata;
   3.6. hard constraints and rejected legacy moment matching;
   3.7. declared learnables and serialized AgentSpec bindings;
   3.8. a compact plate; and
   3.9. one external same-arena kernel.
4. The fixtures pin observable compiler results needed by consumers:
   4.1. compiled plan kinds and group membership;
   4.2. member, role, and parameter order;
   4.3. message, Gaussian, exponential-family, observation, and parameter
        offsets;
   4.4. learnable source and target addresses;
   4.5. schedule kind and fingerprint;
   4.6. template structure hash; and
   4.7. functional evaluator identity where present.
5. Direct failure fixtures pin diagnostics for an unsupported factor,
   constraint, role, parameter schema, observation shape, learnable pairing,
   evaluator name, and missing external implementation.
6. A compact ownership table records which compiler block at the named
   baseline owns topology, registered lowering, specialized lowering,
   learning, array resolution, and final assembly. It identifies repeated
   interpretations that later tickets must remove.
7. Deliberate execution pairs are handled explicitly.
   7.1. Flooding and scheduled runtimes consume the same compiled addresses.
   7.2. Direct and `vmap` runtimes consume the same compiled plan values.
   7.3. Latest-schema and supported legacy checkpoints do not compile a second
        model meaning; their applicable compatibility tests reconstruct the
        structure declared by the baseline template.
   7.4. Host/JIT compiler duplication is not applicable because compilation is
        host-side and produces static values consumed by JIT.
8. Hostile-input planning is explicit.
   8.1. Sparse support, vacuous messages, and improper beliefs are runtime
        values and are not interpreted by compiler lowering.
   8.2. Float32 versus float64 may change derived array dtypes but not plan
        membership, address order, or structure identity.
   8.3. An uncomfortable fixture includes at least 32 groups and verifies
        deterministic plan and address construction.
9. No production compiler behavior, serialized schema, message rule, objective
   term, or runtime status changes in this ticket.
10. The focused compiler, serialization, schedule, learning, external-kernel,
    and representative model tests pass. The dashboard-backed short suite
    passes after the final test change.
11. The ticket fits one implementation context because it inventories existing
    tests and adds only compact compiler-output/failure characterization; it
    moves no production code and starts no new mathematical validation.

## Methodological Suggestions — HOW

These suggestions are optional. An implementer may choose another sound method.

- Extend existing focused tests instead of creating a second compiler snapshot
  format.
- Compare structured values field by field. Avoid one opaque golden blob that
  is hard to review.
- Use the **New factor or rule family**, **New IR field**, and **Learnable or
  checkpoint feature** rows in `docs/ARCHITECTURE.md`.
- Keep fixtures small enough that a reviewer can understand each expected
  address and order.

## Future work

`cff-544x` moves compiled plan and arena values only after this evidence
passes. No other future work is identified.

## Definitions

- **Address:** A stable compiled location for a message, parameter,
  observation, or learning statistic.
- **Characterization test:** A test that records observable behavior at the
  exact cff-209l completion commit before the compiler refactor.
- **Compiler contract:** The compiled plans, addresses, order, identity, and
  diagnostics visible to repository consumers.
- **Static value:** Host-created metadata captured by a compiled JAX function.
