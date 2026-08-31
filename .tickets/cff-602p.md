---
id: cff-602p
status: open
deps: [cff-4sor]
links: [cff-cz6v]
created: 2026-08-10T02:58:17Z
type: feature
priority: 2
assignee: coopwilliams
parent: cff-0f3g
tags: [rslds, hybrid, serialization]
---
# Serialize recurrence topology without changing decision-list results

## Background in Plain English

`recurrence_link` currently derives its decision-list topology from the factor
type and category count. A balanced tree cannot be reconstructed from the
existing template because the model does not store internal branches or leaf
labels. Serialization must become authoritative before the compiler executes
another topology.

This ticket adds explicit recurrence topology and migrates existing templates
to the same decision-list meaning. It does not execute balanced-tree messages;
cff-lv65 owns that numerical change.

## Outcomes / Acceptance Criteria

1. A recurrence topology specification records internal-node identities, two
   ordered children per internal node, leaf labels, and branch orientation as
   model structure rather than runtime data.
2. Validation accepts exactly one rooted acyclic full binary tree with `K`
   uniquely labeled leaves and `K-1` reachable internal gates.
3. Validation rejects cycles, disconnected nodes, duplicate or missing leaf
   labels, invalid children, multiple roots, invalid branch orientation, and a
   topology whose leaf count disagrees with the categorical edges.
4. Public constructors produce a canonical decision-list topology and a
   deterministic balanced topology for every integer `K >= 2`.
5. JSON and `.cffgz` round trips preserve the complete topology and its stable
   ordering.
6. The template structure hash covers every topology field and changes when a
   branch, orientation, or leaf label changes.
7. Current topology-free serialized models load with documented decision-list
   meaning and reserialize through an explicit schema migration.
8. The migration states whether the structure hash changes and gives bundles
   and checkpoints a loud compatibility rule rather than silently accepting a
   mismatched arena.
9. Composition, plate scoping, transactional edits, copying, and validation
   preserve or deliberately reject recurrence topology with a concrete
   diagnostic.
10. The compiler accepts explicit decision-list topology and rejects other
    valid topologies with a diagnostic that names cff-lv65 as the missing
    execution capability.
11. Explicit and migrated decision-list templates produce identical parameter
    stacks, messages, beliefs, run status, and Bethe free energy on compact and
    multi-slice fixtures.
12. Host/JIT, flooding/mixed schedule, and single-instance/`vmap` results stay
    unchanged because the executable topology remains the decision list.
13. Structural hostile cases fail before allocation. Sparse support, improper
    Gaussian beliefs, float32 numeric guards, and uncomfortable numerical
    factor scale are unchanged and covered by existing recurrence tests.
14. README.md documents topology construction. docs/ARCHITECTURE.md records the
    source-of-truth and migration boundary. docs/LIMITATIONS.md states that
    non-decision-list execution waits for cff-lv65.

## Methodological Suggestions

These suggestions are optional. The implementer may choose another sound
method that satisfies the outcomes.

- Use the **New IR field** change-path row. Review validation, JSON/hash,
  plates, compose/edit paths, compiler lowering, bundles, checkpoints,
  estimator input, and public construction.
- Canonicalize topology ordering during validation so hashes and compiler
  tables do not depend on dictionary insertion order.
- Keep legacy decision-list numerical calls on the existing production rule
  until cff-lv65 lands.

## Future work

cff-lv65 depends on this ticket and owns execution of balanced topology.
cff-4c7t later adds explicit coefficient ownership and learning. Destination
softmax is a separate factor family under cff-dlrj.

## Definitions

- **Branch orientation:** The declaration of which child corresponds to a
  logistic gate's accepted or rejected outcome.
- **Schema migration:** A deterministic conversion from an older serialized
  representation to the current representation.
- **Stable ordering:** A canonical order that produces the same serialization
  and hash for the same logical topology.
- **Structure hash:** The deterministic model identifier that covers graph
  structure but excludes runtime observations and parameter values.

## Notes

**2026-08-10T03:20:00Z**

Wiggle-audit retained candidates and intended meanings: canonical=the unique
constructor ordering; preserve=topology identity or existing decision-list
values; public=the topology constructors; same=logical topology or equal
serialized/runtime values as stated; support=the unchanged probability support
contract; complete=all topology fields; land=a prerequisite derivation already
present; remaining and later=balanced execution/learning delegated downstream;
estimate=unchanged resource-estimator numeric surface; exact=one structurally
valid full binary tree or bit/value identity; sound=another method satisfying
outcomes; stable=canonical serialization order; test=the existing recurrence
regressions; validate=structural validation or its named change-path review;
learn=the cff-4c7t future capability; scale=the unchanged hostile numerical
fixture; may=optional method; explicit_owner and named_ticket_handoff=cff-lv65
owns numerical balanced execution; any and every=each integer `K >= 2` and each
serialized topology field; current and existing=the pre-migration repository
schema/rule; later=ordered downstream work. Capability owner: cff-602p owns
topology structure, migration, and decision-list no-regression only.
Intelligible-ticket review: complete.

Classification-note additions: intended=the contextual meaning assignments in
this note; all=the complete topology-field set; only=the decision-list
execution scope before cff-lv65.
