---
id: cff-0f3g
status: open
deps: [cff-9vab]
links: [cff-dlrj, cff-nkbj]
created: 2026-08-10T02:58:08Z
type: epic
priority: 2
assignee: coopwilliams
parent: cff-ahps
tags: [rslds, hybrid, factor-family]
---
# Epic: Tree-structured recurrent transitions

## Background in Plain English

CFFG's current `recurrence_link` implements one source-specific decision list.
Its rule uses sequential prefix sums, and its serialized factor does not name a
transition topology. The legacy rSLDS ecosystem also uses balanced binary
trees. Those trees can shorten gate paths and pool evidence differently, but
changing the tree changes categorical probabilities, messages, learning
statistics, and the local free-energy term.

This epic adds tree topology as explicit model structure and then extends the
existing Pólya–Gamma/Jaakkola–Jordan recurrence family in reviewable stages.
Decision-list behavior remains the compatibility authority. Balanced-tree
support does not block the immediate cff-nkbj decision-list comparison.

## Outcomes / Acceptance Criteria

1. cff-4sor is closed with a topology contract, complete local derivation, and
   independent executable oracle.
2. cff-602p is closed with explicit serialized topology and unchanged legacy
   decision-list inference results.
3. cff-lv65 is closed with fixed-parameter balanced-tree messages, status, and
   free-energy accounting across every affected execution surface.
4. cff-4c7t is closed with explicit gate ownership, correct statistic pooling,
   and continual learning for source-specific and shared parameters.
5. cff-9vab is closed with public rSLDS construction, cross-reference parity,
   end-to-end learning validation, and measured scale tradeoffs.
6. The complete capability keeps generative-model topology separate from
   variational constraints and derives every specialized message and ledger
   term from the declared model.
7. Existing topology-free `recurrence_link` models retain decision-list
   meaning through a documented schema migration.
8. Destination-softmax semantics remain outside this epic.

## Methodological Suggestions

These suggestions are optional. The implementer may choose another sound
method that satisfies the outcomes.

- Preserve one binary-gate algebra and use static topology tables to select
  which gates and branch orientations define each destination.
- Keep full source-specific coefficients as the default and express tying as
  one parameter identity with several consumers.
- Use exact enumeration on small trees before relying on a third-party
  implementation or large learned benchmark.

## Future work

cff-dlrj investigates destination-softmax recurrence as a separate factor
family. cff-nkbj remains the decision-list literature-comparison epic and does
not depend on this work. This epic is grouped under cff-ahps but does not
become a cff-ahps closure dependency unless that roadmap is explicitly
expanded in a later ticket change.

## Definitions

- **Balanced binary tree:** A binary tree whose longest root-to-leaf path grows
  logarithmically with its number of leaves.
- **Gate ownership:** The mapping from factor-local gate uses to the unique
  parameter variables whose posterior evidence they share.
- **Topology:** The internal gates, branches, paths, and leaf labels that map
  binary logistic decisions to categorical destinations.
- **Tree-structured recurrence:** A categorical transition whose gate logits
  depend on a continuous state and whose destinations are leaves of a binary
  tree.

## Notes

**2026-08-10T03:10:00Z**

Legend: prerequisite --> dependent

    [x] cff-k9ws
            |
            +--> [ ] cff-4sor (ready: semantics and oracle)
                          |
                          +--> [ ] cff-602p (serialized topology)
                                        |
                                        +--> [ ] cff-lv65 (balanced inference)
                                                      |
    [x] cff-4apw ------------------------------------+
    [x] cff-y4nx ------------------------------------+--> [ ] cff-4c7t
                                                                  |
                                                                  +--> [ ] cff-9vab
                                                                                |
                                                                                +--> [ ] cff-0f3g

    Linked, not required: [ ] cff-dlrj destination-softmax derivation
    Parallel, not blocked: [ ] cff-nkbj decision-list reference parity
    Roadmap parent, not a dependency edge: cff-ahps

The tk records are authoritative. The sequential chain follows the semantic
coupling: topology meaning precedes serialization; serialization precedes
runtime execution; correct execution precedes pooled learning; and those
capabilities precede end-to-end performance claims.

**2026-08-10T03:20:00Z**

Wiggle-audit retained candidates and intended meanings: end_to_end=the public
builder-through-learning benchmark; fixed=fixed supplied recurrence
coefficients; preserve=one shared binary-gate algebra; public=the documented
rSLDS constructor; implement=the existing recurrence rule or future production
code; support=balanced-tree execution; complete=the enumerated derivation or
epic capability; ready=tk readiness of cff-4sor; remaining and later=explicitly
deferred softmax or roadmap expansion; close=child/epic ticket closure;
correct=agreement with the derivation and owner-pooled statistics;
evidence=posterior observations or measured results; exact=finite enumeration;
oracle=the independent cff-4sor implementation; parity=the named reference
comparison; sound=another method satisfying outcomes; validate=the end-to-end
scientific check; infer and learn=fixed-belief execution and parameter updates;
performance and scale=the cff-9vab measured workloads; small=small enumerated
trees; can=possible shorter paths, not a promised speedup; may=optional method;
every=the named execution surfaces and specialized rules; current and
existing=the repository recurrence family before this epic. Capability owner:
the epic owns only the joint milestone; the five children own the ordered
mathematical, IR, runtime, learning, and benchmark layers. Intelligible-ticket
review: complete.

Classification-note additions: check=the named oracle or validation authority;
speed=whole-model runtime rather than path depth alone; intended=the contextual
assignments in this note; explicit_owner=the named child capability owner;
only=the epic's joint-milestone scope; deferred=softmax and uncommitted roadmap
expansion.
