---
id: cff-4sor
status: open
deps: [cff-k9ws]
links: []
created: 2026-08-10T02:58:17Z
type: task
priority: 2
assignee: coopwilliams
parent: cff-0f3g
tags: [rslds, hybrid, derivation]
---
# Specify tree-structured recurrence semantics and an independent oracle

## Background in Plain English

The current recurrence derivation assumes that destination `j` accepts stick
`j` after rejecting every earlier stick. A balanced tree instead assigns each
destination a path through internal binary gates. Replacing prefix sums with
paths without first defining orientation, labels, sufficient statistics, and
free-energy accounting would make a plausible-looking but unreviewable rule.

This ticket defines the complete mathematical contract before production IR
or runtime code changes. It extends the D8 derivation and supplies a small
independent oracle for both decision-list and balanced topologies.

## Outcomes / Acceptance Criteria

1. docs/DERIVATIONS.md defines one full binary transition tree with `K`
   categorical leaves, `K-1` internal gates, unique leaf labels, and an
   explicit accept/reject orientation on every branch.
2. The derivation defines the normalized
   `p(z_next | z_source, x, topology, coefficients)` for full source-specific
   affine logits and proves that every destination receives exactly one path.
3. The decision-list topology reduces algebraically to the existing D8
   sequential stick probabilities and coefficient layout.
4. The derivation gives the pathwise at-risk and signed-success statistics for
   every source, destination, and internal gate.
5. The exact Pólya–Gamma identity and coordinate-optimal
   Jaakkola–Jordan bound yield explicit messages to the destination regime,
   source regime, and Gaussian drive.
6. The derivation gives the gate-parameter natural statistics and states how
   several gate uses combine when they share one parameter owner.
7. The node-local energy includes the expected bounded log transition and the
   correct categorical and Gaussian entropy ownership. It states the bound
   direction after ordinary Bethe edge corrections.
8. P-substitution, max-product, and any GFE interpretation are explicitly
   unsupported unless separately derived.
9. A pure independent oracle enumerates paths and probabilities without using
   `cffg.recurrence` production functions.
10. Oracle tests cover decision lists, balanced trees, asymmetric valid trees,
    and at least two leaf-label permutations for `K=2`, `K=3`, and `K=4`.
11. The oracle matches the current decision-list probability and D8 statistic
    formulas within a stated dtype tolerance.
12. Exact-zero source or destination responsibility contributes exact zero to
    every affected statistic. Empty support and non-finite coefficients fail
    loudly rather than receiving a numeric floor.
13. Float32 oracle evaluation uses dtype-derived tolerances, and a balanced
    `K=32`, drive-dimension `16` case completes without a
    source-by-destination-by-gate tensor.
14. This ticket adds no production topology field, compiler plan, message
    dispatch, or public factor behavior.

## Methodological Suggestions

These suggestions are optional. The implementer may choose another sound
method that satisfies the outcomes.

- Represent each leaf path as internal-gate indices plus signs, then derive
  sufficient statistics by summing destination responsibility along paths.
- Use direct probability enumeration as the independent authority and only
  then derive a dynamic-programming implementation strategy.
- Apply the **New factor or rule family** change-path row only to enumerate the
  surfaces the derivation must constrain. Compiler, scheduler, ledger, and
  serialization changes are deliberately N/A in this ticket.

## Future work

cff-602p turns this contract into serialized source-of-truth topology while
preserving decision-list behavior. cff-lv65 owns production balanced-tree
messages and free energy. cff-4c7t owns persistent parameter sharing and
learning.

## Definitions

- **At-risk statistic:** Posterior mass whose destination path reaches a gate
  and therefore contributes a binary logistic trial at that gate.
- **Full binary tree:** A tree in which every internal node has exactly two
  children.
- **Jaakkola–Jordan bound:** A quadratic lower bound on a logistic log
  likelihood that becomes a Gaussian-form variational update.
- **Leaf-label permutation:** A reassignment of categorical destination labels
  to tree leaves without changing the unlabeled tree shape.
- **Parameter owner:** One unique coefficient posterior that may supply several
  factor-local gate uses.

## Notes

**2026-08-10T03:20:00Z**

Wiggle-audit retained candidates and intended meanings: n_a=production
surfaces explicitly excluded from this derivation ticket; preserve=retain
decision-list semantics; public=the excluded public runtime behavior;
implement=the suggested later dynamic program; support=literal categorical
support; complete=the full enumerated mathematical contract or completed scale
oracle; sufficient=pathwise Pólya–Gamma statistics; correct=the named entropy
ownership; exact=one path, the PG identity, literal zero, or the definition of
a full binary node; match=agreement with current D8 formulas; oracle=the pure
independent executable reference; sound=another method satisfying outcomes;
test=the listed oracle fixtures; learn=the cff-4c7t future parameter update;
trial=a binary gate trial; small=the explicitly bounded finite oracle;
expected=mathematical expectation; may=optional method or possible multiple
uses per owner; explicit_owner and named_ticket_handoff=the cff-602p,
cff-lv65, and cff-4c7t boundaries; any, at_least, every, and only=the named GFE
exclusion, permutation counts, tree components, statistics, and derivation-only
scope; current and existing=the checked-in decision-list D8 authority.
Capability owner: cff-4sor owns the derivation and independent oracle, not IR
or production execution. Intelligible-ticket review: complete.

Classification-note additions: check=the oracle or derivation comparison;
scale=the `K=32`, drive-dimension-16 oracle case; intended=the contextual
assignments in this note; later=the downstream production implementation.
