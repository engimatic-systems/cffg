---
id: cff-iqqk
status: open
deps: [cff-epdp]
links: []
created: 2026-08-13T04:26:16Z
type: task
priority: 1
assignee: coopwilliams
parent: cff-uy48
tags: [paper-fidelity, agent, oracle]
---
# Compare a literal constraint-flip agent with the two-window runtime

## Background in Plain English

The papers describe one trial model whose future observation is governed by a
planning constraint and later becomes observed data. `RollingAgent` uses two
compiled templates instead: a forward planning window and a one-slice
filtering window. It carries the filtered boundary belief into the next
planning window. This is a sensible fixed-memory implementation, but the
repository has not directly shown where it is equivalent to the paper's
literal changing-constraint construction.

This ticket builds one small exact reference and compares the two
representations. The reference is intentionally smaller than the T-maze so
enumeration can separate a window-transformation error from a message-rule or
behavioral-benchmark error.

## Outcomes / Acceptance Criteria

1. `cff-epdp` identifies the paper source for the changing planning and data
   constraints and the repository's two-window representation.
2. A compact discrete reference model has at least two hidden states, two
   actions, an informative observation channel, a nonuniform prior, and two
   consecutive environment steps.
3. The paper-literal reference represents the same trial variables before and
   after observation.
   3.1. Before action selection, the future observation path carries the named
   planning and P-substitution constraints.
   3.2. After the environment returns data, the same logical observation is
   conditioned by the paper's data or point-mass construction.
   3.3. The reference snapshots record the exact constraint change rather than
   replacing it with an undocumented model rewrite.
4. An independent enumerator computes the planning action posterior, filtered
   state posterior, and any sufficient statistic carried across the boundary.
5. The two-window `RollingAgent` path agrees with the literal reference and
   enumerator on the first planning action posterior within a tolerance derived
   from dtype and enumeration conditioning.
6. After the first observation, the two-window filtering posterior agrees with
   the literal reference and enumerator under the same tolerance rule.
7. Re-anchoring the second planning window from that posterior produces the
   same second action posterior as the literal reference.
8. Factor-local objective terms agree wherever the two representations contain
   the same local construction. Any whole-graph scalar offset caused by a
   boundary factor is derived and checked explicitly rather than hidden by a
   broad free-energy parity assertion.
9. The test demonstrates that the two-window runtime compiles a bounded number
   of structures and does not grow its graph with episode length.
10. Literal zero support survives the constraint change and boundary rollover.
    A truly empty support returns infeasible instead of a normalized belief or
    action.
11. A vacuous boundary prior and a float32 version either agree with their
    independent references or return the documented truthful status.
12. `docs/ARCHITECTURE.md` and `docs/LIMITATIONS.md` state the validated
    equivalence boundary. They do not call two compiled windows the paper's
    literal single mutable graph.

## Methodological Suggestions

These suggestions are optional. An implementer may choose another sound method
that satisfies the numbered outcomes with equivalent evidence.

- Keep the reference model fixed-A and otherwise exact so this ticket isolates
  the trial-window transformation. Functional parameter messages are covered
  separately by `cff-in0q`.
- Build two immutable snapshots of the paper-literal graph if mutating a
  compiled template would violate serialization-as-source-of-truth. Verify the
  snapshots differ only by the declared constraint and data binding.
- Use the **New IR field** change path only if the reference exposes a missing
  representation. A test-only oracle should not add runtime mutation state.

## Future work

No further window-equivalence work is identified for the release gate. Models
with continuous, hybrid, or hierarchical boundary beliefs require separate
oracles before this result is generalized to those families.

## Definitions

- **Boundary belief:** The posterior state carried from filtering into the next
  planning window.
- **Literal model:** The paper-shaped trial graph in which the logical future
  observation changes constraint after data arrives.
- **Reference snapshot:** One immutable representation of the literal model at
  a named point in its trial lifecycle.
- **Window transformation:** Replacing the growing literal trial history with
  fixed-size planning and filtering templates plus an explicit boundary
  belief.
