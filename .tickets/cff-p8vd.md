---
id: cff-p8vd
status: open
deps: [cff-l6i5]
links: [cff-gjjk, cff-iwgl]
created: 2026-07-27T19:26:19Z
type: task
priority: 2
assignee: coopwilliams
parent: cff-mwjo
---
# D6 derivation: Dirichlet parameter-edge messages, FE terms, and conjugate equivalence

Docs-only deliverable: DERIVATIONS.md section D6. Every ticket on the parameter-edge thread needs message forms that do not exist yet. The dirichlet factor type has been registered in ir.py since M1 with zero rules behind it, and compiler.py:390 rejects FactorSpec.param_edges outright. Derive the math once, review it, then build on it.

Covers the whole thread so that cff-fr5c, cff-gjjk and cff-iwgl each start from a reviewed spec:

- BP and VMP messages on a Dirichlet parameter edge for the categorical, matrix_cat and transition_mixture factor types
- node energy and entropy terms, so the FE ledger stays exact when parameter uncertainty enters the graph
- the equivalence argument: in-graph q(A) equals the out-of-graph conjugate posterior when the model is conjugate and the mean-field beads match
- mu1 goal hyperprior, Dir(c|Az+1), from Part II section 7.2, including the posterior-to-prior recursion across trials (cff-gjjk)
- mu3 importance-sampled message, log mu3(A) = z^T xi(A), with the burn-in GFE averaging policy and where PRNG keys enter (cff-iwgl)
- natural-parameter slot layout for parameter edges, consistent with the EF contract in D2

Sequenced after cff-l6i5 so the slot layout is written against the refactored EF families rather than the pre-refactor arenas.

DELEGATION: top level only. Derivations are never delegated per the cff-ahps policy, and this one touches FE ledger semantics.

## Design

One derivation section rather than three, because the messages share a single parameter-edge construction and the FE accounting only makes sense stated together. Reviewing the math before any arena plumbing depends on it is the cheapest place to catch an error.

## Acceptance Criteria

DERIVATIONS.md gains a D6 section covering all six items above. Each message form is stated precisely enough to implement without further derivation. The conjugate-equivalence argument states its exact conditions and what breaks when they fail. No code changes in this PR.

