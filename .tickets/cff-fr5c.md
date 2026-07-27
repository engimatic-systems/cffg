---
id: cff-fr5c
status: open
deps: [cff-x0sv, cff-p8vd]
links: [cff-iwgl]
created: 2026-07-08T22:00:30Z
type: feature
priority: 3
assignee: coopwilliams
---
# Compile param_edges: in-graph parameter posteriors (Dirichlet nodes as factors)

GPT-5.5 feedback item 3. PARTIAL DISAGREE on urgency, recorded here: out-of-graph conjugate updates (B-learning + generic API tickets) deliver the same posteriors for the conjugate cases at a fraction of the machinery, so this is the architectural end-state, not the fast path. What in-graph buys: (a) parameter posteriors participate in message passing WITHIN a trial (joint state-parameter smoothing), (b) parameter uncertainty appears in the FE ledger exactly, (c) the papers' mu3 IS message to q(A) (cff-iwgl) finally has its home — this ticket is the compiler/IR half of that thread. Scope: compiler accepts FactorSpec.param_edges (currently rejected, compiler.py 'learned parameters are separate tickets'); dirichlet factor type gets BP/VMP rules + node energy (registered in ir.py since M1, zero rules exist); arena treats parameter edges as edges with Dirichlet-message slots (natural-parameter arena — coordinate with the EF abstraction cff-8s6a, which may subsume the slot design). Validation: within-trial A-posterior matches the out-of-graph conjugate update on a conjugate model (they must agree exactly when the model is conjugate and mean-field beads match). DELEGATION: Fable derives Dirichlet-edge messages + FE terms and the equivalence argument; Opus does arena plumbing after the rules exist.

## Acceptance Criteria

Conjugate model: in-graph q(A) == out-of-graph conjugate posterior to tolerance; FE ledger includes parameter-edge terms; cff-iwgl unblocked (mu3 has a target arena).

