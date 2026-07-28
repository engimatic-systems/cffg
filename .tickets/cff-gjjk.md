---
id: cff-gjjk
status: open
deps: [cff-007q, cff-fr5c]
links: [cff-x0sv]
created: 2026-07-05T23:03:40Z
type: task
priority: 2
assignee: coopwilliams
parent: cff-mwjo
---
# Goal hyperprior: mu1(c) = Dir(c|Az+1) (learned goals)

Part II section 7.2 goal-learning variant: expose the goal vector c as a
Dirichlet parameter edge on the observation-GFE composite. The composite
sends mu1(c) proportional to Dir(c | A z + 1), where A z is its predicted
observation mass, and the resulting q(c) becomes the next trial's prior.
This is the conjugate goal-parameter client of cff-fr5c; the message is
standard Dirichlet and is unrelated to the functional mu3(A) path.

## Design

Add the plain-English and formal goal-message derivation to DERIVATIONS,
including the constrained node/parameter-edge ledger terms and the exact
posterior-to-prior recursion boundary. A planning sweep may refine one trial's
q(c) but must not accumulate that message repeatedly as if sweeps were new
trials. Template serialization declares the parameter edge and prior; runtime
checkpointing carries the posterior hyperparameters.

Hostile-input plan: exact-zero predicted observation mass contributes zero
fractional count; a proper positive prior keeps q(c) proper; vacuous/improper
priors fail loudly; sparse float32 priors and 10-100x observation cardinality
remain finite without absolute guards.

## Acceptance Criteria

The mu1 Dirichlet message, q(c) posterior and complete constrained FE terms
match direct algebra on a tiny observation model, with a named executable
anchor in DERIVATIONS. One trial contributes A z exactly once regardless of
sweep count; posterior-to-prior recursion across trials and checkpoint/restore
are exact. Learned goals change the next trial's policy in the analytically
predicted direction. Serialization/hash round-trip, exact-zero mass,
improper-prior refusal, sparse float32 and uncomfortable observation scale
pass without fixed-goal regressions.
