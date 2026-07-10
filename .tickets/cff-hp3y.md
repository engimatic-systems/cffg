---
id: cff-hp3y
status: open
deps: []
links: [cff-fxmj, cff-fk2u, cff-4oq8, cff-gul8, cff-cj5h, cff-dfpi]
created: 2026-07-10T17:56:31Z
type: bug
priority: 0
assignee: coopwilliams
parent: cff-ahps
tags: [adversarial-reviewer, math, gfe]
---
# GFE-TM violates cavity consistency and scores a Lagrangian coupling as node energy

ADVERSARIAL REVIEWER FINDING (2026-07-10). The novel transition P-substitution extension in rules.gfe_tm_update is mathematically inconsistent at the substituted out edge. It writes to_out=log(q_w), but the facing downstream factor still sends mu_down, so the edge marginal becomes normalize(q_w * exp(mu_down)) rather than the required q_w. The outgoing cavity message must be normalize(log(q_w)-mu_down), with exact-zero support handled safely. Separately, gfe_tm_energy returns -qj*rho-H(qj), where rho contains Abar^T mu_down; algebraically this is G_tm - E_qw[mu_down], not the node-local G_tm from DERIVATIONS D1 Proposition 1. Since the downstream node and edge are also scored, this folds a Lagrangian message coupling into the FE ledger and double-counts downstream evidence/preferences. Two-state repro: q_w=[0.679691,0.320309], current edge belief=[0.894603,0.105397] (max error .214912); reported energy=-.094125 vs local G_tm=-.761311, difference .667186 exactly -E_qw[mu_down]. Practical impact: multi-slice epistemic_b planning propagates downstream preferences twice through substituted transition edges; next-state marginals, upstream policy messages, reported GFE, convergence traces based on it, and novelty-vs-goal tradeoffs are not trustworthy. Papers I/II do not derive transition P-substitution, so LAIF parity does not validate this extension. Hostile-input plan: exact-zero q_w/mu_down support requires masked cavity subtraction; vacuous mu_down must reduce to current output; float32 and KxU uncomfortable scale must remain finite; non-finite local energy is divergence.

## Acceptance Criteria

1. Add a one-composite edge-consistency test with nonuniform downstream message proving softmax(to_out+mu_down)==q_w; include exact-zero support and float32 variants. 2. Compute G_tm directly as sum(q_w log q_w)+qj dot h+sum(qj log qj), independent of incoming-message coupling except through the stationary belief; test against direct enumeration/algebra, not rho restatement. 3. Add a two-composite chain test against a direct constrained-objective optimizer or independent stationary equations, checking marginals, messages, and total GFE. 4. Re-run Hallway epistemic_b and replace prior claims/tickets/docs if behavior changes; RockSample-style observation-GFE balance belongs to cff-4oq8, not this defect. 5. Correct DERIVATIONS D1 Proposition 4, rules docstrings, LIMITATIONS, and any tests that currently call the Lagrangian stationary value the FE contribution. 6. Full hostile-input contract passes.
