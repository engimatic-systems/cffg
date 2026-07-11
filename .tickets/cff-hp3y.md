---
id: cff-hp3y
status: closed
deps: []
links: [cff-fxmj, cff-fk2u, cff-4oq8, cff-gul8, cff-cj5h, cff-dfpi, cff-ey4p]
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

## Notes

**2026-07-11T04:00:18Z**

Hostile-input plan before implementation: exact-zero q_w support remains -inf in the cavity; 0/0 support (q_w=0 and mu_down=-inf) uses a neutral finite cavity because the facing -inf alone enforces zero edge mass; positive q_w against mu_down=-inf is infeasible and must fail loudly rather than manufacture consistency. Vacuous finite-uniform mu_down cancels under normalization and reproduces q_w/current substituted output. The rule and direct node-local energy must work in float32 using dtype-derived guards only. Tests will exercise a KxU arena 10–100x the two-state fixture and require finite values on supported states. Any non-finite local GFE is propagated for the runtime divergence work in cff-j4mu to classify; this ticket will not conceal it.

**2026-07-11T04:03:46Z**

SUB-AGENT RESEARCH (math_audit, 2026-07-10). Q: What cavity/ledger/chain oracle does the review justify? A: Require normalize(to_out+mu_down)=q_w, so to_out=log(q_w)-mu_down up to gauge, masking q_w=0 to NEG and treating q_w>0 against zero-support mu_down as infeasible. Score only G_tm=sum(q_w log q_w)+qj·h+sum(qj log qj); mu_down belongs to the Lagrangian coupling, not node energy. Independent two-composite oracle minimizes G1(q1)+G2(q2)+H(r)-boundary-message couplings subject to B1q1=r=sum_u q2; compare q1/q2, shared and terminal marginals, cavity messages up to gauge, node terms, and full GFE. Sources: docs/DERIVATIONS.md D1 Props 1/4 and notes/adversarial-review-2026-07-10/07-math-audit.txt transition-GFE section. Confidence: high on cavity and ledger, medium-high on full-ledger normalization until pinned against engine unary conventions.

**2026-07-11T04:55:29Z**

IMPLEMENTATION/HALLWAY HANDOFF (2026-07-10). Refinement to the initial hostile plan: on 0/0 support the cavity itself is NEG (not neutral finite), so q_w=0 can never be resurrected; positive q_w against exp(mu_down)=0 yields NaN loudly. Landed rule behavior: to_out is the masked predictive cavity, and gfe_tm_energy evaluates only direct node-local G_tm. Tests: nonuniform/vacuous/zero/f32/infeasible one-composite cases; direct algebra; K*U=120 scale; independent SLSQP two-composite chain checking both joint beliefs, five edge marginals, all six messages, both node terms, and total GFE. Hallway rerun used the published L=8, H=10, 120-sweep, seed-0, 25-episode setting: known model 1.00; corrected curiosity 0.16 success, median successful steps 10.5, visitation-weighted B KL 2.650, count mass 5800; learn-only/no-curiosity 0.76, 7.0, 2.534, 5490. The invalid pre-fix 0.40/0.80 and marginal-KL-benefit claims were replaced in cff-2gev, LIMITATIONS S1c, and ARCHITECTURE. RockSample claims were not changed (observation GFE; cff-4oq8). Separate discovery cff-ey4p tracks RollingAgent literal 1e-300 underflow. Verification before final dashboard: targeted 15 passed; full repo 1,095 passed with 4 dependency warnings.
