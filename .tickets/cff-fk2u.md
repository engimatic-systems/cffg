---
id: cff-fk2u
status: closed
deps: []
links: [cff-fxmj, cff-hp3y]
created: 2026-07-09T02:47:20Z
type: feature
priority: 1
assignee: coopwilliams
parent: cff-ahps
---
# Dynamics epistemics: derive an information-gain channel for transition learning

Close the gap recorded in LIMITATIONS S1c: learn_b (cff-cd1i) updates transition Dirichlets from experience, but the planner scores policies with the point estimate E[log B_u] — uncertainty about DYNAMICS generates no epistemic value, so the agent never acts to visit poorly-modeled transitions. This matters exactly when a world model must be built by intervention (the continual-learning use cases: an agent cannot pursue goals without learning how states evolve under and without its actions).

WHY THE GAP EXISTS (structural, not an oversight): all epistemic value in this engine flows through the P-substituted goal-OBSERVATION composite (Part II section 4-5). That construction leans on two properties of the observation node that the transition node lacks: (a) the substituted variable x is a FUTURE OBSERVABLE carrying a goal prior c, giving the substitution q(x) -> p(x|z) a well-defined target distribution; (b) the composite sits at a leaf, so its messages close in one Newton solve. The A-parameter novelty channel (E[A log A] pseudo-entropy under Dirichlet counts, rules._dirichlet_stats) rides on that same composite. transition_mixture is interior — both endpoints latent, no goal prior on either — so none of the derived messages apply as-is. Part II does not derive this node's P-substitution; it is genuinely new work.

TWO ROUTES, derive-then-decide (Fable at top level):
1. PRINCIPLED: P-substitution on transition_mixture with per-action Dirichlet q(B_u): substitute q(z')->p(z'|z,u) in the node-local expectation and work out the local GFE. The closed-form coupled term E[B log B] mirrors _dirichlet_stats exactly (that math transfers). The open part is the message structure: the epistemic term should reward visiting (z,u) pairs whose B-columns have low counts — expect a message to the SELECTOR u that scores expected count-novelty under the predicted pair marginal. Watch for the same fixed-point/stability issues the obs composite hit (cff-i45j lessons: hostile-inputs checklist applies — deterministic B columns are exactly the sparse-support regime).
2. PRAGMATIC BASELINE: an additive count-novelty bonus on the control message (the established active-inference B-novelty term: -Delta digamma weighted by predicted visitation, as in pymdp's pB info gain — vendoring policy in ARCHITECTURE allows reading it as a cross-check oracle). NOT free energy — if adopted even temporarily it must be a separately-labeled ledger term, never folded into BFE/GFE numbers.
Decision criterion: route 1 if the derivation closes in reasonable form; route 2 only as an explicitly-labeled interim with a test showing it reproduces route-1 behavior on a small case once route 1 lands. Validation: an agent in a slip-corridor with one UNKNOWN action's dynamics should preferentially exercise that action early (dynamics-curiosity), then exploit; ablation (channel off) shows no such preference.

DELEGATION: derivation and ledger semantics are Fable-only; Sonnet may fetch the exact pymdp pB-info-gain form and any post-2023 literature on transition-novelty in message passing (specific questions); Opus can build the slip-corridor validation harness once rule signatures exist.

## Acceptance Criteria

A written derivation (docstring + ticket comment) for the transition P-substitution or a documented decision that route 2 is the interim; dynamics-curiosity demonstrated on the slip-corridor with ablation; any interim bonus term separately labeled in the FE ledger; LIMITATIONS S1c updated to point here.


## Notes

**2026-07-09T03:23:30Z**

LANDED (2026-07-09, branch b-epistemics — merge of b-learning + gfe-stability — commits 8cd4fda, c61fe8e, 51b3530). Route 1 (principled P-substitution) closed in full; no interim bonus needed. All acceptance criteria met.

DERIVATION (full formal version: docs/DERIVATIONS.md D1; module docstring in rules.py): P-sub on transition_mixture with per-action Dirichlet q(B) yields local GFE G_tm = -I(z'; (z,u), B) - H[q(z,u)]. The MI splits into state-transition info + parameter novelty, and the novelty term is EXACTLY the expected KL of the next Dirichlet update (executable proof: test_novelty_identity). Stationarity gives Corollary 1's fixed point over the JOINT (z,u) simplex with mu_down in the goal prior's role — _gfe_solve/_gfe_rho reused verbatim; the obs composite is now a special case. Optimistic-bound semantics proved via data processing (z' latent), recorded in LIMITATIONS S1c.

BEHAVIOR: engine-level curiosity with exact-indifference ablation; 50-tick agent q(unknown action) starts at the analytic gap and decays toward indifference; notable finding — the ablation is ambiguity-AVERSE (digamma E[log B] penalizes low-count columns), so the composite corrects a bias, not just an absence.

GOTCHAS for successors: (1) hostile-inputs contract tests included (sparse/f32/scale) per AGENTS.md — keep them when refactoring; (2) joint solve is O((K.U)^3) per composite/sweep — structure-exploitation ticket if needed at colony scale; (3) the state-MI channel legitimately favors PREDICTABLE-and-distinctive actions — when testing novelty in isolation hold B-bar equal across actions (see test comments); (4) epistemic_b requires learn_b (counts are the novelty input); (5) models with the composite report GFE, not BFE.
