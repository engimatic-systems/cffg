---
id: cff-dfpi
status: open
deps: [cff-pksr, cff-hp3y]
links: [cff-hp3y]
created: 2026-07-08T16:50:10Z
type: task
priority: 1
assignee: coopwilliams
parent: cff-ahps
---
# Research: continuous/hybrid GFE epistemics (open problem)

The research half of the todo-note: Koudahl 2021 shows EFE in linear-Gaussian systems yields NO purposeful exploration — the continuous GFE story is unresolved in the literature, not merely underived here. Do not promise a curious Kalman agent. Candidate directions to experiment with once hybrid factors exist: epistemic term on a discrete regime variable inside a hybrid model (the switching-KF path — cheapest, most likely to work); nonlinear observation maps (info gain no longer state-independent); state-dependent noise (heteroscedastic emissions make precision itself informative). Deliverable is a decision + experiments, not necessarily shipped rules. DELEGATION: this is Fable-level research; Sonnet sub-agents fetch and summarize specific papers (Koudahl 2021 negative result details; any post-2023 continuous-GFE literature).

## Acceptance Criteria

Written position (ticket comment + LIMITATIONS update) on which continuous epistemic mechanism CFFG adopts, backed by at least one working hybrid experiment showing purposeful exploration.


## Notes

**2026-07-10T18:10:03Z**

ADVERSARIAL REVIEWER LITERATURE/POSITION UPDATE (2026-07-10). The ticket's 'open problem' wording is too broad. Primary source check: Koudahl/Kouw/de Vries 2021 proves standard additive-control linear-Gaussian EFE has constant epistemic terms, but explicitly reports multiplicative/switching controls can explore. Collis et al., Learning in Hybrid Active Inference Models, arXiv:2409.01066 (2024), learns rSLDS discrete abstractions, places a discrete AIF planner over a continuous controller, and says this lifts information-theoretic exploration into discrete space; this supports the ticket's regime-variable route, though it is not a CFFG continuous-GFE derivation. A fixed linear-Gaussian P-sub local objective is analytically tractable (Gaussian MI/logdet + goal cross-entropy); purposeful negative/positive controls should compare action-independent additive sensing against action-dependent C_u/R_u or switching dynamics. Recommendation: elevate a bounded research spike now, after cff-hp3y correctness: derive/test that exact subclass and the discrete-regime hybrid, then decide. Do not frame arbitrary nonlinear/mixture GFE as solvable: mixture entropy and general nonlinear MI require declared bounds/quadrature/MC. Sources: https://wmkouw.github.io/publication/efe-glds/ and https://arxiv.org/abs/2409.01066; confidence high on scope, medium on best integration until derivation.
