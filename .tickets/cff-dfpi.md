---
id: cff-dfpi
status: closed
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

**2026-07-13T02:48:20Z**

RESOLVED (2026-07-12, Fable research spike; branch spike/hybrid-gfe-epistemics). DECISION: CFFG adopts the discrete-regime epistemic mechanism — P-substitution composites on hybrid mixture nodes; NO Gaussian-only GFE message family will be added, and that is now a pinned theorem rather than a literature citation. Full derivation: docs/DERIVATIONS.md D5 (Props 1-6); position recorded in LIMITATIONS S1 + new S1h. Key results: (1) the substituted hybrid objective splits into an exact per-regime state channel I(x;z|s)=half-logdet ratio, an exact Gaussian-goal cross-entropy (mixtures against quadratic log-densities stay closed form), and a regime channel I(x;s) whose only intractable piece is the mixture entropy — scored with the D3 moment-matched collapse, a declared optimistic bound with a two-sided closed-form sandwich (Hershey-Olsen lower, collapse/H(rho) upper), directions pinned by quadrature. (2) Koudahl 2021's negative result re-derived inside our objective (D5 Prop 5): additive-control LGSSM epistemic terms are policy-independent — measured equal to 1e-10 across random policies through the general code path. (3) Stationarity over the regime simplex has exactly the D1/Corollary-1 fixed-point shape (K-logit Newton; existence by Brouwer); selector message = log-potential with own incoming dropped; ledger term = direct Ghat (cff-hp3y lesson respected). (4) Working experiment (acceptance criterion): probe-then-commit hybrid plant, probe strictly risk-dominated — GFE scorer probes 100%, succeeds 100%, post-probe regime channel <1e-5 (satiation); KL-control ablation never probes, 44.5% success. Formulas: src/cffg/experiments/hybrid_epistemics.py; 23 pinning tests in tests/test_hybrid_epistemics.py (incl. float32 + hostile inputs). Bonus: closed-form Gaussian parameter-novelty channel (D5 Prop 6, mirror of D1 Prop 3) recorded for the implementation ticket. Gotchas for the implementer: the surrogate exceeds the H(rho) cap on well-separated regimes (1.86 vs 0.69 nats measured) — keep the smooth surrogate in the solve, report the sandwich in diagnostics; v1 keeps continuous edges on D3 BP/collapse messages (coupled stationarity would reward wishful component separation — excluded until derived). Engine integration: see follow-up ticket.
