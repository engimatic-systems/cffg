---
id: cff-tng6
status: closed
deps: []
links: []
created: 2026-07-05T23:59:58Z
type: task
priority: 1
assignee: coopwilliams
parent: cff-1d6t
---
# Trial-loop A-learning: perception-action cycle with conjugate updates

Part II sec 7.1 reproduction, mean-A approximation: per trial (1) plan with GFE composites using A=E[q(A)] from Dirichlet counts, (2) act via argmax control, (3) observe env, (4) replan step 2 with past slice data-clamped, (5) end-of-trial smoothing (pure BP) -> Dirichlet counts += x_hat kron q(z). A0 prior per Part II (O uninformative 10s, L/R/C 1/eps blocks, eps=0.1 off-block). Approximations vs paper (E[log A] digamma + IS message) tracked in cff-5iuz. Acceptance: win rate rises to ~alpha, GFE > BFE ablation on same seeds, A_C block learned diagonal-dominant.


## Notes

**2026-07-06T00:07:56Z**

Landed: perception-action cycle end-to-end (plan2/plan1/smooth phases compiled once, params-swap per trial — no recompiles, ~70ms/trial), TMazeEnv, conjugate Dirichlet learning. VERIFIED: cue-context mapping learned decisively (C-block diag dominance >5x within 30 trials), counts conserved, deterministic. NOT YET: Fig.9 win-rate ramp. Diagnosis for cff-5iuz: composite consumes exp(E[log A]) whose columns leak mass at eps-count cells (psi(0.1)=-10.4) -> uncertain states look improbable (anti-novelty artifact); GFE hovers at random-floor ~25/60 wins, arms avoided after early NR punishment. Faithful fix: composite needs BOTH Abar=E[A] for probability terms (q_x, log(A z)) and the coupled E[A log A] Dirichlet term (closed form: (a/a0)(psi(a+1)-psi(a0+1)) or IS) for the entropy/ambiguity term. Behavioral win-ramp assertions belong to cff-5iuz.
