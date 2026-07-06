---
id: cff-tng6
status: in_progress
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

