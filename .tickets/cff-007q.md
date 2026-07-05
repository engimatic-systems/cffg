---
id: cff-007q
status: open
deps: [cff-w3ki]
links: []
created: 2026-07-05T23:03:40Z
type: task
priority: 0
assignee: coopwilliams
parent: cff-mwjo
---
# GFE messages: Newton-solved backward state message + FE term

Part II Fig.5/Corollary 1: rho(z)=A^T(log c - log Az)-h(A); solve z*=softmax(rho(z*)+log d) by gauge-fixed Newton in logit space (jacfwd, fori_loop); mu2=Cat(softmax(log z*-log d)); q_x=Az*. Composite FE contribution -z*^T rho(z*) - H[z*] (derived: absorbs goal node term + x-edge entropy). Tests: brute-force 1-D grid argmin of local objective J(z)= -z^T(rho+log d)-H[z] matches Newton; ambiguity aversion with uniform c/d vs BFE-ablation giving exactly d; goal-seeking tilt with biased c; bfe==J_min.

