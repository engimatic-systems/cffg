---
id: cff-007q
status: closed
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


## Notes

**2026-07-05T23:09:13Z**

Landed gauge-fixed Newton (logit space, jacfwd, 25 iters, 1e-9 ridge) for z*=softmax(rho(z*)+log d); mu2=Cat(sigma(log z* - log d)); q_x=Az* written to obs-side x slot. Composite FE term -z*^T rho - H[z*] absorbs goal node + x-edge entropy (derivation verified: engine BFE == grid J_min at 1e-6). Tests: brute-force argmin parity 1e-4, ambiguity aversion direction + exact-0.5 BFE ablation, goal-seeking tilt, uninformative-A no-op, posterior predictive. NOTE for M4: single-node epistemic pull is inherently modest (q~0.52 in the 2x2 case); T-maze contrast comes from sequential structure — don't mistake small single-node tilts for a bug.
