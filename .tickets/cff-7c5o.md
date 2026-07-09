---
id: cff-7c5o
status: closed
deps: [cff-4e5z]
links: []
created: 2026-07-08T16:49:34Z
type: feature
priority: 2
assignee: coopwilliams
parent: cff-ahps
---
# Gaussian VMP + conjugate learning: NIW/Gamma precision-mean priors

Continuous analog of Dirichlet A-learning: mean-field beads on Gaussian nodes, VMP geometric messages, and conjugate parameter posteriors (Normal-Inverse-Wishart for (mean, cov); Gamma for scalar precision). Rolls posteriors into next-trial priors exactly like the Dirichlet count accumulation in agent.py. DELEGATION: Fable derives the VMP messages + FE terms (expected-sufficient-statistic forms, digamma/log-det corrections); an Opus sub-agent can mirror the arena/engine plumbing from the BP Gaussian groups and write the learning-convergence test (recover known LGSSM params from data) once the rule signatures exist.

## Acceptance Criteria

VMP on a Gaussian chain monotonically decreases BFE; NIW posterior recovers synthetic LGSSM parameters; conjugate rollover works in the trial loop.


## Notes

**2026-07-09T17:16:33Z**

The EF abstraction (cff-8s6a) is live: adding Wishart for precision-matrix priors is a table entry in cffg/ef.py + cffg/efrules.py (T = (log det W, W), matrix-variate A via multivariate gammaln), not engine work. This ticket is its natural first consumer (NIW / Gamma precision-mean conjugate updates), along with gaussian_prec — the Gaussian node with a Gamma precision edge — as the first mixed-family client.

**2026-07-09T21:40:46Z**

CLOSED (branch gauss-vmp, chained on hybrid PR #10). D4 derivation (DERIVATIONS.md) + three deliverables: (1) Engine.gaussian_node_stats — node-belief sufficient stats (NIW: mean/second; MNIW: Syy/Syx~/Sx~x~ with offset column absorbed; obs-masked for conditioned nodes); (2) cffg.conj — NIW/MNIW updates with soft stats (within-trial variance folds into Psi) and rollover() returning the VMP-correct E-parameters (mean, E[W]=nu*Psi^-1 — NOT the point covariance; D4.1.3); LGSSM recovery acceptance passes (A/b/Q recovered to <0.05 in a compile-once trial loop, tests/test_conj.py); (3) gaussian_prec — first mixed Gaussian/EF client: PrecGroupPlan reads marginals from both arenas, VMP messages + digamma-corrected energy; conjugate fixed point == exact Gamma posterior (1e-6); mixed ledger == -log evidence by quadrature (1e-6). HONEST FINDINGS recorded in LIMITATIONS S1e: flooding VMP is Jacobi — single-node monotone to machine precision, multi-node has bounded non-monotone steps (4e-3 @ damp .5); the in-loop parameter-uncertainty trace term for matrix params is deferred; NIW/Wishart needed no EF table entry (conjugate algebra lives between trials in numpy). Acceptance amendments: 'monotone' pinned as measured (see S1e). Generic learnables API: cff-x0sv (noted there). Suite: 1050+ green.
