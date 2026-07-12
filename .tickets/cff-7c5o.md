---
id: cff-7c5o
status: open
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

