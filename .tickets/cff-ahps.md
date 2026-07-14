---
id: cff-ahps
status: open
deps: [cff-1o72, cff-3hgi, cff-44we, cff-5edo, cff-6s31, cff-9xmq, cff-bxk4, cff-d56d, cff-dfpi, cff-l6i5, cff-lquu, cff-o7hs, cff-ri8m, cff-wbhu, cff-izjd]
links: []
created: 2026-07-08T16:48:44Z
type: epic
priority: 1
assignee: coopwilliams
---
# Roadmap: universal factor library — continuous, hybrid, constraints, neural

Goal: the most capable, modular, performant factor-graph engine we can get — factors for all useful probability distributions (continuous + discrete + hybrid), traditional constraint-satisfaction/optimization factors (optimization under uncertainty), and neural-network factors with backprop. This epic tracks the expansion of the rule registry and arena machinery beyond the discrete Cat/Dir vocabulary.

PHASES (each a child ticket, deps encode order):
1. Gaussian core — domain variant in IR, canonical-form message arena, linear-Gaussian BP, Kalman parity. Unlocks everything continuous.
2. Gaussian VMP + conjugate learning (NIW/Gamma) — continuous analog of Dirichlet learning.
3. Hybrid discrete-continuous — conditional-Gaussian factors, mixture collapse, switching KF. The recommended epistemic path (GFE stays on discrete regime variables).
4. Nonlinear factors — linearization + sigma-point/EP moment matching (activates the reserved MOMENT_MATCHING bead).
5. Exponential-family library — natural-parameter message abstraction; Gamma/Beta/Poisson/Wishart.
6. Constraint/optimization factors — min-sum/max-product semiring mode + hard-constraint library (alldiff, linear ineq); trad optimization under uncertainty.
7. Neural factors — deterministic NN nodes (Equinox), Laplace/delta messages, FE-gradient backprop, amortized recognition.
8. Research: continuous/hybrid GFE epistemics (open literature problem — Koudahl 2021 shows linear-Gaussian EFE yields no exploration).

DELEGATION POLICY (per Cooper 2026-07-08): all message/FE derivations are done by Fable at the top level and land as docstrings next to the rules + a ticket comment. Opus sub-agents: well-specified engineering with existing patterns to mirror (arena plumbing, test scaffolding, builders) — always reviewed at top level. Sonnet sub-agents: research summaries with specific questions, golden-number generation from reference implementations, doc sweeps. Never delegate: FE ledger semantics, constraint/bead semantics, anything touching the true-BFE invariant.

## Acceptance Criteria

All child tickets closed; engine supports mixed discrete/continuous/constraint/neural factor vocabularies with exact FE accounting where closed-form exists, documented approximations elsewhere.

