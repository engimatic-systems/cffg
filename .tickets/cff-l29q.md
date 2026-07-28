---
id: cff-l29q
status: open
deps: [cff-x0sv]
links: []
created: 2026-07-28T04:20:00Z
type: task
priority: 2
assignee: coopwilliams
parent: cff-mwjo
tags: [derivation, vmp, dirichlet, free-energy]
---
# D6a: VMP Dirichlet parameter-edge messages and FE contract

Derive the conjugate in-graph parameter path before cff-fr5c changes arenas. Scope is deliberately VMP/mean-field: for f(x,theta)=theta_x the exact sum-product message sum_x m(x) theta_x is not Dirichlet unless x is observed, whereas exp(E_q log f) is Dirichlet-shaped with fractional sufficient counts. Cover categorical, matrix_cat and transition_mixture likelihoods, the Dirichlet prior factor, parameter-edge entropy/energy corrections, and the exact conditions under which the in-graph VMP posterior equals the out-of-graph Learnables update. Exact latent-state BP and the nonconjugate Part II mu3 message are out of scope.

## Design

Add one plain-English and formal D6a section to DERIVATIONS.md. State factor-to-parameter and parameter-to-factor VMP messages, natural-parameter conventions, mean-field constraint placement, node terms and complete VFE ledger accounting. Slot packing remains an implementation decision in cff-fr5c, not part of the derivation. Add independent executable anchors using tiny enumerated categorical/matrix/transition models and direct Dirichlet algebra. Hostile-input contract: exact-zero state mass contributes exactly zero count; a proper prior must make the marginal proper and missing/improper priors fail loudly; sparse positive concentrations, float32 expected logs and 10-100x category/action scale are covered without absolute guards.

## Acceptance Criteria

DERIVATIONS D6a includes the plain-English explanation, formal VMP updates, FE/entropy terms, conjugate-equivalence theorem with exact assumptions, and explicit proof that general BP is not Dirichlet-closed. Named independent tests pin categorical, matrix_cat and transition_mixture fractional counts, posterior parameters and total mean-field VFE against direct algebra/enumeration. Tests cover exact-zero mass, improper/vacuous prior behavior, sparse float32 concentrations and uncomfortable category/action scale. No arena/compiler implementation and no claim that mu3 fits a Dirichlet message.
