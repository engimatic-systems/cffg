---
id: cff-fr5c
status: closed
deps: [cff-x0sv, cff-l29q]
links: [cff-iwgl]
created: 2026-07-08T22:00:30Z
type: feature
priority: 3
assignee: coopwilliams
---
# Compile VMP Dirichlet param_edges: in-graph conjugate posteriors

## Plain English

This ticket lets uncertain categorical model parameters live inside the
factor graph. During inference, state beliefs contribute fractional evidence
to a Dirichlet posterior, and the posterior contributes its expectations back
to the state inference and free-energy ledger.

Out-of-graph Learnables updates deliver the conjugate posterior cheaply; this
ticket adds the in-graph mean-field form so a parameter posterior participates
within a trial and the constrained VFE ledger includes it. Implement the
reviewed cff-l29q contract for categorical, matrix_cat and
transition_mixture likelihoods: FactorSpec.param_edges compiles only with the
required mean-field constraint, the Dirichlet prior and likelihood exchange
natural-parameter VMP messages, and the arena stores proper Dirichlet
parameters.

Scope is deliberately not exact latent-state BP. Its sum-product
likelihood-to-parameter message is a mixture, not Dirichlet, except when every
structural role is hard observed; unsupported unconstrained uses must fail
clearly. The Part II mu3(A) message is also nonconjugate and does not fit these
slots. cff-iwgl may use this posterior as its base proposal only after
cff-almk defines a separate functional-message contract.

## Design

Compiler, runtime, serialization and the ledger implement one VMP Dirichlet
parameter-edge path from cff-l29q. No generic callable or sampled message is
smuggled into the natural arena. The independent equivalence oracle is one
converged mean-field trial whose in-graph posterior parameters match a
Learnables fold from the same node sufficient statistics. Structure remains
serializable from template plus blobs.

## Acceptance Criteria

Categorical, matrix_cat and transition_mixture parameter edges compile and run
under their declared mean-field constraints; an unconstrained latent use is
rejected as non-Dirichlet-closed. In-graph posterior parameters match the
independent cff-l29q algebra and the out-of-graph Learnables update to
tolerance. Node energy, parameter entropy and edge correction reproduce the
direct constrained VFE. Template JSON/.cffgz round-trip preserves parameter
edges and constraints. Exact-zero sufficient-statistic entries remain zero;
missing/improper priors fail loudly; sparse positive float32 concentrations
and 10-100x category/action scale remain finite. Existing non-parameter models
and structure hashes do not regress. This ticket supplies cff-iwgl's proper
base posterior, not its mu3 message representation.

## Notes

**2026-07-28T15:47:56Z**

Implemented the D6a in-graph VMP path for categorical, matrix_cat, and transition_mixture param_edges. Added Dirichlet arena plans, proper alpha0+fractional-count packing, expected-log structural updates, stable complete VFE with literal node/edge decomposition anchors, posterior readouts, schedule/support/decode isolation, JSON/.cffgz coverage, and README/ARCHITECTURE updates. Acceptance tests cover Learnables equivalence, exact zeros, improper priors, float32, and a 64x48x20 table. Focused/broad validation: 153 relevant regressions + 14 hygiene/colony/estimate + final 58 focused tests passed. Next: cff-almk defines the separate functional mu3 contract.
