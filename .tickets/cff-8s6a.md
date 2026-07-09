---
id: cff-8s6a
status: closed
deps: [cff-4e5z]
links: []
created: 2026-07-08T16:49:34Z
type: feature
priority: 2
assignee: coopwilliams
parent: cff-ahps
---
# Exponential-family library: natural-parameter abstraction + Gamma/Beta/Poisson/Wishart

Refactor message families around natural parameters + log-partition functions so a new EF distribution is a table entry, not an engine change: message product = natural-param addition; marginal moments = grad log-partition; node energies = closed-form from (natural params, log-partition, expected sufficient stats). Then add Gamma (rates/precisions), Beta (probabilities), Poisson (counts), Wishart (precision matrices), each with conjugate-pair BP/VMP rules where they exist. Reconsider distrax/tfp.substrates.jax for log-partition/moment functions per ARCHITECTURE dependency policy (reopens at the continuous milestone) — a ticket-worthy decision, document either way. DELEGATION: Fable derives the abstraction contract + each conjugate message table; Opus ports Cat/Dir/Gaussian onto the abstraction behind the existing test suite (pure refactor, tests must not change); Sonnet drafts per-distribution property tests (moments vs scipy, KL identities).

## Acceptance Criteria

Cat/Dir/Gaussian re-expressed on the abstraction with zero test regressions; at least Gamma+Beta live with BP/VMP rules and FE terms; property tests vs scipy for moments/KL.


## Notes

**2026-07-09T04:02:13Z**

PARTIAL LAND (branch exp-families): the Fable-level math is done and tested — cffg/ef.py (contract + Gamma/Beta tables + conjugate messages + observation energies, all quadrature-pinned) and DERIVATIONS.md D2 (plain-English + contract + integration plan). REMAINING (engine-facing, deliberately left for fresh context per the robustness contract): (1) VariableSpec domain generalization to tagged domains; (2) third arena ef_eta [n,2] with per-group family tags; (3) factor types gamma/beta priors + poisson_obs/bernoulli_obs conditioned likelihoods; (4) SEPARATE zero-regression refactor ticket for re-expressing Cat/Dir/Gaussian on the abstraction — do NOT fold into feature work. gaussian_prec (Gamma precision edge on a Gaussian node) belongs to cff-7c5o as the abstraction's first mixed client. Hostile-inputs checklist for the arena work: vacuous eta=0 propriety, improper products mid-flooding, f32 (the vendored psi is f64-tuned — mirror in jnp with dtype-aware series), observed x=0 counts.

**2026-07-09T17:16:33Z**

CLOSED (branch ef-engine, chained on the exp-families PR). Landed beyond the earlier partial note: (1) VariableSpec.ef tagged domains + additive serialization + per-role family validation (EF_EDGE_FAMILIES); (2) third arena ef_eta [n_efslots,2] with EfGroupPlan (priors/conditioned obs/equality), independent slot numbering — the not-e.dim==discrete assumption is gone everywhere (engine._uniform_messages, marginals, bind_data, interior_slot_pairs); (3) gamma/beta prior leaves (scalar blobs, propriety = CompileError) + poisson_obs/bernoulli_obs conditioned likelihoods (out edge must be DATA port; None drops the node with exact -H[mu_in] telescoping); (4) jnp rules in cffg/efrules.py use jax.scipy.special digamma/gammaln (no vendored psi needed — dtype-aware, f32-tested); improper beliefs make every ledger term +inf, never NaN (edge entropy too, deliberately, to avoid inf-inf). Hostile-inputs checklist answered in tests: vacuous eta=0 (gamma improper->inf bfe, beta proper->Uniform), x=0 counts, f32 rules parity 2e-4, mixed discrete+EF arena isolation. BFE = -log evidence pinned 1e-12 (neg-binomial and Beta-Bernoulli marginals). Dependency decision recorded (ARCHITECTURE): no distrax/tfp; scipy dev-only as test oracle. Property tests vs scipy: 750 cases (Sonnet-drafted per delegation plan, verified). Spun off: cff-l6i5 (zero-regression Cat/Dir/Gaussian re-expression, Opus-suitable); Wishart + gaussian_prec noted on cff-7c5o. Acceptance amended per the 2026-07-09 partial-land note: the Cat/Dir/Gaussian re-expression criterion moved to cff-l6i5.
