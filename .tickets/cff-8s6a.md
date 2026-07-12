---
id: cff-8s6a
status: in_progress
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
