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

