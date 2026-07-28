---
id: cff-almk
status: open
deps: [cff-l29q]
links: []
created: 2026-07-28T04:20:19Z
type: task
priority: 3
assignee: coopwilliams
parent: cff-mwjo
tags: [derivation, importance-sampling, gfe, parameter-edge]
---
# D6b: functional mu3(A) message and importance-sampling contract

## Plain English

This ticket specifies how a planning graph sends a nonstandard message to an
uncertain observation model. It defines the function being evaluated, the
importance-sampling procedure, reproducible random-key handling, diagnostics,
and the free-energy and convergence rules that the implementation will follow.

Specify the nonconjugate Part II message log mu3(A)=z^T xi(A) before cff-iwgl implementation. The papers explicitly state that mu3 is not a standard exponential-family distribution; it therefore cannot inhabit cff-fr5c Dirichlet natural-parameter slots. Derive the functional log-density contract, its combination with the proper Dirichlet base posterior from perception/prior messages, and the expectations required by the GFE composite.

## Design

DERIVATIONS D6b must choose a serializable executable representation rather than a Python closure: identify the registered evaluator and its parameter/blob/edge inputs in template or kernel metadata, while samples and weights remain runtime state. Specify proposal distribution, log-space weights, normalization, sample reuse/resampling schedule, explicit PRNG splitting, effective-sample-size diagnostics, degeneracy response, and how stochastic FE traces yield converged/budget-exhausted/diverged statuses. Burn-in averaging is a policy-scoring estimator, not permission to label a fluctuating solver converged. Independent anchors use a tiny binary observation model with simplex quadrature or high-accuracy integration. Hostile cases: sparse Dirichlet proposals and exact-zero likelihood support, vacuous/improper base posterior, all-nonfinite weights, float32, and observation/state dimensions 10-100x the toy fixture.

## Acceptance Criteria

DERIVATIONS D6b contains plain-English and formal mu3 derivations, proves why no Dirichlet natural message represents it, and defines the serialized evaluator plus runtime sampling state. Named independent tests compare log mu3 and required expectations with binary-model quadrature, verify deterministic replay from an explicit key, and exercise sparse/zero support, improper proposal refusal, weight collapse, float32 and scale. The status/FE policy distinguishes stochastic estimator averaging from inference convergence. cff-iwgl depends on this contract; cff-fr5c is described only as the conjugate parameter-edge foundation.
