---
id: cff-iwgl
status: open
deps: [cff-fr5c, cff-almk]
links: [cff-fr5c]
created: 2026-07-06T00:26:48Z
type: task
priority: 3
assignee: coopwilliams
parent: cff-mwjo
---
# mu3 IS message: future-slice contributions to q(A)

## Plain English

This ticket lets simulated future observations influence the agent's
uncertainty about its observation model `A`. It evaluates the planning
message with reproducible importance sampling and combines contributions from
multiple future slices into the parameter beliefs used for policy scoring.

The remaining nonconjugate Part II parameter message:
log mu3(A) = z^T xi(A), from planning slices back to q(A). It is not a
Dirichlet or any other standard exponential-family message: cff-fr5c supplies
the proper conjugate base posterior/proposal, while cff-almk specifies a
separate serialized functional evaluator and runtime importance-sampling
state. Do not store mu3 in a Dirichlet natural-parameter slot or moment-match
it silently. This is not needed for Fig.9 reproduction; it matters for full
planning-to-parameter epistemics.

## Design

Implement exactly the cff-almk evaluator, proposal, log-weight, PRNG, effective
sample size and stochastic-status contract. Sampling state is derived runtime
state; the evaluator identity and graph inputs remain reconstructible from the
template. Burn-in GFE averaging estimates policy score but is not itself a
convergence test.

## Acceptance Criteria

On the binary oracle from cff-almk, log mu3 and every required q(A)
expectation match quadrature within the preregistered Monte Carlo tolerance
and reproduce bitwise from the same explicit key. Multiple planning slices
combine their functional log messages without converting them to Dirichlet
counts. Sparse/exact-zero support, improper proposal refusal, all-nonfinite or
collapsed weights, float32 and uncomfortable observation/state scale follow
the derived contract. Runtime results distinguish converged,
budget-exhausted and diverged stochastic inference; non-finite FE or weights
are divergence. The FE ledger and burn-in policy score match the D6b oracle,
and fixed-A/perception-only behavior does not regress.
