---
id: cff-iwgl
status: closed
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

## Notes

**2026-07-28T16:26:17Z**

Implemented the serialized cffg.mu3_is.v1 evaluator end to end. Compiler groups all future slices sharing one A-count blob; Engine.prepare_mu3 draws fixed particles from the proper Dirichlet base with an explicit key and returns next_key; deterministic outer coordinates combine slice log messages, expose weighted A/h moments, log Z, KL and ESS, and report converged/budget-exhausted/diverged outcomes without Dirichlet moment matching. The complete FE ledger adds the shared parameter KL once, and score_mu3_epochs averages only converged post-burn-in epochs. build_rollout(mu3=True) is the opt-in surface; Colony rejects mu3 until it has per-instance key threading, documented in LIMITATIONS. Production tests cover D6b binary quadrature, exact-key replay, shared slices, base-posterior injection, FE/policy-score identities, sparse/exact-zero support, improper/collapsed proposals, float32 scale and fixed-path regressions. Validation: focused regression set 110 passed; final dashboard suite 1264 passed, 0 failed, 4 third-party deprecation warnings. Dashboard refreshed at ~/agent-share/cffg-tests.html.
