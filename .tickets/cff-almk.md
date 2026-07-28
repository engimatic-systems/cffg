---
id: cff-almk
status: closed
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

## Notes

**2026-07-28T15:59:56Z**

Research (question -> answer -> source -> confidence): What exactly is mu3 and which q(A) moments does Fig. 5 need? The PDF-preserved overbars give r=E[log c]-log(Abar*zbar), xi(A)=A^T*r-h(A), log mu3=zbar^T*xi(A); the composite needs only Abar=E[A] and hbar=E[h(A)] (not E[log A]). Sources: Part II sec.5.2/Fig.5/App.C.1-C.2/C.6, notes/pt2.txt lines 1116, 1161, 2032, 2104, 2180. Confidence high (PDF visually checked because extraction drops overbars). Why non-Dirichlet? The log message contains linear A and A*log(A) terms, not fixed coefficients on log(A); only zbar_k=0 columns are vacuous. Confidence high. What does Julia LAIF do versus the paper? LAIF commit 9453782 uses a ContinuousMatrixvariateLogPdf closure, SampleListFormConstraint(20, LeftProposal()), weighted A/h moments, global RNG, 50 iterations and mean FE after iteration 10; BayesBase uses linear-space weights and ESS N/10 resampling/reuse, with no exposed ESS/status. Sources: biaslab/LAIF goal_observation.jl lines 215-227, distributions.jl lines 20-50/95-100, Part2 notebooks/agent.jl/environment.jl; RxInfer form_sample_list.jl and BayesBase samplelist.jl. These are implementation choices: the paper specifies functional message, IS expectations, stochastic FE and burn-in averaging, but not proposal/count/ESS/resampling/RNG. Confidence high, except medium-high that LeftProposal resolves to the MatrixDirichlet base through RxInfer operand ordering.

**2026-07-28T16:01:43Z**

Completed D6b after resolving the PDF overbar ambiguity: mu3 uses r=log c-log(Abar*zbar), with Abar held as the current marginal mean during functional evaluation. DERIVATIONS now defines the serialized cffg.mu3_is.v1 metadata, shared-base multi-slice tilt, self-consistent Abar/hbar moments, base-proposal log-space IS, exact key splitting, fixed-particle convergence, refresh/no-hidden-resampling policy, ESS/collapse behavior, complete -log Z parameter ledger, and burn-in versus convergence statuses. Added binary product-Beta quadrature and explicit-key/hostile/float32/scale anchors in tests/test_mu3_derivation.py; 4 focused tests pass. Research sources and reference-implementation distinctions are in the prior ticket note. Next: cff-iwgl implements this contract.
