---
id: cff-03dn
status: open
deps: [cff-04mb]
links: []
created: 2026-09-25T07:25:11Z
type: bug
priority: 1
assignee: coopwilliams
parent: cff-ectw
tags: [pcgym, numerics]
---
# Prevent rank and safe-inverse artifacts in Gaussian mixture messages

## Background in Plain English

The saved PCGym hybrid model fails because a rank-one Schur precision acquires a tiny positive eigenvalue after floating-point reconstruction. The mixture rule treats it as invertible. Its collapsed covariance becomes indefinite, yet the safe inverse helper substitutes an identity internally and that placeholder escapes as a real outgoing message. Damping changes this branch behavior but does not prove the intermediate arithmetic is sound.

Measured evidence: dyn14 starts with precision diag(400,0); reconstruction has a roundoff eigenvalue about 4.38e-16; collapse covariance has eigenvalues about [-0.00334,1.665e15]. The outgoing isotropic precision about 6.08e-16 is manufactured by the placeholder, not by the model.

## Outcomes / Acceptance Criteria

1. A retained local fixture reproduces rank-one admission and invalid-collapse behavior without PCGym, training or pickle dependencies.
2. The rule distinguishes a numerically supported full-rank component from rank created by reconstruction roundoff. It does not globally change the proper-Gaussian predicate or silently alter process covariance.
3. Safe branch placeholders never become accepted mixture outputs. Invalid collapse produces a documented truthful local failure or an explicitly justified supported improper-message result.
4. Forward and backward mixture updates agree with independent proper-component moment oracles. Valid tiny but well-scaled information is not erased by an absolute threshold.
5. Exact zero selector support, vacuous/rank-deficient inputs, nonsymmetric or indefinite matrices, float32/float64 and at least tenfold dimension or component count have explicit outcomes.
6. Gradients and supported host/JIT/single/vmap paths remain valid where the numerical branch is differentiable; all documented approximations remain honest.
7. Document the mathematical contract and test anchors in the canonical derivations/limitations as needed. Do not claim the full planner is repaired until cff-ectw validates initialization and convergence.

## Methodological Suggestions

Choose another sound method if it meets the outcomes. Follow the Runtime numerical or status invariant row of docs/ARCHITECTURE.md. Before implementation, list local kernels, host/JIT, flooding/scheduled/exact, Engine/Colony and single/vmap surfaces; checkpoints are N/A unless stored state changes. Keep numerical guards relative to dtype and scale. Continue the validated PCGym PR chain, update the intended base before branching, and run focused tests plus one final short suite.

Inspect mixg.py forward/backward/_collapse, gauss.py Schur projection and efrules.py safe inverses. Consider carrying rank decisions from the projection instead of reclassifying a reconstructed matrix using strict Cholesky alone. Do not add a global covariance floor.

## Future work

- cff-ectw consumes this arithmetic repair before selecting initialization and default retry behavior.

## Definitions

Rank is the number of independent informed directions. A Schur complement eliminates Gaussian variables. A covariance is proper only when positive definite. Moment collapse replaces a mixture with a Gaussian having the same first two moments. A branch placeholder is a finite dummy matrix used only to keep an unselected compiled branch safe. JIT is compiled execution; vmap batches instances; dtype is numeric storage precision.



## Notes

**2026-09-25T07:28:04Z**

Contract review (failure_provenance sub-agent): keep shared strict _pd_ok unchanged; d=100 proper diagonal precision spanning1e-12..1e12 is an existing accepted case. Preserve retained spectral rank at the Schur producer before V diag(eig) V.T reconstruction. Proper-input covariance branches bypass that cutoff. Backward A.T G A needs rank of its generating Gram factor, with coordinate equilibration; singular G can still span every input direction, so requiring full-rank G would be wrong. _collapse must own validity for weights, component means/covariances, collapsed covariance and output; all three callers mixlin_forward, mixlin_backward and mix_forward consume it. Distinguish non-normalizable-yet components from arithmetic collapse failure so a finite placeholder cannot masquerade as information. Independent cases: vacuous scalar input,A=(3,4).T,W=I gives rank-one [[16,-12],[-12,9]]/25; proportional-column backward map; unit-rescaled proper diagonal model; collapse overflow; ordinary quadrature/backward-weight oracles. K=1 parity with plain Gaussian rules applies only when the outgoing component can be normalized, because plain Gaussian messages may legitimately remain singular. Sources gauss.py projection, mixg.py collapse/forward/backward and tests/test_gauss_rules.py mixed-scale propriety. Confidence high code/math review; no patch or experiment executed by reviewer.
