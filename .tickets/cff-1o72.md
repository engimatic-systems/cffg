---
id: cff-1o72
status: open
deps: [cff-o7hs]
links: []
created: 2026-07-08T16:50:10Z
type: feature
priority: 2
assignee: coopwilliams
parent: cff-ahps
---
# Neural factors: NN nodes with backprop, Laplace messages, amortized recognition

Neural networks as factors, three integration levels, in order of increasing ambition: (1) DETERMINISTIC NN NODE: y = f_theta(x) as a nonlinear Gaussian factor (builds directly on the nonlinear ticket — linearization via jax.jacfwd through the net; Equinox modules as the parameter container, consistent with ARCHITECTURE dependency policy). (2) LEARNED FACTORS VIA FE BACKPROP: the whole MP sweep is differentiable JAX — grad of BFE/GFE w.r.t. theta through unrolled message passing trains NN-parameterized factors (energy-based learning); needs implicit-diff-vs-unroll decision (fixed_point implicit gradients preferred, document). (3) AMORTIZED RECOGNITION: NN proposes initial messages / importance-sampling proposals for the IS-based GFE messages (variance reduction for cff-iwgl-adjacent machinery). DELEGATION: Fable derives the Laplace-approximation message forms + the implicit-gradient correctness argument; Opus implements Equinox plumbing + training loops on a toy nonlinear regression factor graph; Sonnet surveys deep-FE literature (specific questions: how do Ergo/RxInfer delta nodes handle NN factors; what breaks with implicit diff at loose fixed points).

## Acceptance Criteria

NN observation factor trained by FE gradient recovers a known nonlinear map; gradients through the solver validated against finite differences; amortized init demonstrably reduces sweeps-to-coherence on a benchmark model.

