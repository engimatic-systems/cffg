---
id: cff-v53k
status: closed
deps: [cff-n60h]
links: []
created: 2026-09-15T18:28:19Z
type: feature
priority: 2
assignee: coopwilliams
parent: cff-1o72
---
# Expose and differentiate the certified adaptive fixed-point system

## Background in Plain English — WHY

Neural training needs the derivative of the settled inference result. The derivative must include how changing weights and beliefs change the local Gaussian approximation.

## Outcomes / Acceptance Criteria — WHAT

1. Inference and training share one owner for chart coordinates, exact-island reconstruction, the complete residual, and the raw energy.
2. A canonical parameter plan preserves blob tying through JAX-only parameter injection.
3. A matrix-free adjoint solve certifies its original residual and rejects non-finite, exhausted, singular or unsupported solves before exposing an applicable gradient.
4. Gradients require a matching certified inference result, finite energy and fixed support; stale weights or observations cannot reuse an old certificate.
5. A scalar neural fixture with at most 20 parameters agrees with independent central differences; both local approximation methods include total coefficient derivatives.

## Methodological Suggestions — HOW

Use Runtime numerical or status invariant and New factor or rule family. Audit flooding/mixed reconstruction, host/JIT numerical kernels, ties, both dtypes and stale state. Differentiate R_u transpose lambda=E_u and dE/dtheta=E_theta-R_theta transpose lambda. Do not use the trust normal-equation subproblem as the adjoint certificate.

## Future work

cff-ll6y applies certified gradients; cff-3zzo learns proposals from certified roots.

## Definitions

Adjoint: an auxiliary linear solve for the derivative of a scalar objective. Gauge: equivalent message coordinates removed before solving. Matrix-free: derivative-vector products without a whole-graph Jacobian.

## Notes

**2026-09-15T18:42:38Z**

Shared RootProgram extraction is complete: it owns chart decoding, exact-island reconstruction, adaptive update, support-preserving hard values, residual and raw energy composition. Existing outer inference consumes it. Focused compatibility validation passed 47 tests with 1 slow deselection in 101.66 seconds. A first scalar neural derivative smoke check converged with adjoint relative residual 5.15e-16; full independent finite-difference qualification is next.

Freeze gradient qualification before that comparison: two noisy tanh neural sensors share one scalar latent prior N(0,1) through equality; sensor/evidence variances are 0.2, evidence means +/-0.3. Both sensors share four parameters [0.8,0.15,1.1,-0.05] in a 1-1-1 network (matrix,bias,matrix,bias). Test Taylor and sigma methods against central differences of independently re-solved complete engine energy. Use float64 root tolerance 1e-10, parameter step h=2e-4*(1+abs(theta)), and acceptance atol 2e-5 plus rtol 2e-3. The step keeps root-error/step and roundoff below acceptance while limiting second-order truncation; smaller-step sensitivity will be reported. Tying must sum both reader contributions. This fixture changes cavities and therefore tests coefficient dependence, unlike a single fixed-cavity sensor.

**2026-09-15T18:57:07Z**

Implemented canonical tied-weight injection and the complete implicit derivative with a reusable fixed coordinate chart, dynamic observations/weights, JAX JVP/VJP operators, and GMRES with an independently recomputed original adjoint residual. Frozen qualification passed for both changing-cavity methods, half-step sensitivity, actual adjoint budget exhaustion, stale states, tying, and float32. Final focused validation: uv run pytest -q tests/test_fixed_point_gradients.py tests/test_public_api.py — 12 passed in63.94s. D14 states the local isolated differentiable-root assumption: residual and amplification certificates detect numerical failure but do not prove nonsingularity in every unprobed direction or global root uniqueness. Exact support changes refuse reuse; stochastic mu3 and discrete delta decisions are explicitly unsupported. Architecture/README and dashboard inventory updated. Parent PR still requires one final short suite after training and amortizer land.

**2026-09-15T19:12:52Z**

Follow-up integration: moved the shared root/energy certificate into RootProgram so differentiation and persisted recognition references use one equation owner. Reusable derivative compilation still keeps all model context dynamic. Combined final focused neural validation after the refactor passed 10 tests with 2 opt-in deselections in 118.42s. Unsupported trainable-factor selection now returns an explicit unsupported gradient result.
