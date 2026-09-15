---
id: cff-ll6y
status: in_progress
deps: [cff-v53k, cff-6kwo]
links: []
created: 2026-09-15T18:28:19Z
type: feature
priority: 2
assignee: coopwilliams
parent: cff-1o72
---
# Train neural means with versioned optimizer continuation

## Background in Plain English — WHY

Certified fixed-point gradients need an optimizer and a restartable training state. The saved template remains the authority for the network and its positive noise.

## Outcomes / Acceptance Criteria — WHAT

1. Training applies only certified finite gradients and reports refused updates without mutating parameters.
2. A frozen seeded positive-noise nonlinear regression benchmark beats its untrained network and linear baseline on a held-out grid.
3. A separate versioned optimizer checkpoint binds structure, target blob layout, fixed parameters, numerical settings and complete continuation arrays.
4. Restoration is validated before publication and preserves the next update/objective; missing or legacy schema fails specifically unless explicitly migrated.
5. BFE training is supported; GFE training is admitted only within the complete graph capability boundary, with unsupported combinations rejected.

## Methodological Suggestions — HOW

Use Learnable or checkpoint feature and Runtime numerical or status invariant. Audit tying, reconstruction, dtype, fresh/stale certificates and atomic load errors. Freeze data generation, split, seeds, step rule and numeric acceptance before final comparison. Optimizer state is not a conjugate Learnables posterior.

## Future work

cff-3zzo provides deterministic amortized initialization; exact zero noise remains cff-u06e.

## Definitions

Optimizer continuation: weights, moment arrays, step count and any owned stochastic/data-position state needed to reproduce the next update. Held-out data: examples not used to fit the weights.

## Notes

**2026-09-15T18:56:17Z**

Freeze production qualification before running it: h(x)=tanh(2x); 17 float64 train centers linspace(-1.6,1.6), independent Threefry2x32 JAX keys3101/3102 for latent sigma0.04 and output sigma0.08 noise. Hold out16 midpoint centers linspace(-1.5,1.5) against the noise-free mean. Neural(1,2,1), linearization, initialization key71, seven tied parameters. Run32 full-batch Adam updates, lr0.03, betas0.9/0.999, epsilon=sqrt(dtype eps), dividing energy and gradient by17. Every update requires a certified root and original adjoint residual. Acceptance: held-out RMSE<=0.075 and <=half each untrained and affine baseline RMSE; mean training energy drop>=1 nat. Fit affine baseline to the same positive-input-uncertainty Gaussian evidence objective by evaluating every real root of P^2*a^3+P*C*a^2+(V*(P+M)-P*Q)*a-V*C=0, where M,C,Q are centered input second moment, cross moment, output second moment. Research question: can this fixture distinguish useful training? Answer: analytic exploratory screen predicts RMSE0.04897 vs0.17219/0.23419 and energy drop1.8272; source /tmp/cffg_neural_benchmark_astra.py and independent Gaussian evidence algebra; confidence high for screening, production qualification still required.

**2026-09-15T19:11:48Z**

Implemented immutable full-batch Adam continuation, certified-update refusal, and separate versioned numeric checkpoints. Restore binds original template/initial arrays, observations/reference messages, numerical/optimizer settings, dtype, canonical tied layout, complete moment arrays and step. Trained point arrays export through existing .cffgz owner. No stochastic stream or minibatch position is owned. Both dtypes replay the next update/objective through a freshly reconstructed trainer; malformed/legacy snapshots fail before publication. Frozen production benchmark passed32 certified updates: held-out RMSE0.0489668403 versus untrained0.1721923128 and same-objective affine0.2341920825. Mean energy0.6313725392 -> -1.1958261059. Command uv run pytest -q -m slow tests/test_neural_training.py -s —1 passed,3 deselected in55.82s; log /tmp/cffg-neural-training-benchmark.log. Final GFE sparse-gradient audit is tracked as prerequisite cff-6kwo, and the final parent short suite remains pending.
