---
id: chr-ozmk
status: closed
deps: []
links: [cff-3kdk]
created: 2026-11-25T06:35:52Z
type: bug
priority: 1
assignee: coopwilliams
---
# Propagate partial Gaussian evidence through affine factors

## Background in Plain English — WHY

A scalar measurement of a two-coordinate state informs one direction. Rounding can make the resulting singular precision matrix pass a Cholesky test. Inverting it as a proper covariance corrupted ordinary backward messages. Forward prediction could promote the same rounded matrix. The 32-state example declared convergence while losing information from its final sensor. The full and individual readouts agreed on the incorrect state.

## Outcomes / Acceptance Criteria — WHAT

1. Recover the dense joint posterior and negative log likelihood for the original sparse 32-state and observed 256-state chains.
2. Preserve partial, weak, sharp and vacuous messages in both floating precisions, including noncommuting matrices, changed units and rectangular affine maps.
3. Preserve gradients on ordinary proper inputs, including repeated eigenvalues, independent lanes and disparate coordinate units.
4. Measure both proper-input rule costs on identical inputs and report changes in typical outputs.

## Methodological Suggestions — HOW

The architecture change path is the runtime numerical or status invariant. The registered Gaussian rules serve flood, scheduled and batched execution. Readout, model codecs and solver tolerances are separate surfaces. Check the first incorrect reverse message, and split the graph into two independent models at a known state. Compare each density and entropy with a dense reference.

## Future work

The exact interior observation feature chr-vyg1 uses the corrected ordinary rule. The parent runs the aggregate suite and dashboard.

## Definitions

An affine factor describes a Gaussian with mean Ax+b. A precision matrix specifies the curvature of the log density. A Gram matrix is a matrix product BᵀB. Its quadratic form is nonnegative. A cavity is the message obtained before applying the receiving factor's evidence.

## Qualification

The untreated sparse chain gave 2.59991964505 vs 2.51456900671 nats. A direct parallel sum fixed the 256-step incident but accumulated a negative eigenvalue of −7.26e−17 at state 24. The PSD check discarded the next message. The final QR and Gram form retains the signal. On the 32-step unbound and bound fixtures the final independent maximum energy error was 3.38e−14 nats. The maximum state mean/covariance errors were 8.89e−15 and 6.40e−14. Every node and edge agreed within 1.34e−15. The split native submodels reproduce the direct evidence. All 100 rank-one forward probes per dtype remained within 7.82e−7 (f32) and 1.34e−15 (f64) of the single-observation oracle.

Apple M2 Max CPU, JAX 0.10.2, f64, 5 warmups per arm and 41 randomized paired samples per shape. The timer enclosed the compiled public rule and synchronization of both outputs. Forward/backward candidate-to-baseline median ratios were 0.971/0.766 (1 member, d2), 0.747/0.804 (256, d2), 0.680/0.712 (256, d4), 0.730/0.742 (32, d16). Mean, precision and natural-vector values on these proper examples agreed to 3.55e−14. This is a local rule measurement. Raw quartiles, samples and commands are in ~/agent-share/cffg-hierarchical-runtime/gaussian-tap/rule-timing.json and rule_timing.py. Independent dense, Rauch–Tung–Striebel smoother, and split-node receipts are in 9fcc47a-e0p19emb/qualification.json in that directory.

## Notes

**2026-11-25T06:54:51Z**

Final focused command covering all Gaussian-rule tests, selected readout tests, and the 32-state regression passed 37 tests in 28.42s. This includes forward/backward JVP versus central differences at repeated eigenvalues, compiled lane-vmap, and W=L=diag(1e-10,1e10) in both source precisions. Other targeted Gaussian, Kalman, mixed, scheduled and readout-owner checks passed earlier (123). The short suite belongs to the parent integration gate.
