---
id: cff-n60h
status: closed
deps: [cff-o7hs]
links: []
created: 2026-09-15T18:28:18Z
type: feature
priority: 2
assignee: coopwilliams
parent: cff-1o72
---
# Add portable neural Gaussian mean architectures

## Background in Plain English — WHY

A deterministic network supplies the mean of the positive-noise Gaussian factor. Portable architecture data and flat weight blobs let the model be rebuilt without serializing Python objects.

## Outcomes / Acceptance Criteria — WHAT

1. A namespaced multilayer perceptron mean records layer widths, smooth hidden activation, fixed bias/packing conventions, compatible version, and nonlinear approximation method.
2. Core loading, hashing, editing and bundle inventory remain JAX-free; runtime initialization uses an explicit PRNG key.
3. Values and Jacobians match independent scalar and multidimensional references in float32 and float64.
4. Compilation, named writes, tied parameter rows, JSON/CFFGZ reconstruction, and both schedule routes reuse the nonlinear factor contract.
5. The estimator names persistent network parameters and declared per-sweep work; a tenfold-scale fixture has a measured envelope or truthful non-convergence.

## Methodological Suggestions — HOW

Use the New factor or rule family and New IR field change paths in ARCHITECTURE. Audit host/JIT, single/vmap, flooding/scheduled, latest/legacy loading, and blob ownership. No new central type branch is needed. Sparse discrete support is N/A; vacuity, invalid noise, non-finite weights, and float32 reuse cff-o7hs guards.

## Future work

cff-v53k supplies gradients; cff-ll6y supplies training continuation; cff-3zzo supplies amortization.

## Definitions

Multilayer perceptron: consecutive affine layers with a smooth activation between layers. Flat packing: a declared one-dimensional order of weights and biases.

## Notes

**2026-09-15T18:35:29Z**

Implemented cffg.mean.mlp through the existing configured nonlinear factor: portable widths/activation, fixed matrix-then-bias packing, explicit-key initialization, named parameter ownership, and declared work counts. Initial focused validation:14 passed in27.70 seconds, including independent float32/float64 values/Jacobians, both approximation methods and schedule routes, bundle reconstruction, and a97-parameter (greater than tenfold) model that converges within the60-second/8-KiB fixture envelope. Core import blocking also covers an actual neural declaration. Final expanded focused check is running; parent cff-1o72 still owns full-suite/PR validation.

**2026-09-15T18:35:51Z**

Final focused validation: 27 tests passed in 28.67 seconds across neural rules, core import exclusion, kernel registration, resource estimates, and public exports. No code failures remain in this child. Full parent short-suite certification follows the remaining neural gradient/training/amortization implementation.
