---
id: cff-7piw
status: open
deps: [cff-0266]
links: []
created: 2026-11-26T19:44:48Z
type: feature
priority: 2
assignee: coopwilliams
parent: cff-xikb
tags: [studio]
---
# Rename and resize variables and set factor parameters in one model edit

## Background in Plain English — WHY

A variable name or dimension can occur in factors, observations, learning history and shared arrays. Changing only the visible wire can break compilation or silently detach parameters.

## Outcomes / Acceptance Criteria — WHAT

1. Both clients can preview then commit a general rename, dimension change or compatible initializer with the same command. The preview enumerates affected references, arrays, priors and posteriors.
2. Invalid changes explain the offending shapes or unsupported family. Explicit preserve, map or reinitialize choices handle resized data and learning state without silent loss.
3. Seeded random and compatible linear/noisy Gaussian parameter configurations reproduce resolved values after a save and reopen. Shared parameters identify every affected factor.
4. A scalar exponential-family variable survives action JSON round-trip. Stable note anchors survive renames. A genuine rSLDS exercises tied parameters, regime count and observation dimensions.

## Methodological Suggestions — HOW

Use the New IR field and Learnable or checkpoint feature rows. Check action codec, template serialization, plates, compiler resolution and current/legacy bundle interpretation. Host/JIT, schedule and vmap are N/A unless changing numerical kernels. Include exact zero probabilities, uninitialized parameters, singular covariances, float32 and 10× dimensions in the command contract.

## Future work

Expose the accepted commands through Studio menus, the keyboard palette and the agent interface.

## Definitions

An initializer supplies parameter values. A prior records starting learning information and a posterior records updated information. An rSLDS is a recurrent switching linear dynamical system.
