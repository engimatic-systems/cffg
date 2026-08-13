---
id: cff-0fw2
status: open
deps: [cff-wjqg, cff-lv2w]
links: []
created: 2026-08-13T04:33:03Z
type: feature
priority: 2
assignee: coopwilliams
parent: cff-wbhu
tags: [performance, runtime, gfe]
---
# Compile and batch the deterministic mixed-GFE controller

## Background in Plain English

The mixed-GFE solver uses a proof-backed trust-region method with active support
faces, local inner solvers, stationary free-energy checks, and a complete raw-
root certificate. Its JAX numerical kernels are compiled, but Python still
drives each trust-region decision and runs one controller per Colony instance.

`cff-wjqg` first establishes the ordinary compiled-loop and per-instance status
carry. This ticket applies the same device-control principle to deterministic
mixed-GFE solves without changing their root, chart, trust-region method,
support policy, inner solver, or free-energy meaning.

## Outcomes / Acceptance Criteria

1. `cff-wjqg` is closed with device-side per-instance stopping and trace carry.
2. `cff-lv2w` is closed with the production matrix-free trust kernel and full
   convergence certificate used as the parity base.
3. Deterministic mixed-GFE `Engine.run` executes one compiled controller per
   fixed chart and support phase.
4. A same-structure Colony group batches compatible mixed-GFE controllers.
   Compatibility includes chart signature, support shape, dtype, schedule,
   objective plan, and static solver limits.
5. Each instance carries its own coordinates, frozen residual scale, trust
   radius, support face, inner status, termination reason, accepted and
   rejected counts, map and trial evaluations, linear iterations, and trace
   lengths.
6. One instance's convergence, rejection, support change, budget exhaustion,
   divergence, or infeasibility cannot alter a peer's state or result.
7. The compiled controller calls the existing matrix-free JVP/VJP and
   Steihaug trust kernel. It does not introduce another root solver or change
   the acceptance ratio, Cauchy safeguard, trust-radius update, or restart
   policy.
8. `CONVERGED` still requires the complete raw residual, stable exact-support
   face, successful inner solvers, coherent messages, and finite stationary
   free-energy ledger.
9. A support or hardening change ends the current compiled phase and rebuilds
   the chart through the existing authority before another phase begins.
10. Sequential and compiled results agree on benign roots, finite nonroots,
    support changes, delta phases, retry paths, and every public diagnostic
    under dtype-derived tolerances.
11. Hostile coverage includes exact support collapse, an improper continuous
    belief, non-finite residual or ledger, float32, distinct per-instance
    convergence counts, and at least 32 batched mixed-GFE instances.
12. A committed benchmark reports sequential and batched mixed-GFE timing,
    compilation, first execution, per-member statuses and termination reasons,
    map evaluations, trust-kernel evaluations, and device submissions. It makes
    no accelerator claim without measured accelerator evidence.
13. Public call forms, template serialization, structure hashes, checkpoints,
    solver diagnostics, and objective meaning remain unchanged.
14. Architecture and limitations documents remove the per-instance Python
    controller limitation only after the parity and benchmark evidence passes.

## Methodological Suggestions

These suggestions are optional. An implementer may choose another sound method
that satisfies the numbered outcomes with equivalent evidence.

- Express the existing host controller as immutable array carry and integer
  termination codes. Convert to rich public diagnostics only after the device
  phase returns.
- Partition Colony instances by the complete static controller signature before
  vectorizing them.
- Commit characterization for every termination reason before moving control
  flow.
- Use the **Runtime numerical or status invariant** change path and walk direct
  versus Colony and host versus compiled controller pairs.

## Future work

`cff-g7rc` retains cold-compile and fixed-certificate overhead for tiny mixed-
GFE solves. Stochastic `mu3` batching remains `cff-rqqc` work. This ticket does
not change factor rules or add a new GFE family.

## Definitions

- **Chart:** Minimal coordinates and a fixed metric for the active mixed-GFE
  support face.
- **Raw-root certificate:** The complete undamped fixed-point, support, inner-
  status, message, and free-energy checks required for convergence.
- **Steihaug method:** The truncated conjugate-gradient method used inside the
  trust-region step.
- **Trust radius:** The current bound on a proposed coordinate step.
