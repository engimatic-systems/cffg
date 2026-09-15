---
id: cdic-za1z
status: open
deps: [cdic-7y77, cdic-nuel]
links: []
created: 2026-09-15T09:02:43Z
type: task
priority: 2
assignee: coopwilliams
parent: cdhd-hzw0
tags: [hybrid, numerics, qualification]
---
# Bind the shared iterative policy to the general evaluator and qualify tiny solves

## Background in Plain English — WHY

The current iterative policy directly accesses chain target banks, energy/certificate fields, initialization and an auxiliary tail. The general evaluator contract makes those requirements explicit. Reuse one policy through that boundary and qualify the resulting complete message-plus-auxiliary solve before production binding.

## Outcomes / Acceptance Criteria — WHAT

1. One shared numerical policy handles the qualified chain adapter and general evaluator. Refactoring preserves budgets, globalization, scaled derivative actions, stopping/certification semantics and rejection counters; no policy copy or weaker alternate route is introduced.
2. Generated feasible initialization and explicit-coordinate starts remain distinct entry points. The general initializer establishes required proper aggregate/tilted beliefs or returns a truthful failure; it does not inherit adjacency assumptions silently.
3. Complete target packing, numerical status replacement, fixed evaluation structure and auxiliary bounds follow the explicit interface. Valid empty blocks are distinguished from invalid support, and traced branches support their declared shapes.
4. Returned points distinguish converged, exhausted and diverged using complete certificates and stable finite free energy. Nonfinite values cannot look successful; no-motion or a stationary linear subproblem is not convergence.
5. Prospectively fixed tiny cases exercise complete internal and coupled solves on admitted representative layouts, all required output/statistic groups, initialization sensitivity and explicit hostile/exhausted cases. Independent references and existing approximation/threshold rules govern acceptance.
6. Changed old producers receive new truthful tiny checkpoint evidence and prospective descriptors, including every numerical dependency. Historical tiny004/A3 artifacts remain immutable and clearly source-scoped.
7. Actual retained target/evaluation/derivative storage is inventoried separately from the three-N/one-M CGLS vectors. Numerical moderate and physical memory claims remain downstream.
8. Canonical docs, focused checks, independent review and one passing short suite complete the ticket; public execution and all three recovery fits remain guarded.

## Methodological Suggestions — HOW

Adapt the complete evaluator boundary rather than a residual-only callback. Existing line search constrains the tail after message_size as squared xi, so exterior message unknowns must not enter that tail. Preserve the distinction between a feasible primal objective and a separate dual diagnostic.
Before implementation, record the applicable Derived component coordinate and Runtime numerical or status invariant change-path rows and every affected pair. Unbound numerical host/JIT and single/vmap behavior is in scope where the kernel/evaluator is callable; production synchronous/scheduled, State/lifecycle, ledger/readout activation, serialized learned state and public bundles remain downstream and guarded. Never infer new production support from a local test. Keep signed messages, literal inactive log-zeros, dtype-scaled guards, finite-energy checks, explicit failure status and bounded storage. Sparse support, improper/vacuous cavities, native float32 and independently enlarged incidence/K/d need explicit outcomes. Another sound method preserving all outcomes is welcome.

Qualification source ownership: if old numerical producers change, create new truthful source-bound tiny checkpoint evidence and regenerate the applicable prospective abstract descriptors before the final gate. Include every new transitive numerical dependency in the producer inventory regardless of filename. Preserve historical tiny004 and A3WZ artifacts, specifications, numerical policy, approximation allowances and thresholds. Historical evidence remains valid for its recorded source; it does not qualify changed code. A new complete moderate checkpoint is separately required before production integration. No registered recovery fit, tuning or recovery restart is permitted in these children.

Use a fresh implementation author for this bounded task, atomic commits and focused checks; pause before independent review. If a proposed implementation will not fit one worker context, split before editing. Root owns the final short-suite gate under the unchanged 1200-second cap and stacked PR publication. Long qualification programs remain separate from the default short suite and from registered recovery fits.

## Future work

The moderate qualification ticket measures the complete generalized solver and its storage before cdhd-hzw0 and then cff-7sw5 can complete.

Direct dependent tickets: cdic-w79t. Parent checkpoint: cdhd-hzw0; production integration: cff-7sw5; registered recovery: cff-x0if.

## Definitions

- CGLS: bounded conjugate-gradient least squares using derivative-vector products.
- N/M: unknown-coordinate and residual dimensions.
- Generated start: a separately validated feasible initializer output.
- Explicit start: the exact supplied coordinate evaluated without hidden replacement.
- JIT/vmap: JAX compiled and batched execution.
- FE: free energy.

- D12: the accepted hybrid-regime-ec-v1 approximation in docs/DERIVATIONS.md.
- Scope: the variables represented together in a regional belief.
- Proper belief: a normalized distribution with valid active probabilities and covariance.
- Certificate: an explicit numerical residual/energy/properness check required for a convergence claim.
- Primal/dual: the owned variational objective and the separately evaluated dual quantity; residuals are neither.
- Source inventory: exact code/configuration identities for the producer of a measured result.
- Prospective descriptor: input/output and storage metadata fixed before its qualification run.


