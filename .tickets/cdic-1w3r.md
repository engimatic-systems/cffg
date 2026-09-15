---
id: cdic-1w3r
status: open
deps: [cdic-ikhw]
links: []
created: 2026-09-15T09:02:42Z
type: task
priority: 2
assignee: coopwilliams
parent: cdhd-hzw0
tags: [hybrid, numerics, qualification]
---
# Expose same-label dynamics potentials, projections and stable energy

## Background in Plain English — WHY

D12 dynamics uses D(j,x,y), with one destination label shared by its two hybrid ports. Existing grouped helpers build two label axes and their marginal helper collapses a pair of labels. The affine Gaussian coefficients and stable whitened energy already exist, but general declared incidence needs a direct same-label local interface.

## Outcomes / Acceptance Criteria — WHAT

1. A standalone destination-selected dynamics potential preserves its Gaussian normalization, original matrix/bias/noise and fixed origins on the joint [x;y] continuous scope with one j axis.
2. Cavity assembly matches the same original label at both ports; it does not form a false outer product of labels. Local normalization, status and signed outgoing quotients use the existing labeled-Gaussian semantics.
3. Continuous-block projections preserve conditional means, covariances and correction words without collapsing the only label axis.
4. Stable dynamics expectation uses original channel parameters and retained moments rather than subtracting expanded grouped potentials. Existing grouped callers share the same mathematical owner where extraction applies.
5. Independent Gaussian/reference cases and D12.3 feasible-lift parity cover energy, entropy, projections and derivatives, including sparse support, vacuous/signed cavities, improper tilts, native float32 and uncomfortable d/K.
6. Any changed protected producer receives truthful new tiny source evidence and prospective descriptors; prior accepted artifacts remain immutable and source-scoped.
7. Canonical docs, focused checks, independent review and one passing final short suite complete the ticket without activating production execution.

## Methodological Suggestions — HOW

Start from normalized affine coefficients in hybrid_separator.transition, the channel expectation in transition_expectation, and Natural/Belief projection primitives. D12.2-.3 are the existing derivation; avoid the zero-recurrence or potential-subtraction tricks used by older mathematical tests as a production energy path.
Before implementation, record the applicable Derived component coordinate and Runtime numerical or status invariant change-path rows and every affected pair. Unbound numerical host/JIT and single/vmap behavior is in scope where the kernel/evaluator is callable; production synchronous/scheduled, State/lifecycle, ledger/readout activation, serialized learned state and public bundles remain downstream and guarded. Never infer new production support from a local test. Keep signed messages, literal inactive log-zeros, dtype-scaled guards, finite-energy checks, explicit failure status and bounded storage. Sparse support, improper/vacuous cavities, native float32 and independently enlarged incidence/K/d need explicit outcomes. Another sound method preserving all outcomes is welcome.

Qualification source ownership: if old numerical producers change, create new truthful source-bound tiny checkpoint evidence and regenerate the applicable prospective abstract descriptors before the final gate. Include every new transitive numerical dependency in the producer inventory regardless of filename. Preserve historical tiny004 and A3WZ artifacts, specifications, numerical policy, approximation allowances and thresholds. Historical evidence remains valid for its recorded source; it does not qualify changed code. A new complete moderate checkpoint is separately required before production integration. No registered recovery fit, tuning or recovery restart is permitted in these children.

Use a fresh implementation author for this bounded task, atomic commits and focused checks; pause before independent review. If a proposed implementation will not fit one worker context, split before editing. Root owns the final short-suite gate under the unchanged 1200-second cap and stacked PR publication. Long qualification programs remain separate from the default short suite and from registered recovery fits.

## Future work

The general evaluator combines this API with the separate recurrence payload. It still owns graph cavities, objective counting, complete residuals and derivative assembly; solver and moderate qualification remain downstream.

Direct dependent tickets: cdic-7y77. Parent checkpoint: cdhd-hzw0; production integration: cff-7sw5; registered recovery: cff-x0if.

## Definitions

- D(j,x,y): dynamics belief with destination regime j and previous/next states x/y.
- Same-label: both endpoints share j, rather than carrying independently paired labels.
- Correction word: retained compensated numerical information used by existing covariance arithmetic.
- JIT/vmap: JAX compiled and batched execution.
- K/d: label count and state dimension.

- D12: the accepted hybrid-regime-ec-v1 approximation in docs/DERIVATIONS.md.
- Scope: the variables represented together in a regional belief.
- Proper belief: a normalized distribution with valid active probabilities and covariance.
- Certificate: an explicit numerical residual/energy/properness check required for a convergence claim.
- Primal/dual: the owned variational objective and the separately evaluated dual quantity; residuals are neither.
- Source inventory: exact code/configuration identities for the producer of a measured result.
- Prospective descriptor: input/output and storage metadata fixed before its qualification run.


