---
id: cdic-ikhw
status: open
deps: [cdhd-mfc9]
links: []
created: 2026-09-15T09:02:42Z
type: task
priority: 2
assignee: coopwilliams
parent: cdhd-hzw0
tags: [hybrid, numerics, qualification]
---
# Expose recurrence-scope auxiliary statistics and stable energy

## Background in Plain English — WHY

The general D12 interface names separate recurrence beliefs R(i,j,x). Existing recurrence potential and projection primitives already represent that scope, but auxiliary_statistics first halves a grouped [x;y] belief and stable recurrence energy is embedded in a grouped transition helper. Extracting those existing operations gives the general evaluator a lawful local payload without changing the approximation.

## Outcomes / Acceptance Criteria — WHAT

1. A named local API consumes an actual proper R(i,j,x) belief and original recurrence parameters, preserving source/destination label identities, fixed support and origins.
2. The API returns the existing squared-auxiliary targets, at-risk masks, expected learning statistics and status without passing an already reduced belief through source_region.
3. Stable recurrence energy is computed from original parameters and retained moments, preserving canceled terms and covariance corrections. Expanded-potential subtraction is not its numerical authority.
4. Existing grouped callers delegate through the new primitive while retaining their public numerical interface, arithmetic semantics and complete outputs. No second implementation of the recurrence statistic or energy formula remains.
5. Independent held-belief references and derivative checks cover sparse/zero support, inactive sticks, signed/vacuous input handling, float32 and larger K/d; grouped parity and D12 equations remain explicit anchors.
6. Applicable new tiny source-bound qualification and prospective descriptors are recorded after reviewed protected-source changes, with all historical artifacts preserved. No new moderate or source-recovery claim is made.
7. Canonical ownership documentation, focused checks, independent review and one passing final short suite complete the ticket.

## Methodological Suggestions — HOW

Start from hybrid_separator.recurrence_potential, source_region, auxiliary_statistics and the recurrence portion of transition_expectation. Reuse Natural/Belief and compensated moment operations. This is extraction of existing mathematics, not a new variational family.
Before implementation, record the applicable Derived component coordinate and Runtime numerical or status invariant change-path rows and every affected pair. Unbound numerical host/JIT and single/vmap behavior is in scope where the kernel/evaluator is callable; production synchronous/scheduled, State/lifecycle, ledger/readout activation, serialized learned state and public bundles remain downstream and guarded. Never infer new production support from a local test. Keep signed messages, literal inactive log-zeros, dtype-scaled guards, finite-energy checks, explicit failure status and bounded storage. Sparse support, improper/vacuous cavities, native float32 and independently enlarged incidence/K/d need explicit outcomes. Another sound method preserving all outcomes is welcome.

Qualification source ownership: if old numerical producers change, create new truthful source-bound tiny checkpoint evidence and regenerate the applicable prospective abstract descriptors before the final gate. Include every new transitive numerical dependency in the producer inventory regardless of filename. Preserve historical tiny004 and A3WZ artifacts, specifications, numerical policy, approximation allowances and thresholds. Historical evidence remains valid for its recorded source; it does not qualify changed code. A new complete moderate checkpoint is separately required before production integration. No registered recovery fit, tuning or recovery restart is permitted in these children.

Use a fresh implementation author for this bounded task, atomic commits and focused checks; pause before independent review. If a proposed implementation will not fit one worker context, split before editing. Root owns the final short-suite gate under the unchanged 1200-second cap and stacked PR publication. Long qualification programs remain separate from the default short suite and from registered recovery fits.

## Future work

Same-label dynamics and the general incidence evaluator consume this local API. Shared iterative-policy binding and moderate graph qualification remain later prerequisites of cdhd-hzw0.

Direct dependent tickets: cdic-1w3r, cdic-7y77. Parent checkpoint: cdhd-hzw0; production integration: cff-7sw5; registered recovery: cff-x0if.

## Definitions

- R(i,j,x): recurrence belief over source label i, destination label j and the previous continuous state x.
- Auxiliary: the squared Jaakkola-Jordan logistic-bound coordinate solved jointly with messages.
- At-risk stick: a logistic gate reached by at least one supported destination label.
- JIT/vmap: JAX compiled and batched execution.
- FE: free energy.
- K/d: label count and continuous state dimension.

- D12: the accepted hybrid-regime-ec-v1 approximation in docs/DERIVATIONS.md.
- Scope: the variables represented together in a regional belief.
- Proper belief: a normalized distribution with valid active probabilities and covariance.
- Certificate: an explicit numerical residual/energy/properness check required for a convergence claim.
- Primal/dual: the owned variational objective and the separately evaluated dual quantity; residuals are neither.
- Source inventory: exact code/configuration identities for the producer of a measured result.
- Prospective descriptor: input/output and storage metadata fixed before its qualification run.



## Notes

**2026-09-15T09:29:24Z**

Published as PR154, https://github.com/engimatic-systems/cffg/pull/154, stacked on PR153 with base cdpf-gzix-hybrid-discovery and head cdhd-mfc9-incidence-contract. cdhd-mfc9 is closed. Final code gate at942f8c8 passed3009tests in1152.682seconds under the unchanged1200-second cap; later commits only record tickets/runtime/publication evidence. Historical numerical producers and accepted artifacts remain unchanged. No unchanged suite is required for publication.

Next ready D12 implementation root is cdic-ikhw (separate R auxiliary/statistic and stable-energy extraction), using the published PR154 branch as its intended base. A fresh author must update that base before branching, complete the pre-edit architecture/hostile-input handoff and assess scope before code. The extraction may change protected numerical producers; plan truthful new current tiny evidence and regenerated prospective descriptors rather than editing historical hashes or claiming A3 qualifies changed code. Review the committed code after the author pauses before expensive complete qualification; then resume the same bounded author as appropriate for its qualification/report handoff. Root owns final short-suite gate and PR publication. Registered recovery data, exactly-three-fit allowance and all thresholds remain untouched.

The explicit numerical chain is cdic-ikhw -> cdic-1w3r -> cdic-7y77 -> cdic-3e79 -> cdic-nuel -> cdic-za1z -> cdic-w79t -> cdhd-hzw0 -> cff-7sw5, retaining additional direct dependencies. Ordinary-port definition and numerical coupling are deliberately separate. All three registered recovery fits remain unused. Current moderate evidence remains the accepted A3 homogeneous-chain checkpoint, not source recovery or newly generalized graph qualification.
