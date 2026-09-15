---
id: cdic-ikhw
status: in_progress
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

**2026-09-15T09:32:02Z**

Pre-edit author handoff and scope assessment: this fits one bounded extraction in hybrid_separator.py, with independent held-R checks in a new registered test file and canonical documentation edits. Planned APIs recurrence_auxiliary_statistics(R, weights, offsets, xi) and recurrence_expectation(R, weights, offsets, xi) consume the existing checked Belief without another source_region reduction. Auxiliary retains all tuple fields and gains an active mask property. Grouped auxiliary_statistics and transition_expectation delegate after source_region; full grouped covariance-correction validation remains intact. The stable recurrence expectation is moved directly from its original-parameter formula, never recovered by subtracting expanded potentials.

Applicable ARCHITECTURE change-path rows: Derived component coordinate, at the unbound R local-payload boundary, and Runtime numerical or status invariant, for local energy/statistic status propagation. Affected duplicated surfaces: original recurrence statistics versus grouped auxiliary wrapper; separate R energy versus grouped transition expectation; eager versus JIT; single versus vmap; retained origins versus physical statistic readouts; covariance high word versus optional correction. Independent NumPy held-belief references remain test-only. Production synchronous/scheduled/scan, State/lifecycle, compiler/arena activation, ledger/readouts/Learnables, serialization/checkpoint/bundles, ordinary exterior coupling, and solver policy are N/A in this extraction and remain downstream guarded work. No source-recovery fit is involved.

Hostile-input acceptance before implementation: sparse pair probabilities and exact log-zero labels retain original i/j positions and fixed origins; inactive source/stick suffixes have exact zero target/exposure/learning and stale xi has a nonzero residual. Proper tilted beliefs assembled from signed/vacuous Natural cavities are supported without clamping; improper/empty/nonfinite checked beliefs propagate explicit nonzero status and NaN numerical payload. Parameters retain existing finite/nonnegative-xi rules. Native float32 uses the existing dtype-scaled guards and compensated affine operations, with no promotion. Independently enlarge K and d at least tenfold and check bounded O(K squared times d squared) local storage. Held-R references and derivative checks include offset cancellation, covariance corrections and original-label learning identities; grouped complete outputs must agree.

Qualification boundary: hybrid_separator.py already belongs to the complete current tiny SOURCE inventory, and this phase adds no new transitive numerical module. Its protected hash will change, so historical tiny004/current prospective checks will intentionally be stale at the author pause. Historical tiny004, A3WZ reports/specifications and their source identities remain immutable. After review, new truthful tiny source-bound evidence and applicable prospective descriptors remain required before the final short-suite gate. This first phase runs focused checks only, commits, and ends cleanly; root owns review, final gate under the unchanged 1200-second cap, closure and PR. All three registered recovery fits remain unused.

**2026-09-15T09:59:24Z**

Implementation handoff for the committed independent-review pause: recurrence_auxiliary_statistics now consumes the actual checked R(i,j,x) belief; recurrence_expectation owns the original-parameter stable JJ log-factor expectation. Auxiliary adds an active property without changing its tuple fields. The grouped statistics and transition-energy interfaces delegate through source_region. Both preserve complete grouped covariance-correction status before reduction, including an invalid destination correction's previous INVALID|NONFINITE energy outcome. The statistics and stable recurrence formula each have one JAX implementation. No dynamics kernel, general evaluator, exterior coupling, policy, production runtime or approximation changed.

The new registered test_recurrence_payload.py contains 19 default held-belief cases. Its independent pair/stick enumeration checks energy, targets, exposure and learning; derivatives check original weight/offset learning, the xi=0 equation and moment JVP/VJP actions. It covers original sparse label positions, inactive/stale xi, retained pair origins, covariance low-word sensitivity, improper/empty/nonfinite inputs, proper tilts with signed/vacuous cavities, native float32, and independently enlarged K=20,d=1 and K=2,d=10. ARCHITECTURE records API ownership, DERIVATIONS anchors the existing D12.8 equations to these checks, and LIMITATIONS distinguishes focused coverage from source-bound qualification.

Final focused validation after the last code/test edit: uv run pytest -q tests/test_recurrence_payload.py tests/test_hybrid_separator_kernels.py tests/test_hybrid_separator_numerics.py tests/test_hybrid_centering.py tests/test_hybrid_precision.py tests/test_dashboard.py --tb=short -> 181 passed in 1086.87 seconds (18:06). Log: /tmp/cdic-ikhw-focused-final.log. This is a focused result, not a fresh full short suite. The new file includes compiled local, derivative and scale cases, so its default-suite cost may consume material headroom under the unchanged 1200-second gate. This combined run did not collect per-file timing; no selection was weakened or moved to slow to meet the cap. Root owns the measured final gate and any required performance repair.

Additional numerical parity: uv run python /tmp/cdic-ikhw-grouped-baseline.py loaded the unchanged hybrid_separator.py from base f17e9fa34d264dfbf14e06bcde608ab524ac1f54 and compared complete grouped statistics, transition expectations and three-site evaluator payloads in float32/float64. All 18 comparisons passed, including valid held sparse beliefs, invalid destination correction, negative xi and nonfinite offsets; shapes, dtypes, finite/nonfinite masks and exact status bits agreed, with numerical values within 8 dtype eps. The durable independent mathematical oracles remain in the registered tests; the temporary comparison script/log are /tmp/cdic-ikhw-grouped-baseline.py and /tmp/cdic-ikhw-grouped-baseline.log. git diff --check and tk dep cycle passed.

Qualification is intentionally pending at this author pause. The changed numerical producer hybrid_separator.py has SHA-256 46f6802f35b0386be5c720063d702e3d558f79a0e3255ffaa53a1290fd8b2d09 and is already in current tiny SOURCE; no new transitive numerical dependency was introduced. Historical tiny004/current source checks and applicable prospective descriptors are stale for this changed producer. No historical hash, specification, qualification artifact or selector was edited; no qualification program ran. Root's operational research identified a separate bounded version-selection harness prerequisite before new prospective generation, because the current generator/spec loader targets accepted artifacts. Root will track that prerequisite after this clean pause. After committed-code review and that harness work, new truthful source-bound tiny evidence and prospective descriptors still precede the one final combined short suite. Historical tiny004/A3 evidence remains valid only for its recorded source/layouts. All three registered recovery fits remain unused. Ticket stays in_progress; root owns independent review, final gate, closure and stacked PR publication.
