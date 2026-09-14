---
id: cdh-m75a
status: open
deps: [cff-j191]
links: [cdc-ztuf]
created: 2026-09-14T23:24:33Z
type: task
priority: 1
assignee: coopwilliams
parent: cff-zov5
tags: [hybrid, numerical, qualification]
---
# Bind moderate D12 outputs to an independent fixed-coordinate reference

## Background in Plain English — WHY

The moderate D12 qualification harness reconstructs residuals from saved outputs, but some saved outputs can still change together and pass. A changed message and target can agree without being the result of the fixed model. Adding the same constant to both energies also preserves their difference. Before qualification, an independent calculation must tie beliefs, targets and absolute energies to the actual model inputs and returned message/auxiliary coordinate. This work is split from cdc-ztuf before another large implementation grows inside that ticket.

## Outcomes / Acceptance Criteria — WHAT

1. A bounded independent NumPy evaluator accepts the trusted frozen native-float32 model inputs and supplied coordinate for the zero-origin T32/T100, K4, d2 cases. It computes local regional/separator beliefs, outgoing targets and both absolute energies without using the candidate's saved targets, beliefs or energies as reference inputs.
2. The moderate evidence validator compares the saved returned point against those independent outputs, including regional/separator normalizers and the defined energy identities. It preserves existing local arithmetic and absolute-energy allowances, covariance word semantics and D12 approximation; it does not require temporal exactness or equal solutions from different starts.
3. Valid independent analytic/reduction controls pass. Paired message/target changes, equal shifts of both energies and changed normalizers fail for their mathematical inconsistency, including aggregate reports with refreshed hashes. Controls must already satisfy every other validation rule.
4. The reference uses local O(T*K*K*d**3) work without a global Jacobian, temporal-path enumeration or optimization. Measure a small fixed-coordinate reference evaluation at both moderate horizons to document its validation cost; this is not an extra optimizer or recovery fit.
5. Record the source/provenance dependencies of the reference in the prospective harness contract. Focused tests and independent review pass; pause for the supervisor's combined final short-suite gate before closure. Historical evidence is preserved.

## Methodological Suggestions — HOW

The independent src/cffg/experiments/hybrid_separator.py already has general-dimensional transition and arbitrary-length evaluate functions. A small adapter can build factors with c=sqrt(xi), independently decode the declared candidate coordinate schema, and promote identical stored float32 inputs to NumPy float64 for the reference. Compare raw outputs rather than borrowing a differently scaled reference certificate. Do not remove the tiny T2/T3 guard in tests/hybrid_joint_reference.py. Another independent local method is acceptable if it meets the same boundaries. Coordinate packing and inactive constants differ between existing references and the candidate; check them explicitly.

Architecture change paths: Derived component coordinate and Runtime numerical or status invariant. Affected surfaces: saved initial/terminal point, local regional/separator projection, outgoing targets, energy and report rejection. Host/JIT and single/vmap candidate execution are consumed through the same saved output contract; production compiler/scheduling/lifecycle/serialization/public learning are N/A. Exact sparse support remains distinct from finite-log probability underflow; vacuous messages are valid only with proper resulting beliefs. Native float32 inputs and 10–100x tiny-chain horizon are in scope; arbitrary nonzero coordinate origins and public reference scope expansion are deferred, not silently accepted.

## Future work

cdc-ztuf integrates the reviewed setup and owns source-matched tiny evidence plus the shared final gate. cdh-urn0 independently repairs the discovered covariance discrepancy. cff-a3wz runs the twelve frozen moderate optimizers and storage measurements after both prerequisites. Production integration and cff-x0if's three registered recovery fits remain downstream. No additional recovery fit, threshold, supported-model or approximation change is authorized.

## Definitions

D12 is the hybrid message and auxiliary-variable numerical contract. A returned point is the solver's final message/auxiliary coordinate. A target is a local outgoing message computed from that point. A normalizer is the logarithm of the integral of an unnormalized belief. Absolute energy checks each energy itself rather than only their difference. T is the number of time points, K the label count and d the continuous dimension. A zero origin means the stored local coordinate center is zero. NumPy supplies an implementation independent of the JAX candidate.



## Notes

**2026-09-14T23:24:33Z**

Independent planning research at published d22a43a. Question: can existing independent NumPy code evaluate moderate D12 local outputs and absolute energies at supplied messages/xi, without optimizing or enumerating temporal paths? Answer: yes. src/cffg/experiments/hybrid_separator.py transition(weights,offsets,c,matrix,bias,noise) supports general K>=2,d>=1; evaluate(factors,sites,messages) supports arbitrary nonempty chain length and returns regions, separators, outgoing targets, primal FE and dual FE in O(T*K*K*d**3) local work. Its only mathematical dependency is the independent NumPy experiment regime_pair. chain_energy includes regional factor/entropy terms, internal separator corrections and unary-site expectations; dual uses regional/separator log normalizers.

Smallest bounded adapter for the frozen zero-origin T32/T100,K4,d2 fixtures: rebuild the trusted model inputs and supplied coordinate from identical stored float32 values promoted to NumPy float64, construct each transition with c=sqrt(xi), build unary and directed natural messages, call evaluate once, and compare raw independently computed regional/separator outputs, outgoing targets and BOTH absolute energy values. Saved targets/beliefs/energies must not be reference inputs. This is fixed-coordinate arithmetic validation, not a new optimization, temporal-exactness claim, public model-scope expansion or recovery fit. Use unchanged local arithmetic and absolute energy allowances; report covariance representation/once-rounded distinctions honestly.

Reference boundaries: tests/hybrid_joint_reference.py has general-dimensional Model/factor/lift/normalize/project/quotient/entropy/expectation/statistics helpers and fixed-origin support, but its evaluate->validate explicitly restricts T to2/3; load_case adapts scalar fixtures. Do not silently bypass/remove that guard. Its coordinate packing differs from the candidate and inactive constants are zero rather than native-inf; decode the declared candidate schema independently. Its residual scales also differ, so reuse raw outputs rather than its certificate dictionary. tests/hybrid_separator_reference.py enumeration is K2,T2..6 with scalar states, quadrature T<=3; its scalar helpers are unsuitable. tests/hybrid_joint_solver_reference.py dense optimizer/finite-difference Jacobian/stopping propagation remains tiny-only and is unnecessary here.

The older experiment uses world natural coordinates/moments without retained origins/low words. This is suitable for the registered zero-origin moderate cases, not an arbitrary-origin validator. If later scope requires origins, compose the newer reference's reusable local helpers behind a separate explicit fixed-coordinate entry; do not change the tiny API guard. Confidence high for APIs/restrictions/independence; actual moderate reference runtime and numerical parity have not been measured. Sources: experiments/hybrid_separator.py:160,207,222; experiments/regime_pair.py:10; tests/hybrid_joint_reference.py:92,114,168,345; tests/hybrid_separator_reference.py:14. No edits, tests, compilation or numerical evaluation occurred.

Suggested fresh bounded owner: implement the test-only fixed-coordinate adapter and integrate source-point comparisons into moderate saved-evidence validation, with analytic/reduction controls and paired target/energy corruption regressions. No moderate optimizer needed. Root will decide the ticket split after completed re-review; the current author should not grow another context-sized implementation implicitly.
