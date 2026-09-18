---
id: cff-2dzo
status: closed
deps: [cff-314c]
links: []
created: 2026-09-15T16:30:34Z
type: task
priority: 2
assignee: coopwilliams
parent: cff-o7hs
---
# Represent moment matching at named Gaussian factor endpoints

## Background in Plain English — WHY

Expectation propagation compares selected moments of a factor endpoint with those of its incident edge. A factor name alone cannot select the endpoint or the moments. This ticket gives cff-o7hs an explicit portable declaration before numerical rules execute it.

## Outcomes / Acceptance Criteria — WHAT

1. A moment-matching declaration names one factor, one incident structural role, and the Gaussian first/second-moment projection. Either endpoint of a degree-two edge can be selected independently.
2. Core validation rejects missing or invalid endpoint/projection details in new declarations; factor-only legacy payloads remain loadable for explicit migration but cannot compile or produce a diagram. Migration requires caller-supplied endpoint and projection.
3. JSON, structure hashing, edits, scoping, plate expansion, composition and bundle transport retain endpoint meaning without importing JAX. Diagrams place a moment-matching mark at the selected endpoint.
4. Compiler admission distinguishes a complete declaration from legacy syntax and admits it only when its executable factor binding declares the projection/method contract.
5. Focused tests cover distinct ends of one edge, malformed and legacy details, explicit migration, transport and executable rejection, and docs describe the supported schema.

## Methodological Suggestions — HOW

- Use New IR field and New factor or rule family in docs/ARCHITECTURE.md. Check core IR/validation/serialization/hash, edits, plates/composition, diagrams, compiler constraint admission, bundle transport and exports.
- Host/JIT, flooding/scheduled and single-instance/vmap are N/A for numerical execution in this schema ticket; they must still reject unsupported declarations consistently through shared compilation. Current/legacy decoding is affected and requires explicit migration/rejection tests.
- Sparse support, vacuous messages, float32 and numerical scale are N/A because this ticket performs no numerical updates. Validate multi-endpoint and repeated-template structure instead.

## Future work

- cff-o7hs supplies nonlinear Gaussian messages, site globalization, surrogate energy, and production execution using this schema; cff-1o72 supplies learned neural means.

## Definitions

- Endpoint: one named factor attachment to an edge.
- Projection: the selected sufficient statistics, here Gaussian first and second moments (equivalently mean and covariance).
- Expectation propagation: updating an approximate local factor by matching selected moments.
- Legacy payload: an older serialized declaration without endpoint/projection information.



## Notes

**2026-09-15T16:43:13Z**

Diagram research question: what marks a selected endpoint in Part I Figure 12? Answer: one outlined edge circle containing E, with the line dashed only between that edge bead and each selected factor-node bead. One selected end leaves the opposite half solid; two selections dash both halves. Source: notes/cffg-pt-1.pdf p.17, visually verified; notes/pt1.txt lines 1098–1112. Confidence: high. This is implemented in diagram endpoint metadata and the selected external/internal path segments, retaining the E bead at every compression level. Research performed by GPT-6 Astra xhigh and reviewed at top level.

**2026-09-15T16:48:05Z**

Implemented explicit Gaussian first/second-moment endpoint declarations, validation before serialization can discard duplicate keys, caller-supplied legacy migration, registered projection admission, and Figure 12 E beads with selected external/internal half-edge dashes. Analytic Gaussian prior/linear bindings declare this projection: Gaussian moments determine their full density, so their existing messages and energy remain exact. JSON/hash, edits, scoping/merging, plates, bundles, both endpoint selections, unsupported projections and legacy rejection have executable coverage. README, ARCHITECTURE and LIMITATIONS describe the delivered boundary.

Focused validation: 75 passed in 46.07s across endpoint schema, Gaussian kernels/schedules, registration, visualization and core imports; a final schema/serialization/diagram/layout/SVG/import selection passed 60 tests in 1.69s. The two dimension-100 Gaussian cases passed in the former selection before being marked slow under cff-gvwc. The parent implementation still owes its final short-suite gate. No nonlinear rule is advertised by this commit. Handoff: schema foundation complete; cff-o7hs continues configured mean functions and the accepted full-fixed-point/approximate-energy contract.
