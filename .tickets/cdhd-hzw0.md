---
id: cdhd-hzw0
status: open
deps: [cdhd-mfc9]
links: []
created: 2026-09-15T08:02:15Z
type: epic
priority: 2
assignee: coopwilliams
parent: cff-f3jz
tags: [hybrid, numerics, qualification]
---
# Qualify the complete D12 solver for declared graph incidence

## Background in Plain English — WHY

The accepted cff-a3wz checkpoint proves complete message-plus-auxiliary solves for the tested homogeneous chains. The declared D12 model contract also admits branching connections, unary-only components, and unary factors on an intermediate recurrence/dynamics boundary. A bounded interface audit found that the qualified numerical evaluator cannot represent those declarations by changing compiler indexes alone. Production integration must consume an evaluator and solver that have actually been qualified for their declared inputs.

This prerequisite extends the early numerical checkpoint before cff-7sw5 publishes real component plans. Existing mathematical semantics and declaration admission remain authoritative. The work completes missing numerical interfaces and qualification; it does not revise D12, narrow the declared scope or weaken the recovery gate.

## Outcomes / Acceptance Criteria — WHAT

1. Bounded prerequisite tickets separately own the general incidence/input contract, numerical evaluator and derivative implementation, reuse of the existing iterative policy with tiny independent qualification, and moderate float32 qualification/storage evidence. Split each before assigning it if its scope exceeds one worker context. This epic is not one implementation assignment.
2. Every admitted recurrence, dynamics, unary and identity-adapter incidence has an explicit representation and objective owner. Chain elimination is used only where D12.3 establishes a lossless mapping. Branching, intermediate unary factors and unary-only components are represented without invented transitions or omitted ports.
3. The complete message-plus-auxiliary solve preserves the existing numerical policy, properness rules, finite-energy convergence signal, truthful terminal statuses, gauge removal and bounded matrix-free storage. Any necessary shared evaluator abstraction gets its own review and qualification; a generic wrapper is not considered qualified just because its old chain delegate was qualified.
4. Independent references and tiny checks cover the newly implemented incidence and adapter equations, objective counting, learning statistics, derivative actions and rejected/exhausted cases. Existing approximation allowances and numerical thresholds are retained where applicable; no universal exactness or convergence guarantee is added.
5. Prospectively specified moderate float32 cases exercise complete solves, initialization sensitivity, proper beliefs, finite free energy, learning statistics and measured iteration storage. Cases distinguish held exterior cavities from a coupled exterior solve. The coupled ordinary objective and adapter stationarity receive the applicable qualification before public activation; cff-ulbz retains its separate exterior-mixture contract.
6. Compilation evidence separates lower/compile time, synchronized execution, logical array storage, compiled memory estimates and process high-water memory. Report measured reuse and limitations honestly. Ordinary models remain outside D12 compilation. No cheap D12 mode is implied by omitting required outputs or certificates.
7. All historical source inventories and accepted evidence remain immutable and truthfully scoped. New or changed numerical producers receive complete explicit source inventories and appropriate new prospective qualification; naming a module outside an old glob must not bypass this requirement.
8. All three registered cff-x0if recovery fits remain unused throughout this prerequisite. If ordinary diagnosis and implementation fixes establish a mathematical/scope/approximation/threshold decision or unresolved numerical inability to meet requirements, root presents evidence and a recommended next step to Cooper.
9. Applicable architecture and duplicated numerical surfaces are specified in each child before implementation. Each code child passes focused checks, independent review after the author pauses, and the applicable final short-suite gate under 1200 seconds. The epic closes only when numerical and storage evidence satisfy its children and handoff to cff-7sw5/idhk/f3jz is explicit.

## Methodological Suggestions — HOW

Start with immutable graph-incidence and evaluator-input contracts, preserving existing chain code while mapping its exact applicability. Then split the numerical work using that concrete interface. Reuse local kernels and the existing bounded iterative policy through a typed evaluator boundary rather than copying solver policy. Another sound method preserving these outcomes is welcome.

Hostile inputs required in child planning: exact sparse/zero support and singleton labels; signed/vacuous cavities and improper tilted beliefs; native float32; independently enlarged incidence, cardinality and dimension. Branching changes degree-based entropy corrections. Empty support must fail; unknown storage must remain unavailable rather than zero. Numerical children cover non-finite residual/FE, globalization and changed support explicitly.

## Future work

cff-7sw5 retains real parameter/support resolution, exclusive assembly and fixed-parameter bundle reconstruction. cff-idhk retains one accepted runtime coordinate and transport; cff-brcf retains production objective/readouts; cff-f3jz activates public engine execution. cff-czdq/ulbz/kul2 retain learning, exterior mixture and composition gates. cff-x0if remains the separate registered source-recovery gate.

## Definitions

- D12: hybrid-regime-ec-v1, the accepted approximation in docs/DERIVATIONS.md.
- Incidence: a named connection between a region and one of its separator ports.
- Evaluator: the numerical function producing regional/separator beliefs, stationarity residuals, energy terms, auxiliary updates and certificates for a coordinate.
- Gauge: a redundant common message constant that does not change normalized beliefs.
- Auxiliary: the squared Jaakkola-Jordan logistic-bound coordinate, solved jointly with messages.
- Matrix-free: derivative-vector products without storing a dense trajectory Jacobian.
- Proper belief: a normalized distribution with valid active probabilities and positive-definite covariance.
- FE: free energy, separate from a residual or dual slope.
- Float32: 32-bit floating-point arithmetic, JAX's default precision.
- Adapter: a named equality region between a hybrid separator and an ordinary categorical or Gaussian copy.



## Notes

**2026-09-15T08:03:56Z**

Added cdhd-hzw0 as the explicit general-incidence numerical checkpoint before cff-7sw5 publishes real plans. Its first bounded implementation is cdhd-mfc9, depending on reviewed discovery and accepted A3WZ. The complete source-backed question/answer/confidence audit is recorded in cdhd-mfc9. Existing A3WZ qualifies tested homogeneous chains; branching, unary-only, intermediate H unary and live adapter interfaces require additional implementation and evidence. All original math, declaration scope, approximation and qualification thresholds remain required. IDHK/F3JZ will consume the appropriately qualified generalized interface through the same numerical policy, rather than assuming the unchanged chain API already represents every declaration. This refines the earlier chain-interface handoff; it does not invalidate or rewrite historical A3 evidence.

Legend: prerequisite --> dependent. Companion built from tk dep tree --full cff-7sw5, tk ready and tk blocked; original direct dependencies are retained. No cycles.

[x] A3WZ + [x] K6G5 + [x] cdd-ziiw
      +--> [~] cdhd-fyq6 (reviewed; final gate pending) --+
      +--> [~] cdhd-0ios (reviewed; final gate pending) -+--> [~] cdpf-gzix
[x] A3WZ -----------------------------------------------------+      |
                                                             +--> [ ] cdhd-mfc9
                                                                   |
                                         [ ] cdhd-hzw0 (additional bounded
                                             evaluator/qualification children required)
                                                                   |
[x] OCXL + retained dependencies -----------------------------> [ ] 7SW5
                                                                   |
                                                               [ ] IDHK --> [ ] BRCF --> [ ] F3JZ
                                                                                           /   \
                                                                                     [ ] CZDQ [ ] ULBZ
                                                                                           \   /
                                                                                         [ ] KUL2
                                                                                             |
                                                                                         [ ] X0IF --> [ ] ZOV5

Ready D12 roots are the two reviewed repairs awaiting the combined final gate; cdpf-gzix is their immediate dependent. cdhd-mfc9 is next after discovery closure. cdh-kvy5 remains an independent nonblocking research/audit branch. Implementation remains sequential with a fresh author per bounded ticket; the general checkpoint epic is not one worker assignment. No registered source-recovery fit has run.
