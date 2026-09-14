---
id: cff-idhk
status: open
deps: [cff-7sw5, cff-a3wz]
links: []
created: 2026-09-14T03:05:51Z
type: feature
priority: 2
assignee: coopwilliams
parent: cff-zov5
tags: [rslds, inference, hybrid]
---
# Store and transport one accepted hybrid component coordinate

## Background in Plain English — WHY

The joint solver qualified by cff-a3wz and the component plans from cff-7sw5 need a production storage contract before the engine can run them. Existing State and scheduling contracts explicitly transport ordinary categorical, Gaussian, and exponential-family arrays. Adding a separate hidden component belief would allow messages, objective values, and learning statistics to disagree.

This is the first bounded portion extracted from cff-f3jz. It establishes one accepted component coordinate and the mechanical transport needed by later objective and public-execution work.

## Outcomes / Acceptance Criteria — WHAT

1. A typed production adapter consumes the cff-7sw5 plan and unchanged cff-a3wz solver/result interface. It gives every component coordinate, boundary gather/scatter, auxiliary value, and terminal certificate an explicit owner.
2. Runtime state stores one accepted coordinate. Boundary messages and belief views are derived from it; temporary solver and differentiation buffers are not persistent competing authorities.
3. Initialization, copying/replacement, observation or parameter rebinding, and changed support consistently initialize or invalidate component coordinates and certificates. Required aggregate and tilted beliefs follow the qualified solver's feasible-start contract.
4. State, ExecutionDelta, schedule transport contracts, synchronous/scheduled array transport, and explicit Colony state forwarding agree. Host/JIT and single/vmap preserve array shapes, support, component identity, and certificate fields in applicable transport checks. A single ordinary-model regression covers each changed shared construction path.
5. Literal negative-infinity support is distinguished from invalid active arithmetic in shared host and compiled checks. Empty support and non-finite active coordinates cannot be disguised as finite ordinary sentinels or successful certificates.
6. Production estimator inventory accounts for stored component arrays and qualified bounded solver scratch. It includes auxiliary coordinates in any joint linear-solver vectors and does not claim dense trajectory storage or unbounded iteration history is acceptable.
7. Public D12 execution remains unavailable until cff-f3jz activates the complete objective and status path. An unbound declaration cannot silently run D9 or return placeholder D12 success.
8. The preimplementation handoff applies Derived component coordinate and Runtime numerical or status invariant and enumerates all duplicated transport/lifecycle surfaces. Sparse and changing support, vacuous/improper initialization, float32, and larger component inventories have explicit acceptance behavior; serialized learned-state payloads remain downstream.
9. ARCHITECTURE describes authority and invalidation rules. Focused lifecycle/transport/estimator tests and one final short-suite gate pass after the final relevant edit.

## Methodological Suggestions — HOW

These suggestions are optional; another sound method meeting the outcomes is welcome.

- Start from runtime_types.State, lifecycle.StateLifecycle, execution.contracts.ExecutionDelta, schedule_contracts, execution/transport.py, and estimate.py. Follow callers that explicitly reconstruct State, including Colony.
- Reuse the qualified solver through a typed adapter rather than copying its numerical policy. Keep public activation deferred as an explicit unsupported state.
- Classify GFE/Mu3 shared state forwarding even though their numerical equations are outside D12.

## Future work

cff-brcf assembles objective/readout ownership from this accepted coordinate. cff-f3jz then activates public scheduling, status and batching. cff-czdq owns learned statistics installation and current/legacy persistence, cff-ulbz owns exterior mixtures, and cff-kul2 owns public composition. Registered recovery remains cff-x0if.

## Definitions

- D12: the hybrid-regime-ec-v1 approximation in docs/DERIVATIONS.md.
- EC: expectation consistency, matching declared moments across neighboring regional beliefs.
- Accepted coordinate: the single stored set of component messages, auxiliary values, and associated terminal certificates used to reconstruct beliefs.
- Auxiliary xi: the squared logistic-bound parameter c*c.
- Compiler plan: derived indexes and ownership reconstructed from the serialized template and data.
- JIT/vmap: JAX compiled and batched execution.
- Cavity: incoming messages other than the outgoing incidence being updated.
- Free energy: the stated D12 objective used to judge convergence and score inference.
- Adapter: an explicit equality region connecting a hybrid boundary to an ordinary variable copy.
- D9: the existing factor-local recurrence approximation.

