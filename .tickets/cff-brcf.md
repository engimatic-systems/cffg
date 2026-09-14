---
id: cff-brcf
status: open
deps: [cff-idhk]
links: []
created: 2026-09-14T03:05:51Z
type: feature
priority: 2
assignee: coopwilliams
parent: cff-zov5
tags: [rslds, inference, hybrid]
---
# Assemble hybrid objective and internal readouts from the accepted coordinate

## Background in Plain English — WHY

D12 component messages are useful only if the objective and reported beliefs describe the same returned state. Existing ordinary ledgers independently count categorical and Gaussian edge entropies, and D9 recurrence readouts reconstruct beliefs from ordinary inputs. Reusing those calculations for a hybrid component would double-count energy or discard the source/state association.

This bounded prerequisite of cff-f3jz establishes complete energy ownership and read-only belief/statistic views before public execution is activated.

## Outcomes / Acceptance Criteria — WHAT

1. The production objective owns exactly the D12 regional factor energies, labeled regional entropies, hybrid separator corrections, explicit adapter entropies, and applicable ordinary exterior corrections declared by the compiler plan.
2. Ordinary categorical and Gaussian ledger paths exclude component-owned incidences. Every energy and entropy contribution has one declared owner, including interior and endpoint adapters; no unused D9 term remains in a D12 score.
3. Objective and internal readouts use the single accepted component coordinate from cff-idhk. They neither rerun a solver nor reconstruct D9 beliefs as a substitute.
4. At held accepted coordinates, the full objective agrees with the cff-a3wz independent tiny D12 reference and stationary dual under its declared arithmetic/stopping tolerances. The residual is never reported as free energy, and inconsistent/invalid states cannot receive a finite successful objective.
5. Internal readouts retain regime probabilities, conditional first/second and cross-time moments, source-conditioned learning increments, and approximation identity needed by inference qualification and cff-czdq. Learning-state mutation and checkpoint migration are excluded here.
6. Host/JIT, single/vmap, and synchronous/scheduled held-coordinate objective/readout paths agree where applicable. Sparse support, inactive xi, signed/vacuous cavities, non-finite states, float32, and moderate local inventories follow the qualified numerical contract. The known D12 temporal approximation falsifier remains disclosed.
7. Interior Gaussian and categorical attachments exercise ownership at their named adapter ports. The later cff-ulbz exterior-mixture binding is not assumed available, and unsupported associations do not silently collapse to singleton boundaries.
8. The preimplementation handoff applies Derived component coordinate, Runtime numerical or status invariant, and the readout surfaces of Learnable or checkpoint feature. It enumerates duplicated paths and marks current/legacy persistence and public solver activation N/A for this ticket.
9. ARCHITECTURE and D12 executable anchors describe objective/readout ownership. Focused ledger/readout/reference tests and one final short-suite gate pass after the final relevant edit.

## Methodological Suggestions — HOW

These suggestions are optional; another sound method meeting the outcomes is welcome.

- Review ledger.make_belief_energy, ledger_terms.gaussian_energy, readouts/recurrence.py, readouts/gaussian.py, readouts/views.py, and Engine's readout adapters.
- Test explicit ownership exclusions and independent accepted-coordinate calculations, including finite and invalid paths; do not create a second numerical solver for the ledger.

## Future work

cff-f3jz activates public schedules, solver status and batching using these objective/readout owners. cff-czdq installs/version-controls learning statistics and persisted state; cff-ulbz provides compatible exterior mixture projection; cff-kul2 qualifies public composition before cff-x0if runs registered recovery fits.

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

