---
id: cff-ulbz
status: closed
deps: [cff-ocxl, cff-f3jz]
links: []
created: 2026-09-13T18:07:15Z
type: feature
priority: 2
assignee: coopwilliams
parent: cff-zov5
tags: [rslds, inference, hybrid, gaussian]
---
# Add moment-consistent Gaussian mixture boundaries for recurrent composition

## Background in Plain English — WHY

D11 evaluated a pair-conditioned Gaussian recurrence region with explicit
first/second-moment consistency, then withheld implementation authorization
because separate boundaries lose external source/state association. An rSLDS
also contains switching Gaussian
dynamics. The current D3 Gaussian mixture rule collapses an outgoing mixture
before including the receiving cavity, while its node belief uses both
Gaussian cavities. Those operations do not generally satisfy D11's moment
constraints. Merely connecting the existing message arrays would therefore
not make the complete model stationary for D11.7.

D12 accepts hybrid-regime-ec-v1 with explicit ordinary adapters. Internal
destination-selected affine dynamics is eliminated under D12.3, so its D3
port is unnecessary. Exterior mixture factors still need the EC binding:
keep their existing mixture region scope and add an explicitly selected EC
binding. It projects the full tilted marginal and then divides by the
receiving cavity. D11.14 derives this local companion, but it does not itself
solve the source-association obstruction. The existing D3 binding retains
its declared behavior. No production implementation starts before a positive
mathematical decision and its required protocol dependency.

## Outcomes / Acceptance Criteria — WHAT

1. The binding follows D12.5 for exterior mixture ports and depends on
   cff-f3jz (declared region execution), transitively including cff-k6g5
   declarations and cff-7sw5 plans. Internal eliminated dynamics do not gain
   a redundant mixture port. The handoff inventories New factor or rule family,
   New IR field, Runtime numerical or status invariant, and Learnable or
   checkpoint feature, using that accepted generic region protocol.
   Host/JIT, flooding/scheduled, single/vmap, and current/legacy persisted
   identity pairs are explicitly classified.
2. Serialized local moment constraints select a named EC binding for
   gaussian_mixture and gaussian_mixture_linear, including composition,
   hashing, checkpoint identity, and validation. Names/model classes never
   select the rule and unsupported constraints never silently fall back.
3. The factor belief is the full tilted labeled Gaussian mixture from D3.2.
   Each Gaussian output follows D11.14's projected-marginal/cavity quotient;
   categorical output, all learning moments, and exact labeled node entropy
   use that same belief.
4. The complete node/edge energy and stationary moment residual match an
   independent tiny integration reference. The two-component N(+/-2,1)
   mixture with N(0,1) receiving cavity yields edge variance 3/2 and outgoing
   precision -1/3, rather than D3's edge variance 5/6.
5. Ordinary Gaussian and categorical attachments, equality branches,
   renaming, and factor reordering preserve explicit selection. A composed
   recurrence/dynamics fixture satisfies the shared D11.13 constraints.
6. Literal zero selector support gives exact zero weighted statistics;
   empty support, nonintegrable active tilted blocks, non-finite FE/residual,
   and unresolved precision fail loudly. Vacuous/improper cavities are
   accepted only when the required tilted beliefs are proper. Signed
   messages are retained without clamping.
7. Float32 and a 10–100x horizon or regime case obey dtype-scaled guards and
   O(T K (d_in+d_out)^3) local-evaluation work, with polynomial local storage.
   Iterative convergence requires finite FE stability and stationary residual;
   exhausted budgets and divergence are distinct.
8. All changed execution surfaces agree within declared dtype tolerances.
   Learning and persistence use versioned semantics and reject or migrate
   old shape-compatible state explicitly.
9. ARCHITECTURE, DERIVATIONS, LIMITATIONS, and README explain selection,
   ownership, signed messages, the absence of a global evidence bound, and
   the remaining approximation. The full short-suite gate passes.

## Methodological Suggestions — HOW

These are optional; another method satisfying the outcomes is acceptable.

- Reuse the full tilted Gaussian blocks and labeled node-energy algebra.
- Bind through the accepted local region protocol rather than model dispatch.
- Use the D11.14 counterexample before an end-to-end learned trajectory.

## Future work

cff-x0if qualifies source-specific recovery only after the cff-ocxl decision,
its resulting implementation, and any required companion binding land.
Whole-trajectory fusion, GFE/P-substitution, and
empirical recovery runs are outside this ticket.

## Definitions

- EC: expectation consistency, meaning neighboring beliefs share declared
  sufficient-statistic moments rather than their full densities.
- Cavity: the product of messages arriving from other incident factors.
- Tilted belief: the normalized factor times all incoming cavities.
- Natural quotient: subtraction of Gaussian precision and information
  vector coordinates, including signed precision when required.
- Region: the normalized joint belief whose energy and entropy the factor owns.



## Notes

**2026-09-13T18:13:29Z**

Dependency correction after the decisive D11.15 falsifier: cff-ocxl is now a mathematical gate, not an accepted implementation protocol. This ticket remains blocked and conditional on its decision; no production EC binding is authorized yet. D11.14 fixes the projection-order mismatch but does not transmit E[1_i x] across separate singleton boundaries. Reassess the binding and its dependencies after the hybrid separator/grouping is derived.

**2026-09-14T01:15:18Z**

D12.5 confirms this companion for ordinary exterior mixture ports after cff-f3jz implements the declared region protocol and execution. D12.3 eliminates the internal destination-selected affine switching channel, so no redundant internal D3 port is needed. The Gaussian identity adapter test reproduces variance 3/2 and signed quotient precision -1/3 from the full tilt. cff-kul2 consumes this binding in its public composition gate; existing D3 semantics remain separate.

**2026-09-15T13:20:19Z**

Closed as not pursuing at Cooper's direction with the D12 recovery path; see cff-ocxl. The derivation and preserved local-kernel branches establish the full-tilt projection and signed quotient needed by an ordinary Gaussian mixture boundary. A complete coupled solve with ordinary neighbors and a production binding remained unfinished. Existing D3 semantics are not replaced by this closure, and no composition or integration acceptance is claimed.
