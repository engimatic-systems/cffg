---
id: cff-x3a8
status: closed
deps: [cff-fely]
links: []
created: 2026-08-24T12:42:26Z
type: task
priority: 1
assignee: coopwilliams
parent: cff-3v7q
tags: [rslds, validation, stability, derivation]
---
# Derive a joint switching-dynamics stability intervention

## Background in Plain English — WHY

The completed classic D10 cohort did not find a recurrence prior that was
robust across every declared 10,000-step rollout. Prior 1.0 passed 43 of 48
cells and prior 0.001 passed 45 of 48. The no-fit cff-fely block-swap study then
showed that recurrence parameters strongly influence long-run behavior, but do
not own every failure: four of eight miss cells followed the full recurrence
donor rule, none followed the non-recurrence donor rule, and the remaining four
required both fitted blocks or were otherwise ambiguous.

Another prior or retained-seed block tune would now be post-outcome selection.
Before fitting again, this ticket derives a prospective stability intervention
that treats the state-dependent switching law and affine dynamics as one
system. The intervention must have an explicit probabilistic or optimization
meaning; it cannot be a magic rollout threshold or a favorable-seed filter.

This is a design-and-oracle ticket. It does not claim that robust classic
periodicity has been achieved and does not execute NASCAR fits or rollouts.

## Outcomes / Acceptance Criteria — WHAT

1. Extend `docs/DERIVATIONS.md` with a plain-English and mathematical account
   of the one-step second-moment or Foster-Lyapunov drift for the actual rSLDS
   transition:
   - source-conditioned recurrent probabilities over the destination regime;
   - destination-selected affine dynamics;
   - process covariance and affine offsets; and
   - an explicit state or regime-dependent Lyapunov function.
   Every expectation, source/destination axis, and conservative inequality has
   an independent executable oracle.
2. Separate three claims that must not be conflated:
   - an exact one-step expected-drift identity;
   - a sufficient global or region-wise bounded-moment certificate; and
   - a descriptive finite-grid or posterior-weighted score.
   A finite probe grid may diagnose a candidate but may not be called a global
   stability proof.
3. Compare at least these two principled candidate classes before selecting an
   implementation path:
   - a common quadratic contraction certificate that is sufficient under any
     switching sequence; and
   - a recurrence-aware, regime- or region-dependent quadratic drift condition
     that can retain stabilizing state-dependent switching.
   Record the rejected candidate and its falsifier. Selection is based on the
   derivation and synthetic oracles, not on cff-45ia/cff-fely outcome cells.
4. Specify exactly how the selected intervention enters learning. If it is a
   prior, constraint, projection, or outer merit term, derive that operation
   and state whether it preserves the registered rSLDS probabilistic family,
   the D10 inner fixed point, and the D10 free-energy semantics at the ticket's
   implementation base commit. A method
   that silently changes the objective or certifies a damped movement instead
   of the undamped map is rejected.
5. The selected candidate exposes a truth-free, deterministic API whose inputs
   are point parameters and declared numerical settings. It returns finite
   evidence plus distinct `satisfied`, `violated`, `infeasible`, and
   `diverged` outcomes. It never reads generating states, labels, or the prior
   development rollouts.
6. Independent tests cover K=2 and K=3 dense hand calculations, affine offsets,
   nonzero process covariance, a stable system, an individually stable but
   switching-unstable system, and a recurrence-stabilized system when the
   selected certificate supports that case. The oracle checks gradients or
   accepted projections if the intervention needs them.
7. Numerical-contract tests cover literal categorical zeros, empty support,
   improper/non-finite parameters, float32 guards derived from dtype, a
   near-unit spectral boundary, and K=8 at a domain 10–100 times the compact
   fixture. No absolute ridge or stability margin is hard-coded below the
   dtype's resolution.
8. Update `docs/ARCHITECTURE.md` and `docs/LIMITATIONS.md` in the same commit as
   the selected authority. The documents name the owner of the certificate,
   objective contribution, parameter installation, status, and persistence
   boundary. If no candidate is both sound and compatible with the parity
   claim, close this ticket with that explicit negative result instead of
   implementing a heuristic.
9. Create a separate prospective experiment ticket only after this derivation
   passes review. That ticket must use fresh development coordinates, both
   initialization tiers, the unchanged transition/prediction/regime/
   bounded-periodicity gates, and untouched calibration and final seeds.
10. No CFFG production inference, NASCAR fit, retained cff-45ia/cff-fely
    rollout, adapter chain, SLDS control, calibration seed, final seed, or
    confirmatory conclusion changes or runs in this ticket.

## Methodological Suggestions — HOW

These suggestions are optional; the derivation and falsifiers control the
choice.

- Start from a quadratic drift identity such as
  `E[V_{z'}(x') | x,z] - V_z(x)`, retaining the affine offset and covariance
  trace rather than reasoning only from dynamics spectral radii.
- Treat a common quadratic Lyapunov function as a conservative baseline. It is
  valuable because it gives a real guarantee, even if it proves too strict for
  the NASCAR switching geometry.
- For a recurrence-aware candidate, make the domain and probability bounds
  explicit. Logistic probabilities over a finite observation-derived region
  do not control behavior outside that region unless a separate boundary
  argument does.
- Keep pure algebra and small optimization oracles outside `Engine` until the
  ownership and objective contract are settled.

## Future work

- A child experiment ticket may implement and preregister the selected
  intervention on fresh classic development coordinates.
- cff-3v7q remains the owner of the scientific decision and cannot select a
  prior or authorize a freeze until every declared development gate passes.
- If the only sound guarantee changes the model family beyond Cooper's parity
  target, record robust long-rollout adequacy as a limitation and return to the
  broader parity epic without disguising that tradeoff.

## Definitions

- **Foster-Lyapunov drift:** The expected one-step change in a nonnegative
  function of state; a negative drift outside a bounded set can establish
  bounded moments under stated regularity conditions.
- **Common quadratic certificate:** One positive-definite quadratic function
  that contracts for every permitted dynamics mode, independent of switching.
- **Recurrence-aware certificate:** A certificate that uses the recurrent
  destination probabilities and therefore reasons about switching and
  dynamics jointly.
- **Descriptive drift score:** A measured drift on a finite set or posterior
  distribution. It can rank or falsify candidates but is not automatically a
  global proof.

## Notes

Architecture handoff: the learnable/observed-only fit boundary and the
runtime numerical or status invariant rows in `docs/ARCHITECTURE.md` are
potentially applicable. IR/compiler/engine scheduling, persistence, and
factor-rule rows are structurally irrelevant to a pure host-side authority. If
the selected derivation requires a new runtime owner, those rows become
applicable and the ticket must name them before implementation.

Duplicated surfaces: both initialization tiers and every dynamics/recurrence
parameter writer are future consumers, but this ticket changes only the pure
authority, derivation, documentation, and independent tests. Experiment
scoring, adapters, controls, calibration, and final execution are deliberately
excluded and delegated to the future prospective experiment ticket.

Hostile-input decisions: exact zero transition support remains zero; empty or
improper support is infeasible; non-finite moments or objectives diverge;
float32 tolerances scale with local operands and operation counts; near-unit
systems remain distinguishable at the active dtype; K=8 evaluation must remain
polynomial and bounded in memory.

Legend: prerequisite --> dependent

```text
[x] cff-45ia prospective cohort (scientific miss)
      |
      +--> [x] cff-fely retained block attribution (interaction/ambiguous)
                  |
                  +--> [ ] cff-x3a8 joint stability derivation (this ticket)
                              |
                              +--> [ ] fresh prospective intervention cohort
                                          |
                                          +--> [ ] cff-3v7q classic decision
```

**2026-08-24T12:52:44Z**

Research question -> What stability statement is exact for the actual source-conditioned stick-breaking recurrence plus destination-selected affine dynamics, and what can enforce it without silently changing D10? Answer -> For V_i(x)=x^T P_i x, the exact drift is sum_j p_ij(x)[(A_jx+b_j)^T P_j(A_jx+b_j)+tr(P_jQ_j)]-x^T P_i x. A global inequality E[V_next|i,x] <= lambda V_i(x)+C with lambda<1 bounds second moments. One common P with A_j^T P A_j <= rho P for every j is a valid arbitrary-switching sufficient condition but is deliberately conservative. A recurrence-aware verifier may certify the scalar inequality over all tail cones/gate-tie faces plus a compact interior; a finite probe grid is descriptive only. It can remain a host-side point-parameter verifier with the same rSLDS family and D10 inner objective, but enforcement must explicitly be an outer gate, hard constraint/projection, or changed prior/objective—there is no behavior-changing option that changes none of those. Falsifiers -> K=2 shear matrices with individual spectral radius 0.5 have an unstable switching second-moment lift; K=3 scalar recurrence can suppress an A=1.2 mode in both tails and satisfy a global drift bound even though the common-quadratic per-mode test rejects. Sources -> Linderman et al. 2017 rSLDS equations and pinned NASCAR generator for the model convention; Johansson & Rantzer 1998 for piecewise quadratic Lyapunov methods; Fang & Loparo 2002 for Markov jump second-moment lifts. Recommendation -> common quadratic only as baseline/falsifier; research authority should be source-conditioned mode-quadratic drift with certified tails plus compact verification. Confidence -> high on identity/conditions/compatibility; medium-high on scalable K=8 subdivision pending oracle.

**2026-08-24T14:02:46Z**

Acceptance complete on integrated commits 9942f5e, 974dc29, 7317042, and runtime record 778a029. Selected authority: a truth-free recurrence-aware global Foster certificate with exact one-step drift, verified contraction matrix inequalities, closed orthant tails and gate-tie faces, and compact interval coverage. The common-quadratic certificate remains a sound arbitrary-switching baseline but is rejected as the selected path because the K=3 recurrence-stabilized oracle falsifies its usefulness. Independent review reproduced and closed three numerical false-proof classes: float32/outward rounding, x64-disabled truncation, and ill-conditioned generalized contractions. Exact reviewed code head e1634ab passed the one final short suite: 1,539 passed, 40 deselected, 5 warnings; pytest 708.34s, dashboard 716.502s. No NASCAR, fitting, rollout, calibration, final seed, production inference, or objective path ran or changed. Negative boundary: this ticket derives a safety/eligibility certificate, not a repair, feasible-point constructor, or convergence-guaranteeing solver; rejection may stall. The next pure synthetic bridge is cff-sawy; the one-shot cohort cff-i4o0 is already blocked on it. Legend: prerequisite --> dependent. [x] cff-fely --> [x] cff-x3a8 --> [ ] cff-sawy --> [ ] cff-i4o0 --> [ ] cff-3v7q.
