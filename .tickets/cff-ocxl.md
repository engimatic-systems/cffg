---
id: cff-ocxl
status: open
deps: [cff-iur0]
links: []
created: 2026-08-31T03:47:35Z
type: task
priority: 2
assignee: coopwilliams
parent: cff-zov5
tags: [rslds, inference, hybrid, compiler]
---
# Derive joint regime/state separators before recurrence implementation

## Background in Plain English — WHY

cff-iur0 ended with its explicit negative outcome. D11's local Gaussian
region can retain source/destination-conditioned moments, but cannot receive
source/state association from another factor through separate categorical
and Gaussian boundaries. Two switched priors with opposite source-conditioned
means have identical singleton cavities and require opposite recurrence
learning updates. D11.15 reproduces this even when the JJ bound is exact and
the local stationary residual is zero.

The previous implementation ticket is therefore replaced by this derivation
gate. The next candidate must carry the association across a joint boundary
or put the factors that infer and consume it under one owner. A benchmark
topology classifier or a new learning readout alone cannot supply it.

## Outcomes / Acceptance Criteria — WHAT

1. The proposed separator/grouping retains source probabilities,
   E[1(z=i)x], and E[1(z=i)xx^T] from the factor that infers them to the
   recurrence factor that consumes them.
2. An independent K=2, scalar, horizon <=6 reference compares both signs of
   the switched-prior D11.15 fixture, a source-dependent variance analogue,
   and a genuinely temporal Gaussian-dynamics case. Required first/second
   moments and learning increments agree in declared exact reductions;
   every approximation gap is reported with tolerances fixed beforehand.
3. The derivation selects the smallest justified region and separator scopes,
   listing every belief, cross-moment, auxiliary, constraint, support
   convention, and gauge. It compares a hybrid separator over (z_t,x_t)
   against merging recurrence and switching-dynamics factors.
4. One constrained variational objective gives every internal node energy,
   region entropy, separator correction, counting number, and bound
   direction. It does not infer a global evidence bound from local JJ
   exactness or moment consistency.
5. Stationary equations or a complete gauge-reduced residual cover every
   internal coordinate and every ordinary exterior categorical/Gaussian
   boundary. The declared exterior never cuts the required source/state
   association without an explicit approximation and its falsifier.
6. Gaussian message precision may be signed; aggregate and tilted beliefs
   must be proper. Existing D3 collapse-first outputs are not assumed
   stationary for EC. The derivation either incorporates the D11.14 companion
   tracked by cff-ulbz or proves why it is unnecessary.
7. Work is linear in horizon per objective/residual evaluation and polynomial
   in regime count/dimension, with explicit local and global iteration-budget
   multipliers. Persistent/transient storage and differentiation/linear-solve
   costs are stated. No complete path enumeration or dense continuous
   trajectory covariance is used outside the tiny reference.
8. Executable anchors compare boundary messages, cross-moments, the complete
   local objective, and its gradient/residual against the independent
   reference. A 10–100x horizon or regime case tests the stated inventory.
9. Before implementation of the experiment, hostile-input criteria cover
   literal zero support, empty supported regions, inactive auxiliaries,
   support changes, improper/vacuous Gaussian cavities, signed messages,
   float32, and non-finite objectives/residuals. Guards derive from dtype
   properties; convergence, exhaustion, and divergence remain distinct.
10. The plain-English companion explains what information crosses each
    boundary and why the D11.15 twins can now produce opposite learning
    signals. Fixed-drive and drive-independent local D9 equivalence are
    preserved; source-homogeneous nonzero slopes are not mislabeled exact.
11. A positive decision creates separate bounded production implementation
    tickets, with the Derived component coordinate change-path inventory,
    serialized selection/identity, execution parity, learning/checkpoint
    migration, composition tests, documentation, and the full test gate.
    cff-x0if receives dependencies on those tickets before this one closes.
    A negative decision records the remaining mathematical blocker and
    does not authorize implementation.
12. This ticket changes no production selection, engine behavior, learning
    readout, public factor API, or checkpoint format.

## Methodological Suggestions — HOW

These are optional; another sound construction meeting the outcomes is welcome.

- Start with a hybrid separator whose conditional Gaussian moments retain
  the association lost in D11.15, then inspect adjoining region scopes.
- Reuse D11's local JJ algebra, true labeled entropy, signed Gaussian quotient,
  and xi=c² auxiliary residual where their constraints remain valid.
- Use New factor or rule family for the mathematical surface inventory and
  Runtime numerical or status invariant for solver semantics. All production
  host/JIT, flooding/scheduled, single/vmap, and checkpoint pairs are N/A
  until the separate implementation tickets.
- Keep exact local reductions separate from evidence of source recovery.

## Future work

cff-ulbz tracks the independently discovered Gaussian-mixture projection
boundary and depends on this mathematical decision. cff-x0if owns empirical
source-recovery qualification and remains blocked on this gate, the
compatible mixture decision, and the implementation tickets created by a
positive outcome. GFE/P-substitution, new recurrence link functions, and
whole-trajectory sampling are separate work.

## Definitions

- Hybrid separator: a shared belief over both a regime and continuous state,
  including source-conditioned moments.
- Region: a normalized belief whose factor energy and entropy have one owner.
- EC: expectation consistency, meaning specified sufficient-statistic
  moments agree even when the full neighboring densities differ.
- JJ: the Jaakkola–Jordan quadratic bound on the logistic log factor.
- Cavity: the product of other incident factors' messages at a boundary.
- Signed Gaussian message: natural precision that may be negative while its
  product with other messages defines a proper Gaussian belief.
- Gauge: a coordinate change that leaves the represented belief unchanged.
- Stationary residual: equations whose zero set is constrained stationarity
  on the declared active support.
- D11.15 twins: two switched priors with opposite source-conditioned means
  but identical separate regime and Gaussian boundary moments.
