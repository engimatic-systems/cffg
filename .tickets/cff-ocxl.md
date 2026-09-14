---
id: cff-ocxl
status: closed
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

## Notes

**2026-09-14T00:53:37Z**

Planning handoff (stacked on PR #141): apply New factor or rule family for mathematical region/message/objective ownership and Runtime numerical or status invariant for the experimental solve. Production compiler grouping, slots, packing, Arena/member_of, synchronous/scheduled execution, ledger, readouts, serialized identity, plate scoping, edits/composition, learning/checkpoints, estimator, public exports, host/JIT, single/vmap, and current/legacy persistence are N/A under criterion 12. Candidate: adjacent hybrid states with per-pair joint Gaussian regions and per-regime moment-consistent separators. Before coding: literal zeros restrict a fixed active support without floors; empty regions fail; inactive auxiliary coordinates equal zero; changed support rebuilds the active coordinates; signed or vacuous cavities are allowed only when aggregate and tilted beliefs are proper; invalid shapes/asymmetry/non-finite coordinates fail; float32 uses finfo-derived resolution guards; non-finite objective/residual is divergence; iteration budgets distinguish convergence/exhaustion/divergence and proposed steps use damping/backtracking. Independent reference remains K=2, scalar, T<=6; candidate inventory will be checked at 10–100x horizon and separately higher dimension. No exactness claim for source-homogeneous nonzero slopes or global evidence bound. Exterior attachment counting and mixture projection remain explicit decision gates.

**2026-09-14T01:04:41Z**

Research question → smallest closed hybrid region scopes, lawful exterior, and residual approximation. Answer → destination-selected affine Gaussian dynamics admits alternating R(i,j,x)–H(j,x)–D(j,x,y)–Y(j,y) scopes; eliminating D/H gives the tested pair-Gaussian block with identical stationary moments and objective because the conditional y|x,j is independent of i and H(D)-H(H)=E G(C_j). Hybrid messages require label-dependent Gaussian normalization constants, with only a common scalar gauge. Ordinary ports need explicit reduced-measure identity adapters and full-tilt/cavity quotients; D3 collapse-first is incompatible where exterior mixture ports remain. The D11.15 mean/variance twins are repaired. A second, temporal reflected-mixture example has exact JJ and zero EC moment residual yet different exact learning increments: retained conditional moments remain an approximation, not exact source recovery. Sources → D11/D3/D9 in docs/DERIVATIONS.md plus direct independent derivations by hybrid_separator_math and separator_minimality; candidate and tiny reference tests reproduce the relevant equations. Confidence → high; no production correctness or empirical recovery claim. Positive outcome will be limited to a named EC approximation and bounded implementation tickets.

**2026-09-14T01:15:18Z**

Decision and implementation handoff: D12 accepts hybrid-regime-ec-v1 as an explicitly approximate EC coordinate. The runtime remains D9; no production selection, engine path, learning readout, public factor API, or checkpoint format changed. The mathematical authority and every future duplicated surface are inventoried in the new Derived component coordinate row. The experiment is NumPy-only and unbound; production host/JIT, synchronous/scheduled, single/vmap, and current/legacy payload pairs are N/A for this derivation.

Acceptance evidence: the sign and variance twins transmit the required weighted moments and learning increments; independently enumerated one-group and decoupled temporal reductions agree. A coupled T=3 fixed-JJ case has maximum pair/first/second errors 8.08e-6/2.78e-5/3.79e-4, FE error 1.21e-5, and maximum learning-increment error 2.70e-5; these are disclosed approximation gaps, separate from the <8e-11 stationary quotient residual. Literal quadrature pins the local JJ gap. Independent dual derivatives cover constant, linear, and precision coordinates; the xi derivative exposes the false c=0 stationary point. Alternating/grouped entropy and quadratic-energy equivalence, both ordinary adapters, signed quotients, zero/empty/support-change cases, vacuous inputs, non-finite residuals, float32, T=100, K=20, and d=10 are executable. A reflected temporal mixture has zero EC residual and exact JJ yet opposite exact learning biases: D12 explicitly does not guarantee exact temporal learning or a global evidence bound. The experiment solves held auxiliaries only; the complete joint residual is derived and its production solver is cff-f3jz.

The six bounded production tickets are cff-k6g5 (serialized semantics), cff-ffsg (local JAX kernels), cff-7sw5 (compiler plans), cff-f3jz (execution/joint solve/ledger), cff-czdq (learning and persisted identity), and cff-kul2 (public composition/parity). cff-ulbz is confirmed for exterior mixture ports and now depends on cff-f3jz. cff-x0if directly depends on all six plus cff-ulbz; source-recovery qualification cannot start after this mathematical gate alone.

Legend: prerequisite --> dependent. Transitive/redundant records are omitted
from this compact companion; tk remains the source of truth.

[x] cff-iur0 (negative singleton-boundary gate)
  --> [ ] cff-ocxl (D12 accepted; final suite pending)
        +--> [ ] cff-k6g5 (declarations; ready after closure)
        |       --> [ ] cff-7sw5 (plans)
        |                 |
        +--> [ ] cff-ffsg (kernels; parallel ready root after closure)
                          |
              plans + kernels --> [ ] cff-f3jz (execution)
                                        +--> [ ] cff-czdq (learning identity)
                                        +--> [ ] cff-ulbz (exterior mixtures)
                                                  |
                           learning + mixtures --> [ ] cff-kul2 (composition)
                                                      --> [ ] cff-x0if (recovery)
                                                            --> [ ] cff-zov5 (epic)

Built from tk dep tree --full cff-x0if, tk ready, and tk blocked; tk dep cycle reports no cycles. The only ready work in this chain before closure is cff-ocxl; declarations and local kernels become parallel roots after it closes. Full short-suite validation remains the final gate; focused derivation/D9/hygiene/dashboard checks passed 91 tests after the final relevant code edit.

**2026-09-14T01:16:24Z**

Independent bounded mathematical review: natural normalizers, signed quotient gauges, primal/dual equality, adapter entropy counting, destination-selected elimination, and the reflected temporal falsifier agree with D12. The review requested explicit precision-coordinate derivative factors (-1/2 diagonal, -1 symmetric off-diagonal); D12.9 now states them and the executable gradient test already checks the diagonal factor. No remaining mathematical blocker was identified. Review does not claim an implemented joint auxiliary solver or empirical recovery.

**2026-09-14T01:43:20Z**

Final validation and landing: uv run python tools/dashboard.py --run-suite passed exactly once after the final code edit: 2,329 passed, 41 deselected, four dependency deprecation warnings, 1,218.79 seconds pytest time. The full 1,560-second quiet wait was honored without process/log polling. The dashboard refreshed at ~/agent-share/cffg-tests.html with 34 new tests; runtime history updated. Focused tests passed 91 before this run. No slow empirical or recovery runs were started. Closing under the positive mathematical outcome: D12 names the approximation and ordinary exterior contract; all six implementation dependencies and cff-ulbz are attached to cff-x0if. Next ready roots are cff-k6g5 (declarations) and cff-ffsg (local kernels); current/legacy learning migration remains mandatory in cff-czdq. The tree and dependency artifacts receive final checks before the stacked PR is published.

**2026-09-14T01:44:19Z**

Published stacked PR #142: https://github.com/engimatic-systems/cffg/pull/142 . Base is cff-iur0-regime-trajectory-derivation (PR #141), as requested. cff-ocxl is closed, all code/doc/ticket work is committed, the single final short-suite result is green, and the next implementation roots are cff-k6g5 and cff-ffsg. No remaining work in this derivation ticket; source-recovery qualification remains blocked on implementation and public composition.
