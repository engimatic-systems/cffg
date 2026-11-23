---
id: cff-iur0
status: closed
deps: [cff-s6yy]
links: []
created: 2026-08-31T03:47:35Z
type: task
priority: 2
assignee: coopwilliams
parent: cff-zov5
tags: [rslds, derivation, inference, hybrid]
---
# Derive the smallest composable regime-trajectory approximation

## Background in Plain English — WHY

D9 repaired one important defect: recurrence learning now uses the inferred
joint pair `q(z_t, z_{t+1})` instead of multiplying two separate regime
marginals. It still represents the complete regime path and continuous path
separately. That separation forces source-weighted continuous moments to lose
dependence that may matter when both paths are inferred from observations.

Historical development evidence narrows the question. Truth-clamped
consecutive regimes let the local recurrence update recover distinct source
rows. Exact categorical smoothing, D9 pair readout, posterior replacement,
coverage, affine alignment, and outer convergence each passed their direct
checks. A D10 whole-model coordinate did not meet the recovery cap, so its
existence is not proof that D10 is the right answer. This ticket must establish
the smallest justified dependence and its factor-graph boundary before any
production implementation begins.

## Outcomes / Acceptance Criteria — WHAT

1. docs/DERIVATIONS.md states exactly which conditional dependence D9 removes
   between regime membership and continuous-state moments and which recurrence
   learning statistic is changed by that removal.
2. A tiny independent reference with `K=2`, continuous dimension `1`, and
   horizon no greater than `6` enumerates the discrete path and uses analytic
   Gaussian integration or independently implemented quadrature. It imports
   no production structured-recurrence or proposed component solver.
3. The reference includes one reduction case where D9 is exact and one coupled
   case with nonzero source-conditioned continuous moments that D9 cannot
   represent. Both cases state dtype-scaled comparison tolerances before the
   proposed approximation is evaluated.
4. The derivation selects the smallest variational family or region belief
   that retains the dependency from criterion 1. It lists every retained
   marginal, cross-moment, auxiliary coordinate, normalization constraint,
   and gauge freedom; it also states why each larger candidate is unnecessary.
5. The derivation gives stationary update equations, or a complete
   gauge-reduced residual when closed-form updates are unavailable, for every
   internal coordinate and every incoming and outgoing boundary message.
6. The component's node-energy terms, region entropies, separator corrections,
   counting numbers, and bound direction are derived from one stated
   variational objective. No diagnostic residual or fitted-parameter change is
   relabeled as free energy.
7. The accepted construction requires work linear in horizon and polynomial in
   regime count and continuous dimension. It does not enumerate complete
   regime paths or materialize a dense covariance over the entire continuous
   trajectory. Exact asymptotic work and persistent/transient storage are
   recorded.
8. Executable derivation anchors compare boundary messages, required
   source-conditioned moments, the complete local objective, and its gradient
   or stationary residual with the independent tiny reference.
9. The hostile-input analysis covers literal zero regime support, an empty
   supported component, improper or vacuous Gaussian cavities, float32, and a
   case 10–100 times the smallest horizon or regime count. Guards derive from
   dtype properties, and non-finite quantities have a loud status or error.
10. docs/DERIVATIONS.md includes a plain-English companion explaining what the
    richer coordinate remembers and why ordinary hidden-regime discovery does
    not by itself provide source-specific learning.
11. The final decision either authorizes cff-ocxl with a named approximation
    identity and explicit boundary contract, or records that the evidence does
    not justify a new component and rewrites cff-ocxl before implementation.
12. This ticket changes no production compiler selection, engine execution,
    learning readout, checkpoint format, or public factor behavior.

## Methodological Suggestions — HOW

These suggestions are optional. The implementer may choose another sound
method that satisfies the outcomes.

- Start from the D9 factor-local pair regions and add only the
  regime-conditioned Gaussian information needed by criterion 1.
- Use the pure D10 equations from the historical PR #93 branch as a candidate
  or falsifier, not as the required design.
- Apply the **New factor or rule family** row only as a surface inventory and
  the **Runtime numerical or status invariant** row for solver semantics.
  Production surfaces are N/A for this derivation-only ticket.
- Prefer algebraic reduction and tiny independent references over another
  learned NASCAR run.

## Future work

cff-ocxl implements only the accepted mathematical contract. If the selected
coordinate needs a new serialized declaration, cff-ocxl must also apply the
**New IR field** change path. Source-function qualification remains entirely
owned by cff-x0if.

## Definitions

- **Boundary message:** The canonical message a component receives from or
  sends to a neighboring factor through a declared boundary variable.
- **Cross-moment:** An expectation involving a regime indicator and a
  continuous state, such as `E[1(z_t=i) x_t]`.
- **Gauge freedom:** A change in numerical coordinates that leaves the
  represented probability distribution unchanged.
- **Smallest coordinate:** The accepted candidate with the lowest-order joint
  marginals or cross-moments and lowest asymptotic storage among the candidates
  that satisfy criteria 1–10.
- **Region belief:** A normalized approximation over several variables whose
  entropy and expected energy have explicit counting ownership.
- **Stationary residual:** A vector that is zero exactly when the declared
  constrained variational stationarity equations hold on the active support.

## Notes

Change-path handoff: the applicable existing rows are **New factor or rule
family** for the mathematical surface inventory and **Runtime numerical or
status invariant** for any proposed iterative solve. Every production surface
in those rows is N/A in this ticket because acceptance criterion 12 limits the
change to derivation, independent oracles, and their executable anchors.

Duplicated-surface handoff: host/JIT, flooding/scheduled,
single-instance/`vmap`, and current/legacy checkpoint pairs are N/A until
cff-ocxl. The independent reference must cover x64 and float32 and must not
import the production implementation it evaluates.

Intelligible-ticket review: complete. This ticket owns only the derivation,
plain-English explanation, and independent tiny references. cff-ocxl owns all
production integration; cff-x0if owns learned-model qualification.

Wiggle-audit review retained clear contextual uses. `canonical` is the
ordinary boundary message representation; `integration` is mathematical
Gaussian integration; `N/A` marks the four explicitly excluded production
surface groups; `public` is the unchanged factor API; `implement` refers to
planned cff-ocxl work, an independent quadrature implementation, or the
production code forbidden here; `provide` means hidden-regime inference alone
does not supply source-learning information; `support` means literal active
probability support; `complete` enumerates every path, equation block, or local
objective named by its sentence; `remaining` names cff-x0if's later scope;
`approximation` is the mathematical family selected by this ticket; `closed`
means a closed-form update; `check` and `pass` refer to the listed direct
historical checks; `convergence` is the historical outer fixed-point status;
`evidence` means historical measurements or the formal decision record;
`exact` identifies literal algebra, exhaustive smoothing, asymptotic formulas,
or a zero residual; `oracle` means the independent executable references;
`sound` means another method satisfying acceptance; `fit` is a fitted-parameter
diagnostic excluded from free energy; `infer` means hidden-state posterior
calculation; `learn` and `recover` mean recurrence statistics or estimating
withheld source rows as stated; `scale` is dtype scaling; `small` uses the
defined Smallest coordinate or the enumerated tiny fixture; `expected` is a
mathematical expectation; `may` marks possibility or methodological choice;
`any`, `every`, and `only` are bounded by their locally enumerated dependencies,
coordinates, boundaries, and scope exclusions; `current` appears only in the
named current/legacy review pair; and `existing` names the two architecture
rows quoted in the handoff.

**2026-09-13T17:51:33Z**

Planning handoff: use New factor or rule family for region/message/energy ownership and Runtime numerical or status invariant for auxiliary iteration. Production compiler, host/JIT, flooding/scheduled, single/vmap, and current/legacy checkpoint surfaces are N/A. The experiment and independent oracle will cover exact zero support (prune without floors), empty support (error), vacuous/improper Gaussian cavities (accept only integrable tilted components), float32 guards derived from finfo, and a 10–100x horizon case. Candidate comparison starts with pair-conditioned Gaussian recurrence regions and explicit boundary moment constraints; a full trajectory coordinate is not presumed necessary.

**2026-09-13T17:59:40Z**

Research question → smallest region and ordinary boundary feasibility. Answer → b(i,j,x)=rho_ij N(x;m_ij,S_ij) with explicit first/second-moment consistency; Part I §5.4 Eq.31 permits this EP construction. Shared source/stick JJ auxiliaries use exposure-weighted conditional squared logits. Exact local region entropy and Gaussian separator entropy define one constrained Bethe functional; JJ raises it at the same feasible beliefs, with no global evidence-bound claim. Gaussian quotient precision may be negative, and source-homogeneous logits alone do not reduce to D9. Sources → Part I https://biaslab.github.io/pdf/arxiv/2306.08014.pdf ; Opper & Winther 2005 https://www.jmlr.org/papers/volume6/opper05a/opper05a.pdf ; direct algebra and independent scalar counterexample. Confidence → high. These findings will be distilled into D11 and the cff-ocxl handoff.

**2026-09-13T18:10:27Z**

The boundary audit found a decisive source-learning falsifier: two external switched priors with source-conditioned means (+2,-2) versus (-2,+2) have identical source probabilities and Gaussian marginal moments (mean 0, variance 5). With R=r=0 and observed success, exact weighted source first moments are (+1,-1) versus (-1,+1); the local triplet sees the same cavities and returns zeros. JJ is exact here. Separate singleton constraints cannot convey the missing association, even with the companion EC mixture messages. Decision: use criterion 11 negative outcome, withhold production authorization, and rewrite cff-ocxl as the hybrid-separator derivation gate. The local equations remain useful, independently tested candidates, not an accepted repair.

**2026-09-13T18:15:52Z**

Roadmap decision and review handoff:
- D11 derives the local pair-Gaussian/JJ candidate, its full local and EC objectives, stationary messages/auxiliaries, all required learning moments, gauges, support semantics, signed outputs, and bounded-work/storage costs.
- The independent K=2 scalar oracle enumerates paths up to T=6 and integrates Gaussian paths or literal/JJ factors without production/candidate imports. Exact reductions, nonzero local cross-moments, objective/gradient, float32, T=100/K=8, and d=10 are executable.
- The decisive D11.15 twins have identical separate boundaries but opposite required source-weighted means. The variance twin loses the precision statistic too. Local exactness therefore does not justify a composable source-learning implementation. This is criterion 11's negative outcome, not completion of the proposed larger construction.
- cff-ocxl is rewritten as the joint hybrid separator/grouping derivation gate. A positive result must create bounded production tickets and attach them as qualification blockers. cff-ulbz separately tracks D3's projection-order mismatch; its implementation is conditional on the gate.
- No production compiler/runtime/readout/checkpoint behavior changed. Applicable change paths remain the inventory in the planning note; duplicated production pairs are N/A.
- Focused validation after final code edit: 57 passed (regime-pair derivation, structured recurrence, repository hygiene, dashboard). Full short-suite gate follows.

Legend: prerequisite --> dependent
[x] cff-s6yy (D9 foundation)
  --> [ ] cff-iur0 (negative decision; validation in progress)
        --> [ ] cff-ocxl (hybrid separator derivation; ready after this gate)
              +--> [ ] cff-ulbz (conditional mixture EC binding)
              |       --> [ ] cff-x0if (source recovery)
              +----------> [ ] cff-x0if
                              --> [ ] cff-zov5 (recovery epic)
No parallel implementation root is ready: cff-ulbz waits on the mathematical
decision, and cff-x0if waits on both branches plus future implementation
dependencies. Built from tk dep tree --full, tk ready, tk blocked.
tk dep cycle: no cycles.

**2026-09-13T18:42:45Z**

Final validation and landing: uv run python tools/dashboard.py --run-suite passed once after the final code edit: 2,295 passed, 41 deselected, 4 dependency deprecation warnings, 1,220.64 seconds pytest time. The required 1,560-second quiet wait was honored. Dashboard refreshed at ~/agent-share/cffg-tests.html; runtime history updated. Focused checks passed 57 tests before the full run. tk dep cycle reports no cycles; git diff --check is clean.

Closing under acceptance criterion 11's explicit negative outcome. No larger construction or production implementation is claimed accepted. D11.15's independently integrated source-mean and variance counterexamples falsify the separate-boundary repair. cff-ocxl is the ready next mathematical gate after this closure; cff-ulbz remains conditional on its decision; cff-x0if is blocked on both and must receive future implementation dependencies before qualification. No unanswered user decision is needed for this bounded negative-result PR.

**2026-09-15T13:20:19Z**

D12 follow-up was deferred at Cooper's direction; see cff-ocxl for the short downstream summary and preserved branch pointers. This ticket remains closed for its completed derivation and boundary findings. The deferral is a prioritization decision, not a reversal of those findings or a source-recovery result.
