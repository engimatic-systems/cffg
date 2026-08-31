---
id: cff-iur0
status: open
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
