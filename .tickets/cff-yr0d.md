---
id: cff-yr0d
status: in_progress
deps: [cff-nbo9]
links: []
created: 2026-08-22T15:45:12Z
type: task
priority: 1
assignee: coopwilliams
parent: cff-0oqo
tags: [rslds, inference, derivation]
---
# Derive structured recurrence smoothing and free-energy semantics

## Background in Plain English — WHY

The cff-nbo9 oracle proves that the current recurrence factor is implemented
correctly for its edge-factorized variational family, but that family loses the
posterior dependence between consecutive regimes. Exact or forward-backward
smoothing produces materially different pair responsibilities, and the
cff-rk7i clamping ladder shows that those pair responsibilities are sufficient
to restore heterogeneous transition recovery.

Changing only the recurrence learning readout would be inconsistent: the
messages, beliefs, sufficient statistics, and free-energy entropy terms would
then describe different approximate distributions. Before production code
changes, the project needs one structured categorical-chain objective whose
stationary messages, pair beliefs, learning statistics, and Bethe/generalised
free-energy accounting agree.

## Outcomes / Acceptance Criteria — WHAT

1. `docs/DERIVATIONS.md` defines the structured variational family for the
   complete regime path and states exactly which Gaussian, Pólya–Gamma, and
   parameter beliefs remain factorized.
2. The derivation gives the bounded expected log-transition potential for each
   source/destination pair and shows how it becomes a log-space chain potential
   without using generating states or parameters.
3. Forward and backward categorical messages, normalized singleton beliefs,
   and consecutive-regime pair beliefs follow as stationary conditions of the
   declared objective. With a fixed observed drive, they reduce to ordinary
   sum-product smoothing.
4. Messages and learning statistics for the Gaussian drive and recurrence
   coefficients use the same consecutive-regime pair belief, including
   decision-list at-risk and signed-success masses. A product of singleton
   beliefs is not retained as a hidden fallback.
5. The node-local bounded transition energy, chain entropy, factor-region
   entropy, categorical edge corrections, region/counting numbers, and bound
   direction are explicit. The derivation states how ordinary BFE and any
   supported GFE/P-substitution mode differ; unsupported modes fail at compile
   time rather than silently reuse the old energy.
6. The design chooses and justifies one production ownership boundary: hybrid
   sum-product categorical messages inside the existing recurrence factors, a
   named structured-chain region, or another explicit mechanism. It traces the
   applicable architecture change-path surfaces before implementation.
7. A pure executable oracle checks the derived messages, singleton/pair
   beliefs, recurrence statistics, and free energy against enumeration for
   K=2 and K=3 chains. It also proves equality with forward-backward for a
   longer finite chain.
8. The oracle includes a case where singleton products are wrong, an uncertain
   Gaussian-drive case that retains the declared JJ/PG approximation boundary,
   and a fixed-drive case that isolates categorical exactness.
9. Literal zero support remains zero; inconsistent empty support and non-finite
   potentials fail loudly; vacuous evidence is normalized; float32 tolerances
   derive from dtype precision; and K=8, T=100 runs in O(T*K^2) memory/time
   without path enumeration.
10. The conclusion creates bounded implementation tickets for the selected
    engine, energy, learning-readout, serialization, and consumer surfaces. No
    production behavior or NASCAR final/calibration seed changes in this
    derivation ticket.

## Methodological Suggestions — HOW

These suggestions are optional. The implementer may choose another sound
method that satisfies the outcomes.

- Start from the cff-nbo9 enumerator and express the structured categorical
  block as expected log transition potentials plus local categorical evidence.
- For a chain, check the entropy identity using consecutive-pair entropies minus
  the repeated interior-singleton entropies before mapping it onto CFFG region
  and edge corrections.
- Keep the continuous-drive JJ/PG bound explicit. Exact categorical smoothing
  does not make that separate approximation exact.
- Prefer a reusable categorical structured-inference owner over a model-specific
  fitter exception, unless the architecture audit proves the general owner
  would violate CFFG factor or serialization boundaries.

## Future work

The implementation tickets created here block `cff-0oqo`. After they land, the
existing truth-free clamping ladder and heterogeneous development recovery gate
must be rerun before budgets, calibration, or final evidence are reconsidered.
This ticket does not alter the paper-equation adapter.

## Definitions

- **Structured regime belief:** One approximate joint distribution over the
  whole categorical path, represented by consistent singleton and consecutive-
  pair marginals.
- **Counting number:** A coefficient assigning each region or node entropy its
  contribution to a region-based free-energy objective.
- **Expected log-transition potential:** The categorical pair score obtained by
  averaging the bounded recurrence log potential over the current continuous
  and parameter beliefs.

## Notes

Architecture handoff: the **New factor or rule family** and **BFE or GFE
objective** rows in `docs/ARCHITECTURE.md` are the conservative starting point.
The derivation must decide whether the existing recurrence `FactorType` can own
the categorical sum-product path or whether a new structured region is needed.
IR, compiler planning, schedule, engine dispatch, belief/readout APIs, energy,
learning, serialization, checkpoints, and model builders are applicable or
must be explicitly marked structurally irrelevant in the implementation split.

Duplicated surfaces: source/out categorical roles, singleton and pair beliefs,
forward/backward messages, decision-list gates, Gaussian and coefficient
messages, BFE/GFE support, node/edge entropy, exact zeros, both dtypes, host/JIT,
serialization, and learning readouts must remain aligned.

Hostile-input decisions: literal zero support is preserved; contradictory empty
support and non-finite potentials diverge loudly; vacuous evidence yields a
normalized belief; dtype-scaled tolerances govern float32 and float64; K=8 and
T=100 exercise uncomfortable size without exhaustive enumeration.

Legend: prerequisite --> dependent

```text
[x] cff-rk7i clamping localization
      |
[x] cff-nbo9 exact smoothing ownership oracle
      |
      +--> [ ] cff-yr0d structured objective derivation (ready)
                  |
                  +--> [ ] structured recurrence implementation tickets
                              |
                              +--> [ ] cff-0oqo recovery gate
                                          |
                                          +--> [ ] fresh calibration and freeze
```
