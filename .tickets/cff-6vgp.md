---
id: cff-6vgp
status: closed
deps: [cff-yr0d]
links: []
created: 2026-08-22T15:57:07Z
type: task
priority: 1
assignee: coopwilliams
parent: cff-0oqo
tags: [rslds, inference, engine]
---
# Implement structured recurrence pair rules and node energy

## Background in Plain English — WHY

The current recurrence rule is correct for an approximation that treats the
source and destination regimes as independent. Exact chain smoothing shows
that this loses material consecutive-state dependence. `docs/DERIVATIONS.md`
D9 defines a replacement that keeps the existing time-local recurrence factor
but gives it a joint source/destination belief.

This ticket owns the mathematical rule boundary. It must provide one tested
pair-potential, sum-product, pair-belief, Gaussian-message, and node-energy
implementation before the compiler and engine begin dispatching it.

## Outcomes / Acceptance Criteria — WHAT

1. `cffg.recurrence` exposes the D9 bounded expected log pair potential,
   log-space messages to both categorical roles, and one normalized
   source/destination pair belief from incoming cavities.
2. The pair belief's row and column marginals equal the two categorical edge
   beliefs at a stationary point under a dtype-scaled tolerance.
3. The Gaussian-drive message uses D9's pair-derived at-risk and
   signed-success masses. It does not reconstruct a product of singleton
   beliefs.
4. The recurrence node energy uses pair entropy and the expected bounded pair
   potential. Together with one categorical separator correction, two adjacent
   recurrence factors match the D9 chain objective and `-log Z` on the compact
   oracle.
5. Fixed observed drives match exact K=2 and K=3 enumeration. Uncertain drives
   retain the documented conservative Pólya–Gamma/Jaakkola–Jordan bound rather
   than claiming exact logistic integration.
6. Literal log-zero support remains zero. Empty support, improper Gaussian
   moments, and non-finite parameters are loud failures. Float32 guards derive
   from machine precision, and K=8 works without a cubic transition tensor.
7. P-substitution on the recurrence factor remains an explicit compile-time
   rejection. Ordinary recurrence factors may still coexist with supported
   GFE factors elsewhere in a model.
8. Current production dispatch remains unchanged in this ticket. The new rule
   and energy are independently callable authorities for cff-fbmr.

## Methodological Suggestions — HOW

These suggestions are optional. The implementer may choose another sound
method.

- Reuse the pure oracle in `cffg.experiments.structured_recurrence` and the
  existing D8 stick-score algebra.
- Introduce a small named pair-belief result instead of passing unrelated
  singleton and statistic arrays.
- Evaluate the node energy directly from the pair belief. Do not infer it from
  outgoing messages, whose gauges are arbitrary.

## Future work

cff-fbmr will connect these rules to compiler plans, schedules, and the engine.
cff-ifbr will replace public learning readouts after engine state owns the pair
belief. cff-63dd owns checkpoint and migration compatibility.

## Definitions

- **Pair belief:** The joint probability of a source regime and its next
  destination regime at one transition.
- **Separator correction:** The categorical edge-entropy term that prevents an
  interior singleton entropy from being counted twice by adjacent pair
  factors.
- **JJ bound:** The Jaakkola–Jordan lower bound obtained from the
  coordinate-optimal Pólya–Gamma auxiliary.

## Notes

Architecture handoff: apply the **New factor or rule family** and **BFE or GFE
objective** rows conservatively. Authority is `cffg.recurrence`. Compiler,
arena, schedule, public readouts, checkpoints, and model builders are N/A
except for regression checks that production dispatch did not change.

Duplicated surfaces: source/out roles, pair/messages/energy, fixed/uncertain
drives, float32/x64, exact-zero/dense support, and host/JAX calculations must
agree.

**2026-08-22T16:15:50Z**

Implemented in CFFG commit e3a94cb. Public authority is re-exported from cffg.recurrence: structured_pair_potential -> StructuredPairPotential(log_potential, omega, proper); structured_pair_belief -> StructuredPairBelief(to_out, to_in, probability, source_probability, destination_probability, proper); structured_gaussian_message -> StructuredGaussianMessage(eta, precision, exposure, signed_success, proper); structured_pair_energy; structured_separator_energy; and structured_node_energy. Inputs use source rows and destination columns. Shape failures raise exact ValueError boundaries documented in the public docstrings. Under jit/vmap, numeric failures return proper=False with NaN result arrays; energy failures return +inf. Literal -inf categorical support stays exact zero. Production engine/compiler dispatch remains unchanged; cff-fbmr should wire these authorities without recreating singleton products. Validation: 52 focused tests passed across the new rule/oracle/current recurrence suites in 61.71s, including K2/K3 fixed-drive enumeration, uncertain-drive JJ direction, pair marginal consistency, pair-derived Gaussian masses, separator/-logZ energy, exact zero/empty/improper/nonfinite/f32/K8, jit/vmap, and the existing recurrence P-substitution rejection.
