# Glucose: the CFFG insulin controller

**Ticket:** cff-t2g3 · **Priority:** parked — do not start until Cooper says go.

## The bar (verbatim intent, recorded 2026-07-05)

Cooper's own VERSES-era hand-tuned experiments: **67% TIR on unannounced
adults AND adolescents, zero critical episodes, 8-virtual-hour episodes.**
The end goal is an insulin pump doing **continual learning with a white-box
algorithm and a modular physiological world model**. Beating 67%-with-zero-
criticals on that protocol is the win condition — not our reactive
controller's 87.6%, which is on an easier protocol (adults only, 24 h, no
critical-episode accounting) and is *not comparable*.

## Benchmark work needed first (cheap, do when unparked)

Extend `cffg/benchmarks/glucose.py`:
- adolescent cohort (`adolescent#001..010`) alongside adults;
- 8-hour episodes (multiple per day-phase, or rolling starts);
- a **critical episode counter** — confirm the definition with Cooper
  (BG < 54 mg/dL sustained? < 40 at any sample? early-termination events?)
  before implementing; it is the primary metric, TIR second.

## Modeling sketch (the actual project, later)

- Discretize a compact physiological state: BG bucket ⊗ BG-trend bucket ⊗
  insulin-on-board bucket ⊗ carbs-on-board bucket. Keep it small (H4 tabular
  wall!) — think 15×5×5×4, factored transitions, not one 1500-state blob.
- Controls: basal multiplier ⊗ bolus bucket (small grid; safety-capped).
- Transitions Bs: seed from insulin/carb action curves (white-box priors),
  refine with Dirichlet continual learning per patient — this is the
  "modular physiological world model": separate insulin-response and
  meal-response modules composed via ports, swappable per cohort.
- Goal prior c: peaked on in-range BG buckets with a steep penalty side
  toward hypo (asymmetric utility is the whole safety story).
- Meal detection falls out of state inference (COB posterior rising) rather
  than a threshold rule; the epistemic term prices probing actions —
  **cap it**: curiosity about hypoglycemia is contraindicated. A
  hard action-mask layer over the control posterior is non-negotiable.
- Runtime: `RollingAgent` at CGM cadence (3 min), per-patient instance in a
  `Colony`, checkpoints per patient per day (`.cffgz`).

## Honest risks

Tabular discretization vs. continuous physiology is the H4 wall; if bucket
models can't clear the bar, the continuous/Gaussian thread
(continuous-gaussian.md) becomes the prerequisite rather than a nice-to-have.
