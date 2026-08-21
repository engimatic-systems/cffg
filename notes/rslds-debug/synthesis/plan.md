# rSLDS Debug Synthesis

Review inputs: `codex/adapter.md`, `codex/heterogeneous.md`,
`independent-reviewer/adapter.md`, and
`independent-reviewer/heterogeneous.md`.

This plan is aimed at Cooper's parity criterion: CFFG should express the same
probabilistic factor architecture as the classic rSLDS and perform the NASCAR
task roughly as well as the independent paper-equation implementation. Raw
coefficient identity, sampler identity, and universal algorithm parity are not
required.

## Where the reviews agree

- The completed run was right to withhold its comparison. It is immutable
  diagnostic evidence, not a failed final verdict on CFFG.
- Classic adapter failure is concentrated in recurrence quantities. The
  heterogeneous adapter additionally enters genuinely different segmentation
  modes, including a dead-regime mode.
- The 1,280-step heterogeneous training path is too short for reliable
  source/gate coverage. Rank and conditioning are sound; rare event counts are
  the problem.
- CFFG's inner message solves are healthy. The heterogeneous convergence misses
  occur in the undamped outer parameter-replacement loop and in its fixed
  budget/certificate, with similar behavior in recurrence-free SLDS controls.
- Heterogeneous recovery is real but unreliable. It is best on observed support,
  worse on the common-state grid, and especially poor for rare sources and the
  native initialization. This is not evidence that CFFG ignores source rows.
- Calibration must block a freeze when mixing, coverage, convergence, or
  recovery fails. A capped extrapolation from already exhausted calibration
  runs is not a defensible budget.

## Useful additions from the independent review

1. **Use 6,400 total / 5,120 training steps as the heterogeneous candidate.**
   Codex found that 4,800 / 3,840 was the smallest tested coverage-only scale
   with a 0.99 lower confidence bound. The independent study found 300/300
   coverage passes at 5,120 training steps and also showed that the current
   recovery cap is near the reference posterior's identifiability limit at
   1,280 steps. The larger candidate is the better engineering choice because
   it provides margin for recovery as well as eligibility. Coverage at this
   scale is measured; improved recovery and convergence are still hypotheses
   to verify.
2. **Audit the convergence certificate before using it as a controller.** On a
   development fit, the single-observation initial-regime posterior produced a
   total residual of 0.338 when all other parameter families were at 0.099.
   That is valid evidence that one raw maximum can misrepresent scientifically
   relevant stability. Per-family and functional residuals should be retained
   before an outer globalization rule is selected.
3. **Treat a moderate shared recurrence prior as the leading adapter pilot.**
   Changing recurrence strength from 0.001 to 1.0 removed saturated gates and
   moved three of four heterogeneous chains into the good mode. This strongly
   supports a prior study in the shared CFFG/reference model. It did not remove
   the fourth collapsed chain or make transition functionals mix, so it is not
   yet the repair.
4. **Budget exhaustion is sometimes a cut through slow oscillatory progress.**
   Three development rSLDS fits converged by 16 iterations, while one same-K
   SLDS control converged at 22 after its residual first grew to 0.40. This
   justifies an honest tail study and a larger provisional development budget;
   it does not by itself prove that an unguarded full step is safe.
5. **The current 0.15 heterogeneous contrast target is not identifiable at
   1,280 steps.** Even good-mode reference posterior draws pass that cap only
   about half the time. The right response is to provide more information and
   better inference, not to select favorable final seeds.

## Points I would challenge

- I do not agree that only heterogeneous modal disagreement legitimately
  withholds the comparison. Classic label-canonicalized transition functionals
  still fail R-hat and effective-sample-size gates. Raw recurrence coefficients
  should become secondary, but the claim-bearing predictive, regime, supported
  transition, and rollout quantities must themselves be stable before they can
  support parity.
- I would not yet prescribe "change the prior, not the sampler." A stronger
  prior changes the probability model shared by CFFG and the adapter. The pilot
  is persuasive enough to test, not strong enough to register. It must preserve
  classic NASCAR behavior and must not merely make a collapsed posterior look
  concentrated.
- I would not simply exclude the initial-regime posterior or declare the
  development fit converged at iteration 10. A single initial state can still
  affect the inferred trajectory. Replace the raw maximum with a justified
  resolution-aware and functional certificate, then demonstrate that ignored
  coordinate movement does not change beliefs or claim-bearing outcomes.
- The evidence proves that more data fixes coverage. It does not yet prove that
  5,120 training steps fixes outer contraction or recovery identifiability;
  those conclusions currently rely on classic-family behavior and square-root
  extrapolation. Both require direct development checks at the chosen scale.
- Similar regime accuracy makes it plausible that CFFG reached the adapter's
  bad basin, but it does not prove common basin membership. Compare fitted
  occupancy, gate norms, likelihood, and transition surfaces directly before
  making that diagnosis durable.
- Keep the 0.15 recovery ceiling as a capability target unless Cooper changes
  the scientific claim. Calibration should establish that the target is
  attainable and set the required scale; it should not loosen the target until
  the current implementation passes.

## Unified execution plan

Legend: prerequisite `-->` dependent; braces are parallel work.

```text
{ cff-l5ag adapter eligibility
  cff-q1au heterogeneous scale
  cff-qulu outer certificate and learner
  cff-3v7q classic periodicity }
                    |
                    +--> cff-0oqo heterogeneous recovery
                                  |
                                  +--> new calibration and freeze
```

### 1. Repair experiment eligibility in parallel

**`cff-l5ag` — reference adapter**

- Replace raw recurrence coefficients as primary mixing monitors with
  label/decision-list-invariant, claim-bearing quantities. Retain raw
  coefficients as diagnostics.
- Run a development-only shared-prior study in the 0.1–1.0 recurrence-strength
  range on both families and both implementations. Preserve all density and
  transition oracles and classic task adequacy.
- Canonicalize initialization labels, but do not mistake that hygiene for mode
  handling. Use overdispersed chains and an explicit dead-regime detector.
- If the moderate prior, chosen scale, and honest draw budget do not remove
  heterogeneous collapse, test a real mode-moving strategy such as tempered
  warmup, registered restarts, or component-birth moves.
- Require R-hat at most 1.05 and effective sample size at least 50 on the
  claim-bearing quantities, stability after a longer-chain check, and no
  unreported collapsed chain before eligibility.

**`cff-q1au` — scale**

- Register 6,400 total / 5,120 training steps as the conservative candidate,
  reproduce the coverage study as a manifested experiment artifact, and
  preserve the classic generator golden and bounded-rollout checks.
- Benchmark CFFG and adapter memory/runtime at that scale before selecting the
  next protocol. This is a resource projection, not a reason to weaken a gate.

**`cff-qulu` — outer learning**

- First retain per-family parameter residuals, functional transition change,
  belief change, accepted step, and terminal inference. Repair the SLDS
  control's stale terminal-state/result mismatch at the same contract boundary.
- Define a convergence certificate that respects statistical resolution while
  still detecting consequential initial-regime movement. Keep the 0.1/two-hit
  rule until an alternative has an explicit meaning and regression evidence.
- Observe the full-step tail on new development seeds at the chosen scale. Use
  that result to set an honest provisional budget.
- Then implement one shared rSLDS/SLDS outer coordinator with safeguarded steps
  in posterior sufficient-statistic or natural-parameter coordinates. Do not
  use cross-replacement Bethe free energy as a line-search objective unless its
  comparability is derived and tested.

**`cff-3v7q` — classic periodicity**

- Resolve the narrow classic rollout miss independently. It is the only current
  direct classic task-adequacy miss attributable to CFFG rather than the
  reference adapter.

### 2. Re-evaluate recovery after the prerequisites

**`cff-0oqo` — heterogeneous recovery**

- On fresh development seeds at 5,120 training steps, compare both CFFG tiers
  with good-mode reference draws using occupancy, gate norms, regime accuracy,
  on-support transition error, and the common-state source/contrast metrics.
- Require both tiers to meet the fixed 0.15 worst-source and contrast ceilings.
  Report on-support and grid metrics together; neither substitutes for the
  other.
- If recovery still fails after certified convergence, localize source/gate
  sufficient statistics, feature scaling, prior geometry, and initialization.
  Add a registered restart or basin-selection rule if direct evidence confirms
  component collapse. Change recurrence mathematics only if an independent
  local oracle exposes an update error.

### 3. Earn the claims in stages

Do not spend fresh final seeds until all applicable development gates pass on
at least three declared seeds per family, every required tier/control is
present, and longer-chain or longer-iteration checks leave the conclusions
stable.

Run and report the claims independently:

1. **Classic parity:** same factor semantics plus roughly equivalent classic
   NASCAR performance against the paper-equation reference, with the classic
   adequacy outcomes passing.
2. **Heterogeneous extension:** the classic claim remains intact, and CFFG also
   recovers genuinely source-dependent transition functions below the fixed
   capability ceilings.

A heterogeneous miss must not erase a successful classic result. A classic
miss must not be reframed as heterogeneous parity. Each new final run needs a
fresh calibration, immutable registration, fresh final seeds, and complete
failure reporting.
