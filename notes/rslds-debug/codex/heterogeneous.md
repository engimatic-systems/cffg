# Heterogeneous CFFG Review

Reviewer: Codex

Ready for comparison: yes

## Scope

`cff-q1au`, `cff-qulu`, and `cff-0oqo`

## Observed evidence

I did not inspect `notes/rslds-debug/independent-reviewer/` during this review.

The immutable final evidence and hashes are the same as those recorded in the
adapter review. The final protocol authorized CFFG commit
`a686faaddbfe0d980b99517d02ab1664f2333a12`.

The evidence separates into three different problems. They should not be
treated as one heterogeneous-engine failure.

### Coverage: the selected trajectory length was not reliable

The registered training path had 1,280 time steps. Two calibration paths had
minimum binary-outcome counts of nine, so the scale selector accepted it with
nominal 1.5-times headroom over the required five. Untouched paths then
produced minima of six, four, and twenty-two. Seed 2102 correctly stopped
before fitting because source 3, gate 2 had only four successes among 51
exposures.

The selector inferred reliability from only two correlated trajectories. That
was insufficient for the ticket's desired 0.99 dataset-eligibility
probability.

I ran a generator-only audit on 300 new development seeds 40000 through 40299.
It generated each 6,400-step legacy stream once and evaluated its prefixes. It
did not fit a model, inspect another review, use a final seed, or write an
artifact.

| Total steps | Training steps | Eligible | Failures | One-sided 95% lower bound |
|---:|---:|---:|---:|---:|
| 1,600 | 1,280 | 272/300 | 28 | 0.87428 |
| 2,400 | 1,920 | 291/300 | 9 | 0.94823 |
| 3,200 | 2,560 | 296/300 | 4 | 0.96975 |
| 4,800 | 3,840 | 300/300 | 0 | 0.99006 |
| 6,400 | 5,120 | 300/300 | 0 | 0.99006 |

The lower bounds are exact one-sided 95% Clopper-Pearson bounds. At 4,800
steps, the minimum outcome count across all 300 datasets was six, the 1st
percentile was 7.98, and the minimum source/gate exposure was 103. This makes
4,800 total and 3,840 training steps the smallest measured candidate that
meets `cff-q1au`'s reliability target on this declared development set.

### Convergence: inner inference is healthy; the outer map is not globalized

Classic is the useful negative control: all six primary rSLDS fits and all six
SLDS controls converged.

For heterogeneous seed 2101, both primary fits converged, at residuals 0.0701
and 0.0725. Both controls exhausted sixteen updates, at 0.2770 and 0.1048.
Seed 2102 did not fit because of coverage. For seed 2103, both primary fits
exhausted sixteen updates, at 0.2251 and 0.4840, and both controls exhausted at
0.1701 and 0.2218.

Every failed primary hidden-state solve converged on its first inner attempt
with message damping 0.9. There were no inner retries. Seed 2103 shared used
22,985 inner iterations across seventeen healthy solves; native used 20,216.

The outer residual traces do not look like a uniform one-iteration shortfall:

- shared minimum residual was 0.1274 and ended at 0.2251 after oscillation;
- native minimum was 0.1178 and then grew to 0.4840;
- the two-hit 0.1 rule correctly accepted seed 2101 only after two consecutive
  sub-0.1 values.

Coverage does not explain this contrast. Failed seed 2103 had better ground-
truth coverage and conditioning than converged seed 2101: minimum
exposures/successes/failures were 76/25/22 versus 55/6/10.

Calibration had already exposed the instability. At its cap of twelve, only
4/8 heterogeneous primary fits and 4/8 heterogeneous controls converged,
compared with 8/8 classic primary fits. The final cap of sixteen was calculated
as the maximum observed iteration count plus four, but the maximum included
right-censored, budget-exhausted fits. It was not an observed convergence
time.

In `src/cffg/models/rslds_fit.py`, `damp` applies only to inner
`Engine.run`. After a healthy inner solve, every conjugate posterior is
recomputed from the immutable prior and installed at full step. There is no
outer damping, line search, trust region, or cycle handling. The matched SLDS
control fitter uses the same full replacement.

The recorded inner Bethe free energies cannot serve as an outer line-search
objective. Each value belongs to a different fixed-parameter inner problem and
does not include a complete parameter-posterior objective. On failed native
seed 2103, the values became more negative while the fixed-point residual grew.

The artifacts retain only the scalar maximum residual, not per-parameter
deltas. They therefore prove the mechanism class but cannot identify whether
recurrence, dynamics, emission, label permutations, or an affine gauge
dominates the raw residual.

There is also a separate control-fitter defect. `fit_slds` returns beliefs from
the inference immediately before its last parameter installation while
returning the new parameters. It lacks rSLDS's terminal solve. This does not
cause the next-iteration residual failure, but it makes returned control
beliefs and parameters one coordinate inconsistent.

### Recovery: capability exists, but it is not reliable

The final converged seed 2101 fits had small affine-aligned latent errors,
0.0089 and 0.0091. The scoring oracles recover exact truth under arbitrary
invertible affine coordinates and complete label permutations to below
`1e-14`. A source-pooled model fails those same tests. This makes an affine or
label-scoring defect an unlikely primary explanation.

Shared initialization scored:

- held-out on-support transition RMSE 0.0754 overall, with per-source values
  0.0361, 0.0635, 0.0982, and 0.1216;
- common-grid worst-source RMSE 0.1946; and
- common-grid worst source-contrast RMSE 0.2498.

The gap shows that much of the miss is uncertainty or extrapolation away from
the naturally occupied source/state combinations. The common grid is
deliberately more demanding: every source is evaluated on every registered
gate-boundary state.

Native initialization had a genuine occupied-support failure for source 3:
0.4235, with common-grid worst-source and contrast errors 0.3761 and 0.3736.
That cannot be dismissed as counterfactual scoring.

Calibration demonstrates that recovery is possible but unstable. On seed
3101 replicate 1, both tiers met the 0.15 functional caps:

- native worst-source 0.1239 and worst-contrast 0.1459;
- shared worst-source 0.1037 and worst-contrast 0.1391.

Other calibration replicates ranged as high as 0.4292/0.6235 shared and
0.4072/0.5991 native. Several of the best and worst recovery results had the
same outer terminal status. Therefore convergence certification and recovery
accuracy are related reliability problems, but one does not substitute for
the other.

The local recurrence learner is source-specific rather than pooled. Its
direct Pólya-Gamma batch oracle recovers two distinct source rows to weight
tolerance 0.01 and offset tolerance 0.005, and the engine aggregation matches
the direct natural-parameter update. The final learned model also produced
different source behavior. There is no current evidence of a broadcast-row
defect or a basic recurrence-statistics indexing error.

## Competing explanations

1. **Coverage scale selection from two paths — certain for `cff-q1au`.** The
   300-seed audit reproduces the failure rate and supplies a measured remedy.
2. **A noncontractive, oscillatory, or symmetry-sensitive full outer map —
   high for `cff-qulu`.** Jagged residuals, healthy inner solves, and failures
   in both rSLDS and ordinary SLDS controls support this.
3. **Initialization-sensitive local optima — high for recovery.** Recovery
   varies widely across replicate and tier while latent affine fits stay good.
4. **Insufficient posterior-effective source/gate evidence — medium-high.**
   Ground-truth cells can pass at only five or six minority outcomes, and soft
   inferred responsibilities may supply less effective information. The
   common-state grid demands coefficient accuracy beyond occupied support.
5. **Raw residual measures a label or affine gauge rather than a functional
   change — medium.** The artifact lacks vector and aligned-functional traces,
   so this must be tested before selecting an outer controller.
6. **The registered convergence rule is too strict — low.** A 0.1 tolerance is
   already loose, and two hits protect against the observed oscillation.
7. **An incorrect functional scorer or source-pooled recurrence update — low.**
   Exact affine/label oracles and direct source-specific learning oracles argue
   against both.
8. **A larger fixed outer budget is the complete fix — low.** It may rescue
   boundary cases such as residual 0.1048, but it is not justified for growing
   seed-2103 traces.

## Discriminating checks

1. On heterogeneous calibration seed 3101 replicate 1, retain per-array and
   grouped outer residuals for initial law, dynamics, recurrence/transition,
   and emission. Also retain the residual-vector direction, label-aligned
   transition-function change, affine/gauge change, inner status, iterations,
   and inner BFE. Use one rSLDS tier and the same-K control first.
2. Extend that diagnostic full-step run to 32 or 48 outer updates without
   changing the stopping rule. Decay just beyond the old cap indicates a
   budget issue; alternating directions show a cycle; one dominant block
   locates ownership; raw-large but functional-small indicates a symmetry or
   certificate problem.
3. On the identical development dataset and initialization, compare the full
   step with an adaptive under-relaxed prototype in valid exponential-family
   natural coordinates. Re-run inner inference to measure the trial residual.
   Do not choose steps from the inner BFE.
4. If necessary, freeze recurrence and Gaussian blocks in separate diagnostic
   ablations. Failure in the SLDS control excludes recurrence as the only
   cause; rSLDS-only recurrence oscillation locates the additional coupling.
5. After outer stabilization, fit several new development paths at 3,840
   training steps. Decompose functional error by source, gate, grid region,
   initialization tier, inferred effective exposure, and observed support.
6. If source recovery remains poor with stable convergence and robust data,
   compare recurrence posterior sufficient statistics with an exact small
   oracle and evaluate standardized-feature parameterization or a shared
   regularizing prior. Any prior change must apply to both CFFG and the adapter.

## Findings

`cff-q1au` is an experiment-scale defect, not a CFFG defect. A 1,280-step
training trajectory does not make the eligibility contract reliable. The
smallest candidate measured here is 3,840 training steps.

`cff-qulu` is a real CFFG learning-workflow limitation shared with the ordinary
SLDS control fitter. Inner message passing is healthy; the outer full
parameter-replacement iteration lacks globalization and sometimes oscillates
or grows. More diagnostics are needed to distinguish functional cycling from
raw-coordinate symmetry before implementing the controller.

`cff-0oqo` is a reliability problem, not evidence that source heterogeneity is
missing from the engine. The engine can recover below the cap on some
development replicates, and its local source-specific update is correct under
direct oracles. At the registered scale it does not recover reliably across
datasets and initializations.

## Recommended repair and validation

Proceed in this order:

1. **Close the coverage-design gap.** Treat 4,800 total/3,840 training steps as
   the current candidate. Reproduce the declared 300-seed audit as a manifested
   calibration artifact, preserve generator surfaces and bounded rollouts, and
   measure fit resources before freezing it.
2. **Instrument and globalize the outer map.** Build one shared outer
   fixed-point coordinator for rSLDS and SLDS. Separate `inner_damp` from outer
   step control. Under-relax posterior sufficient statistics or natural
   parameters, not arbitrary covariance point arrays. Use safeguarded
   residual-merit evaluation and retain step size, rejections, per-block
   residuals, cycle diagnostics, inner evidence, and terminal state.
3. **Repair the SLDS terminal contract.** Run inference after the final
   parameter installation and retain the same phase diagnostics as rSLDS.
4. **Re-evaluate recovery before changing recurrence mathematics.** Robust
   coverage and stable outer learning may be sufficient. Require both
   initialization tiers to meet worst-source and contrast RMSE at most 0.15 on
   a declared set of new development paths.
5. **Only then localize any remaining recurrence error.** Inspect inferred
   source/gate effective exposure, source-local sufficient statistics, feature
   scaling, prior geometry, and initialization ensembles.

Do not weaken the 0.1/two-hit convergence rule, treat budget exhaustion as
success, use inner BFE as an outer objective without a derivation, or tune on
the completed final seeds.

A new final run is justified only after coverage, convergence, controls, and
recovery all pass on newly declared development evidence. It then needs a new
calibration, immutable freeze, artifact identity, and fresh final seeds.

## Risks and disconfirming evidence

- If extended full steps converge smoothly just after sixteen on several new
  development paths, downgrade noncontractivity and make budget derivation the
  primary issue.
- If label/affine-aligned functional residual is already below threshold while
  raw parameter residual cycles, repair canonicalization or the convergence
  certificate rather than damping a scientifically stable model.
- If one parameter-block statistic disagrees with an independent oracle,
  repair that update before adding a general controller.
- If 3,840-step training paths pass coverage but recovery does not improve
  after stable convergence, the limitation is recurrence identification or
  inference rather than data volume alone.
- If an adaptive step reduces raw residual while worsening a properly derived
  total variational objective or fails across new development paths, reject
  that controller.
- If controls remain unstable after shared globalization while rSLDS is
  stable, separate misspecified-control behavior instead of weakening the
  primary rSLDS contract.

## Confidence

- Very high that the original coverage scale was unreliable and 3,840
  training steps is the smallest candidate passing this 300-seed audit.
- High that failed primary fits are outer-map failures with healthy inner
  inference and no outer globalization.
- Medium-high that adaptive residual globalization is the right repair class.
- High that recovery is initialization- and dataset-sensitive rather than a
  broadcast-row or scoring defect.
- Medium that robust coverage plus outer stabilization will be sufficient to
  meet the recovery cap; this must be tested rather than assumed.
- Low-medium on which parameter family dominates outer cycling until per-block
  traces are retained.
