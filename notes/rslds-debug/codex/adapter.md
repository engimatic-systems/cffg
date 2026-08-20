# Reference-Adapter Mixing Review

Reviewer: Codex

Ready for comparison: yes

## Scope

`cff-l5ag`

## Observed evidence

I did not inspect `notes/rslds-debug/independent-reviewer/` during this review.

The immutable evidence is the completed experiment at commit
`6c0437a8ec282a96b2e906063b35cdabfca1f17b`. The aggregate result is
`artifacts/rslds-confirmatory/results.json`, SHA-256
`eaf4bdccb6b2d9ecf35f59a91fa1012873bba697c8fe499781b28c1f787697c4`.
The manifest SHA-256 is
`b71c41ad46cd2e210a9e204743d8fd58f1a76a69305afae8601c7e307b64a56a`.

All forty adapter chains completed. They form ten four-chain arms, each with
400 warmup iterations and 1,600 retained draws. Every arm failed the registered
mixing requirement of split R-hat at most 1.05 and effective sample size at
least 50.

The adapter monitors 43 label-order statistics, constructed at
`reference/legacy-runtime/src/legacy_runtime/comparison_adapter.py:415`:

1. mean log likelihood per time step;
2. twelve sorted recurrence-weight norms;
3. twelve sorted recurrence offsets;
4. eight sorted dynamics-covariance eigenvalues; and
5. ten sorted emission-covariance eigenvalues.

### Classic arms

The classic failure is confined to recurrence parameters:

- Across six arms, worst R-hat ranged from 1.666 to 2.143 and minimum
  effective sample size from 5.358 to 7.110.
- All six likelihood variables passed both gates.
- All 48 dynamics-covariance variables passed both gates.
- All 60 emission-covariance variables passed both gates.
- Of 144 recurrence variables, only nine passed R-hat and twelve passed
  effective sample size.
- A sorted recurrence offset was the worst R-hat variable in every arm.

This is slow, nonstationary recurrence exploration rather than simple
between-chain label disagreement. For classic seed 2001 under shared
initialization, three chains' worst-offset retained-half means moved from
`53.9` to `73.2`, `50.1` to `78.0`, and `49.6` to `69.3`. Their lag-one
correlations were 0.80, 0.89, and 0.81. During the same draws, per-step log
likelihood had R-hat 1.004 and effective sample size 2,021.

The classic generator has large, nearly deterministic gate coefficients. The
fitted model estimates separate source-and-gate rows with a weak Gaussian
recurrence prior of precision 0.001, or marginal variance 1,000 before data.
Rare or separated binary gate cells can therefore leave very long coefficient
tails. Pólya-Gamma Gibbs updates are moving slowly along those directions even
though the likelihood and Gaussian covariance summaries are stable.

### Heterogeneous arms

The heterogeneous failure is broader and contains genuinely different modes:

- Worst R-hat ranged from 3.217 to 9.623 and minimum effective sample size
  from 4.055 to 4.434.
- Seed 2101 native was dominated by a recurrence-norm statistic at R-hat
  4.523.
- Seed 2101 shared was dominated by per-step likelihood at R-hat 3.288.
- Seed 2103 native was dominated by a dynamics-covariance eigenvalue at R-hat
  9.623; likelihood reached 7.036.
- Seed 2103 shared was dominated by a recurrence offset at R-hat 3.217;
  dynamics covariance reached 2.998 and likelihood 2.312.

For seed 2101 shared, chains zeroed different components after starting with
all four occupied. Two chains retained approximately three-regime occupancies
`[402,546,332,0]` and `[402,548,0,329]`; two retained all four near
`[403,535,208,134]`. Their per-step likelihood means split into modes near
26.53 and 27.01. For seed 2103 native, one chain likewise lost a component and
settled at a materially lower likelihood. Empty components return toward broad
prior parameter draws and are difficult for the current local Gibbs updates to
revive.

Label switching cannot change log likelihood or sorted covariance spectra.
Those failures prove that the heterogeneous chains are not merely alternative
names for the same posterior state.

### Diagnostic validity and protocol selection

The current split R-hat and effective-sample-size implementations are
recognizable classical diagnostics and pass their independent-IID regression.
They are not modern rank-normalized and folded diagnostics, but failures this
large remain decisive.

There is a narrower representation issue. Sorting raw recurrence norms and
offsets removes array permutations, but a destination relabeling in a
sequential decision list generally requires a nonlinear re-expression of its
sticks. The current invariance test only reverses array axes. Raw coefficient
order statistics can therefore reject representationally equivalent
transition functions, especially in weakly identified directions. This does
not explain component death, likelihood disagreement, or covariance-spectrum
disagreement.

Calibration had already reported worst R-hat 9.214 and minimum effective
sample size 4.117. Its linear projection requested 2,430 retained draws, but
the protocol then applied a 2,000-total-iteration cap and used only 1,600
retained draws after warmup. Prefix analysis shows that a modest extension
would not solve the problem: heterogeneous maximum R-hat stayed around 4.5 or
10 from 200 through 1,600 draws, while classic values were nonmonotone as slow
coefficient drift became visible.

## Competing explanations

1. **Classic gate separation plus a weak recurrence prior — high.** This
   explains large drifting offsets while likelihood and Gaussian summaries
   mix well.
2. **Heterogeneous component death and posterior-mode trapping — high.** This
   is directly visible in occupancy, invariant likelihood, and covariance
   spectra.
3. **Incomplete decision-list canonicalization — medium.** It may inflate some
   recurrence diagnostics but cannot explain the invariant failures.
4. **The chain budget is merely too short — low as a complete explanation.**
   Prefix diagnostics do not improve with retained length, and some modes are
   internally stable but mutually different.
5. **An algebraic adapter defect — low.** Recurrence, Pólya-Gamma auxiliary,
   dynamics, emission, and destination-indexed updates have independent local
   oracle coverage. This remains falsifiable by the isolated microchain below.

## Discriminating checks

Use calibration seeds 3101/3102 or newly declared development seeds. Do not use
final seeds to choose a repair.

1. Hold development latent states and regimes fixed and sample only
   Pólya-Gamma auxiliaries and recurrence coefficients. Compare the current
   chain with an independently implemented logistic-posterior sampler and an
   exactly transformed standardized-feature parameterization. This isolates
   separation and augmentation from SLDS state coupling.
2. On one heterogeneous development dataset, retain component occupancy during
   warmup for a small factorial: full Gibbs, fixed latent states/regimes, fixed
   dynamics/emission, truth-started diagnostic, and observation-only start.
   This locates the update coupling responsible for component death.
3. Align regimes through observation-space dynamics and evaluate complete
   destination-probability tensors on a frozen observed-support grid. Verify
   that all 24 equivalent regime relabelings give identical diagnostics while
   a collapsed component and a perturbed transition surface remain distinct.
4. Compare staged or tempered warmup, parallel tempering or component-birth
   moves, and a collapsed state update on development data. Require actual
   mixing; do not select a method from chain completion alone.
5. Double retained draws after a candidate passes. Function-space and
   downstream posterior summaries should remain stable rather than merely
   staying inside a threshold at one budget.

## Findings

The frozen result was correct to withhold the reference comparison.

Classic adapter mixing is a narrow recurrence-coefficient problem. It is
plausible that observable transition functions and the primary predictive
estimands are already stable, but the current evidence does not measure that.
The next diagnostic should establish this in function space before deciding
whether the coefficient tails require a new sampler.

Heterogeneous adapter mixing is a substantive sampling failure. Some chains
lose a regime and occupy likelihood and covariance modes that cannot be
removed by relabeling. A better canonicalizer is necessary for sound
diagnostics but is not sufficient to repair these chains.

## Recommended repair and validation

For classic, first add label- and decision-list-invariant transition-function
diagnostics on a frozen support grid and chain-wise versions of the downstream
comparison estimands. Keep raw coefficients secondary. If function-space
mixing fails, repair the separated recurrence sampler or use a better
parameterization. If it passes, a replacement preregistration may justify
mixing eligibility in terms of the quantities that support the parity claim.

For heterogeneous, add an update capable of moving between occupied and
collapsed-component modes. Tempered warmup, split/merge or component-birth
moves, or collapsed state updates are credible classes; the development
factorial should choose among them.

Changing recurrence-prior strength is not an adapter-only fix. It changes the
probabilistic model and must be changed identically in CFFG and the adapter,
then recalibrated and frozen under fresh identities.

Before another freeze, both families, both initialization tiers, and four
overdispersed development chains should:

- pass R-hat at most 1.05 and effective sample size at least 50;
- retain every required component;
- pass function-space and downstream-estimand diagnostics;
- remain stable after retained draws are doubled;
- canonicalize all equivalent relabelings without merging genuine modes; and
- preserve all independent adapter density and update oracles.

Only then should a new protocol and entirely fresh final seeds be frozen.

## Risks and disconfirming evidence

- If the fixed-state recurrence microchain mixes under the current prior, the
  classic problem comes from latent-state coupling rather than separation in
  the coefficient update alone.
- If rigorously canonicalized transition functions still have poor R-hat and
  effective sample size in classic, the slow raw coefficients are not harmless
  nuisance directions.
- If collapsed heterogeneous chains freely regain the missing component during
  longer development runs, component death is transient and the main issue is
  warmup selection. The retained final prefixes currently argue against this.
- If an independent conditional-density oracle disagrees with the adapter on
  the fixed-state microchain, repair that algebra before changing the sampler.
- A stronger prior that mixes well but is not mirrored in CFFG cannot support a
  same-model comparison.

## Confidence

- High that classic mixing failure is recurrence-specific slow exploration.
- High that heterogeneous chains contain genuine component-collapse modes.
- Medium-high that separation and the weak recurrence prior dominate classic.
- Medium that incomplete decision-list canonicalization materially inflates
  some recurrence diagnostics.
- Low that an algebraic adapter defect is the primary cause.
