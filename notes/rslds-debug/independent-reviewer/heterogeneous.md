# Heterogeneous CFFG Review

Reviewer: independent-reviewer (Claude, 20 August 2026 session)

Ready for comparison: yes

## Scope

`cff-q1au`, `cff-qulu`, and `cff-0oqo`

## Observed evidence

All evidence comes from the immutable cff-oyie artifact
(`artifacts/rslds-confirmatory`, results.json SHA-256 `eaf4bd…697c4`),
the frozen calibration artifact (`artifacts/rslds-dual-calibration`), the
retained reference-adapter chains for seed 2101 (inspected read-only to
diagnose the recorded failure), and development-seed studies on fresh
seeds (4201 to 4203 for fitting, 5000 to 5499 for coverage). No final
seed tuned any proposed change. I verified that regenerating the seed-2101 dataset
reproduces the recorded observation digest before using its truth states.

### Coverage (cff-q1au)

The frozen contract requires, for every source and gate cell: at least 20
exposures, 5 successes, 5 failures, design rank 3, design condition at
most 100, and Fisher condition at most 1,000.

Recorded final-seed margins show the contract was already marginal:

- 2101: eligible with minimum successes 6 (cell source 3 gate 2) — a
  one-event margin
- 2102: not ready, source 3 gate 2 had 4 successes
- 2103: comfortable (minimum successes 25); trajectory-to-trajectory
  variation is large

Development study (500 fresh seeds, 5000 to 5499, registered generator and
validator code unchanged): at the frozen 1,280-step training length,
456 of 500 datasets pass (91.2%; one-sided 95% lower confidence bound
0.888). The chance that 3 fresh final seeds all pass is about 0.76, so
about a 1-in-4 chance of at least one not-ready dataset per fresh freeze.
The 2102 miss was predictable bad luck, and 2 calibration seeds could
never have shown this (they measure a 9% event with 2 samples).

The failure is structural, not one unlucky seed. Across the 44 failing
datasets at 1,280 steps (some fail more than one check), the failing
checks are: source 2 gate 2 too few failures (33 datasets), source 3 gate
2 too few successes (28), source 3 gate 0 too few successes (1). Every
binding cell is a rare-outcome cell of a rare source, and the same 2
cells that broke seeds 2101 and 2102. Rank and conditioning never failed
in any of the 500 datasets, so the generator's design is sound and only
event counts are short.

Longer training paths repair reliability without touching the generator:

- 2,560 steps: 297 of 300 pass (99.0%; 95% lower bound 0.974)
- 5,120 steps: 300 of 300 pass (100%; 95% lower bound 0.990)

So 5,120 steps meets the registered reliability requirement (one-sided
95% lower bound at least 0.99) on exactly the declared evidence, and
2,560 does not.

### Outer convergence (cff-qulu)

The outer loop (`fit_rslds`, and the control loop in the validation
repository's `slds.py`) is an undamped fixed-point iteration: infer hidden
state to inner convergence, replace every conjugate posterior in full,
reinstall point parameters, repeat. The registered `damp=0.9` applies only
to inner message passing. There is no step control, no monotonicity check,
and no globalization on the outer replacement. Convergence requires 2
consecutive scale-normalized residuals at most 0.1.

Recorded traces separate the inner solver from the outer loop cleanly. On
seed 2103 (shared tier, budget exhausted): the inner solver converged on
all 17 solves, while the outer residual bounced in 0.13 to 0.33 for 15
iterations with no trend, and recorded free energy moved monotonically in
one direction after the first replacement (26,469 to 27,312 in magnitude).
On 2101 (converged, 9 iterations) the residual tail decreases smoothly
into the threshold. So the stall is in the coordinate-replacement loop,
confirming the ticket's framing that the failure is not in the message
solver. Whether the loop wanders forever or merely takes longer than the
budget is answered by the development runs below: it takes longer.

Two further recorded facts constrain the explanation:

- the learned SLDS controls, which have no recurrence factor at all,
  budget-exhausted on 2101 and 2103 with the same loop structure. The
  instability is a property of full-replacement coordinate iteration at
  the 1,280-step scale, not of the recurrence family alone.
- every classic arm (10,000 steps) converged in 7 to 14 iterations, and
  heterogeneous 2101 converged in 9 to 12. The fixed-point map is
  contractive at large data scale and only marginally stable at 1,280
  steps, where posterior segmentation uncertainty is much larger.

The same behavior appeared at calibration: CFFG budget-exhausted on
calibration seed 3101 (both shared replicates) and 3102 (one native
replicate), and controls exhausted on both calibration seeds. The frozen
outer budget (16 = calibration maximum 12 plus 4) was set from evidence
that already contained exhaustion.

The recorded free-energy trace disagrees with the residual about progress
even on converged fits (classic 2001 free energy magnitude grows by
14,000 over 12 iterations while the residual falls to 0.05). The
diagnosis step the ticket asks for (do free energy, residual, and
parameter displacement agree) should treat the free-energy ledger's
cross-replacement comparability as an open question to settle first.

Instrumented development reproduction (seed 4201, frozen configuration,
budget raised to 48 outer iterations only to observe the tail, per-family
residual attribution via the registered residual code):

- rSLDS: converged at 12 iterations (inside the frozen 16). Early
  residuals are dominated by the inverse-Wishart scatter posteriors
  (`psi` of dynamics and emission), late residuals by the recurrence
  offsets. The residual is non-monotone on the way down (0.21 to 0.30
  bump) but contracts.
- same-K SLDS control (no recurrence factor): converged at 22 iterations
  — under the frozen budget of 16 this exact fit would be recorded
  budget-exhausted, reproducing the recorded control failures. The
  residual trace shows the mechanism: it falls to 0.13 by iteration 10,
  swells back to 0.40 by iteration 17, then collapses to 0.03 by 22. The
  dominating family is `psi` (dynamics and emission scatter) at every
  iteration; recurrence cannot be the cause because the model has none.
  The recorded free energy moves smoothly through the residual swell, so
  the loop spirals in parameter space while slowly drifting — a
  marginally stable, undamped fixed-point iteration, not divergence.

Two further development seeds (4202, 4203) converged at 16 and 12
iterations. Seed 4202 converged on exactly the last permitted iteration
after a residual swell (0.14 at iteration 8 rising to 0.23 at iteration
11); one more bump and the frozen protocol would have recorded it as
budget exhausted. So the recorded exhaustions are the tail of a
distribution whose centre already sits at the budget boundary, not a
separate failure class.

The per-family attribution exposes a second, independent defect in the
convergence criterion. The residual is a maximum over every parameter
family, scale-normalized, and it includes the initial-regime Dirichlet
posterior. That learnable is declared with 1 source and 1 target: it is
estimated from a single time step's belief about the first regime
(confirmed in the recorded `declared_learnables` of the final run). Its
value therefore moves by an O(1) amount whenever the inferred first
regime shifts, and it never contracts with more data, because more data
does not give it more observations.

On seed 4203 this single family decided the iteration-10 outcome: the
recorded residual was 0.338, but excluding the initial-regime family it
was 0.099 — below the 0.1 threshold. Every other family had effectively
converged, and one single-observation posterior tripled the residual and
reset the consecutive-iteration counter. The same family drove the
residual on 3 of 12 iterations for that seed, and on seed 4201 at
iteration 4.

This means the residual has a noise floor set by its worst-resolved
family. No amount of globalization or extra budget removes it, and it
gets relatively worse as the threshold tightens.

This gives cff-qulu its acceptance-criterion-1 regression: a
deterministic development fit whose outer residual leaves the threshold
neighborhood and returns, crossing the frozen budget boundary, with the
responsible parameter family named by the instrumentation.

### Source recovery (cff-0oqo)

Recorded seed-2101 decomposition (grid metric, 108 states):

- shared tier per-source transition RMSE: 0.040, 0.054, 0.195, 0.136
  against the 0.150 cap; worst contrast 0.250 (pair 2 to 3)
- native tier per-source: 0.081, 0.112, 0.194, 0.376; worst contrast 0.374
- source data volumes: 403, 534, 209, 133 transitions. Error rises
  monotonically as source data falls
- held-out on-support per-source RMSE, shared tier: 0.036, 0.063, 0.098,
  0.122 — all under the cap on realized states. The grid metric roughly
  doubles source 2's error (0.098 to 0.195) because the balanced grid
  includes states the rare sources never visit, where the error is
  parametric extrapolation
- native tier on-support source 3: 0.423 — a qualitatively different,
  catastrophic miss, not an extrapolation artifact
- latent alignment is excellent in both tiers (affine-aligned latent RMSE
  0.009), so alignment contributes almost nothing

Identifiability limit from the reference posterior (read-only diagnosis of
the recorded failure): I evaluated the registered functional-recovery
metric on retained draws of the seed-2101 adapter chains, with per-draw
alignment and labels.

- good-mode chains (all 4 regimes alive, the higher-likelihood mode):
  worst-source RMSE has posterior mean 0.120 (standard deviation about
  0.023). Draws meet the 0.150 worst-source cap only 88 to 90% of the
  time. Draws meet the 0.150 worst-contrast cap only 46 to 47% of the
  time. This is the exact reference model sampling from its own
  posterior: at 1,280 steps the caps sit at, and for the contrast below,
  the identifiability limit
- collapse-mode chains: worst-source RMSE 0.49 — the dead regime's row is
  pure prior
- for context, the truth contrast signal is only 0.18 to 0.31 RMS, so a
  0.15 cap on contrast error demands the contrast be resolved to roughly
  half its own size

The threshold derivation confirms the caps were never evidence-based at
this scale: the calibration rule was min(0.15, max(0.0075, 1.25 times the
maximum calibration error)), and the observed calibration errors were 0.41
to 0.62, so the domain boundary 0.15 overrode the calibration evidence in
every cell (`threshold_derivation` in the calibration artifact).

Segmentation quality ties the CFFG gap to the mode structure found in the
adapter review: the reference good-mode chains reach label-aligned regime
accuracy 0.994 on 2101, the reference collapse-mode chain reaches 0.877,
and both converged CFFG tiers sit at 0.852 to 0.862 — at the collapse-mode
level, not the good-mode level. CFFG's converged fits appear to live in
(or near) the same suboptimal basin that traps half the Gibbs chains,
which pollutes the rare sources' sufficient statistics.

Shared and native tiers therefore fail for overlapping but not identical
reasons: both lose accuracy on the two rare sources (data volume), and the
native tier additionally destroys source 3 outright (0.423 on-support),
consistent with an initialization that merges or misroutes the rarest
regime.

## Competing explanations

Coverage:

1. Bad luck on one seed against an otherwise-safe generator. Falsified:
   the per-dataset failure rate at 1,280 steps is about 9%, and the
   binding cell (source 3 gate 2) is structural — source 3 is the rarest
   regime and gate 2 its rarest outcome.
2. Generator defect. No evidence: rank and conditioning pass everywhere in
   500 development seeds; only event counts fail.

Outer convergence:

1. Inner message solver fails at scale. Falsified: inner status converged
   on every recorded solve, including all budget-exhausted fits.
2. The recurrence family destabilizes the loop. Insufficient: the
   recurrence-free SLDS controls exhaust the same way at the same scale.
3. The undamped full-replacement fixed-point map loses contraction at
   small data scale, where segmentation uncertainty makes successive
   replacements overshoot; the residual then swells without a Lyapunov
   guard. Confirmed: on 3 development seeds the residual falls, swells,
   and falls again, and the SLDS control needed 22 iterations against the
   frozen budget of 16.
4. The convergence criterion itself is badly formed: a maximum over
   families of wildly different statistical resolution, including a
   single-observation Dirichlet that never contracts. Confirmed as a
   real contributing cause on development seed 4203 (residual 0.338
   against 0.099 without that family). Not the whole story: the SLDS
   control's swell to 0.40 was driven by the dynamics and emission
   scatter posteriors, and the recorded 2103 residuals bounce in 0.13 to
   0.33 while free energy still moves. Both defects are present and both
   need repair.

Recovery:

1. Insufficient rare-source data at 1,280 steps against caps set by a
   domain boundary rather than calibration evidence. Supported by the
   posterior-draw study (contrast cap below the identifiability limit)
   and by the calibration threshold clamp.
2. CFFG learning defect in recurrence statistics or replacement. Not the
   primary driver for the shared tier: on-support errors are under cap
   and the residual gap to the good-mode posterior (about 0.07 on the
   grid metric) tracks segmentation accuracy, not the recurrence update
   itself. The native-tier source-3 collapse is a real fit-quality defect
   (basin selection), shared with the reference sampler's collapse mode.
3. Alignment or label defects. Falsified: latent RMSE 0.009, complete
   label permutations, and the label-accuracy pattern is explained by
   basin membership.

## Discriminating checks

1. Coverage reliability curve over 500 fresh seeds at 5 training lengths
   (separates bad-luck from structural shortfall, and measures how much
   length buys reliability).
2. Recorded inner-status against outer-residual traces on every
   budget-exhausted arm (localizes the stall to the replacement loop).
3. Control-arm comparison (removes the recurrence family as a necessary
   cause).
4. Instrumented development fits (3 rSLDS seeds and one SLDS control) with
   per-family residual attribution at 48 outer iterations (tests
   oscillation against slow convergence, names the responsible family,
   and measures how close the frozen budget sits to the boundary).
5. Recomputing each development residual with the initial-regime family
   excluded (isolates the criterion defect from the loop defect).
6. Posterior-draw evaluation of the registered recovery metric from the
   reference chains (separates learner error from identifiability limit).
7. On-support against grid-metric comparison per source (separates
   estimation error from extrapolation amplification).
8. Regime-accuracy comparison across reference modes and CFFG tiers
   (places CFFG fits in the basin structure).

## Findings

1. Coverage: the 1,280-step design misses the frozen coverage contract on
   about 9% of datasets, concentrated in the rarest source's rarest gate.
   Seed 2102 was an expected draw from that distribution. Two calibration
   trajectories could not have detected this.
2. Outer convergence: two independent defects compound.
   First, the outer loop is an undamped full-replacement fixed-point
   iteration with no globalization. At 1,280 steps it is only marginally
   contractive: residuals fall, swell, and fall again, and the frozen
   budget of 16 cuts through the middle of that oscillation. The
   recurrence-free SLDS control reproduces it (converged only at
   iteration 22 after swelling back to 0.40), so the recurrence family is
   not necessary for the failure, and the inner solver converged on every
   recorded and reproduced solve.
   Second, the convergence criterion is mis-formed. The residual maximizes
   over parameter families that include the initial-regime Dirichlet,
   which is estimated from one time step and never contracts. On
   development seed 4203 it alone turned a converged iteration (0.099)
   into a failing one (0.338).
   Both were visible at calibration, and the frozen budget rule
   (calibration maximum plus 4) converted known exhaustion into the
   registered budget.
3. Recovery: at 1,280 steps the registered caps are at or beyond the
   identifiability limit for the rare sources — the exact reference
   posterior fails the contrast cap on about half its own draws in its
   best mode. On top of that limit, CFFG's converged fits land in a
   suboptimal segmentation basin (regime-collapse-adjacent), which is the
   main CFFG-side accuracy loss, and the native tier's initialization
   loses source 3 entirely.
4. Three of the four defects share one design lever: training length.
   More data fixes coverage reliability (measured: 300 of 300 at 5,120
   steps), restores outer contraction (the classic 10,000-step family
   converged everywhere), and moves the identifiability limit below the
   caps. The convergence-criterion defect is the exception: it is a code
   defect that more data cannot fix, because the offending posterior is
   estimated from one time step at any scale.
5. Nothing here requires the recurrence probability model to change. The
   engine work is confined to the outer convergence test and step
   control; the rest is experiment design, scale, and thresholds.

## Recommended repair and validation

1. Scale (experiment design, cff-q1au and cff-0oqo): raise the
   heterogeneous training length until both gates clear with margin.
   From the development evidence, 5,120 training steps already meets the
   coverage reliability requirement (300 of 300 development seeds pass;
   one-sided 95% lower bound 0.990), while 2,560 does not (99.0%; lower
   bound 0.974). The 4-times data increase also moves the rare-source
   identifiability limit from about 0.12 to roughly 0.06 (square-root
   scaling), turning the 0.150 caps from boundary cases into
   2-times-margin targets; confirm this with a good-mode posterior-draw
   study at the chosen scale before freezing. The generator, its legacy
   stream, and the classic family remain byte-identical; only the
   registered length changes.
2. Convergence criterion (engine, cff-qulu): fix the residual before
   adding any globalization, because a mis-formed merit function would
   misdirect a step-control rule. Two concrete changes: weight or
   normalize each family by its statistical resolution (effective
   observation count) instead of taking a raw scale-normalized maximum,
   and either exclude single-observation posteriors such as the
   initial-regime Dirichlet from the convergence test or give them a
   resolution-matched tolerance. Keep them in the reported diagnostics.
   Verify on the development seeds above: seed 4203 must certify at
   iteration 10, not 12.
3. Globalization (engine, cff-qulu): add a step-controlled outer update
   with an explicit acceptance rule — for example, relax the replacement
   (convex combination of old and new natural parameters) with a
   backtracking condition on a monotone quantity, or accept the full step
   only when the corrected residual decreases. Keep converged,
   budget-exhausted, and diverged as distinct statuses and keep
   non-finite always diverged. First settle whether the Bethe ledger is
   comparable across parameter replacements; if it is, use it as the
   merit function, otherwise use the corrected residual with a
   non-monotone (reference-value) safeguard. Give the SLDS controls the
   identical rule; the control converging only at iteration 22 is the
   regression this rule must fix.
4. Basin quality (learning path, cff-0oqo): the biggest CFFG-side
   accuracy lever is segmentation basin selection, not the recurrence
   statistics. Candidate repairs, to test on development seeds: a
   collapse detector at fit time (zero-occupancy regime with saturating
   gates) followed by a registered reinitialization or restart rule, and
   the moderate shared recurrence prior recommended in the adapter
   review, which removes the saturated absorbing geometry that creates
   the bad basin. The same prior change serves both arms and preserves
   the same-model comparison.
5. Metric design (cff-0oqo, secondary): keep the grid metric, but report
   the on-support metric beside it, and set caps from calibration
   evidence at the chosen scale rather than from a domain boundary clamp.
   If the domain requires 0.15, the scale must be chosen to make 0.15
   attainable; the current freeze did it the other way round.
6. Freeze hygiene (all three): a calibration phase that fails its own
   final-phase gates (coverage incidence, outer exhaustion, recovery
   caps) must block the freeze, not feed capped projections into it.
7. Validation before a fresh freeze: on at least 3 fresh development
   seeds at the chosen scale, both tiers converge within budget under the
   globalized loop, meet both recovery caps, and coverage passes; the
   classic family regression suite (golden generator, on-support
   ceilings, periodicity floors) stays green; float32 and 10-times-scale
   batches stay finite and loud on failure. Then freeze fresh final
   seeds. Seeds 2101 to 2103 stay untouched as immutable evidence.

## Risks and disconfirming evidence

- The instrumented runs show the marginal case: the control converges at
  22 iterations after a residual swell, so on seed 4201 the failure is
  "budget inside the oscillation timescale" rather than a permanent limit
  cycle. If wider development evidence shows most exhausted fits converge
  by, say, 32 iterations, a cheaper repair (raise the budget, keep the
  loop) competes with globalization; the recorded 2103 traces (residual
  bouncing with free energy still moving after 16) suggest some cases
  are worse than seed 4201, so I still recommend the step-controlled
  update as the durable fix, with the budget honesty rule as a backstop.
- If longer development fits at a larger scale still exhaust, contraction
  is not data-limited and the replacement rule itself needs the stronger
  (trust-region) form.
- The identifiability-limit argument rests on the reference model being
  the true model and on my per-draw alignment approximation. If a
  better-aligned evaluation shows good-mode draws passing the contrast
  cap comfortably, the "caps at the limit" claim falls and more blame
  shifts to the learner. The draw-level spread (standard deviation 0.02
  to 0.03 with means near the caps) makes that reversal unlikely.
- The basin-membership claim for CFFG (accuracy 0.86 near the collapse
  mode's 0.877) is correlational. A direct check is cheap on a
  development seed: compare CFFG's fitted regime occupancy and gate norms
  against the two reference modes.
- Raising the scale raises runtime; the classic family already costs
  about 800 seconds per fit. The chosen scale needs a runtime projection
  in the freeze, not an after-the-fact cap.

## Confidence

- High: the coverage failure rate, its structural location, and that
  5,120 steps repairs it; the stall living in the outer replacement loop
  with converged inner solves; the controls failing without recurrence;
  the residual swell being real (3 development seeds plus the control);
  the single-observation Dirichlet defect in the convergence criterion;
  the caps-versus-identifiability finding at 1,280 steps; the
  calibration-already-showed-it findings.
- Medium: the relative weight of the two convergence defects — how many
  recorded exhaustions the criterion fix alone would clear; the
  basin-membership explanation of CFFG's residual accuracy gap.
- Lower: the exact globalization rule, and the projected identifiability
  limit at 5,120 steps (extrapolated by square-root scaling, not
  measured).

## Evidence log

Immutable inputs: `artifacts/rslds-confirmatory` (results.json SHA-256
`eaf4bdccb6b2d9ecf35f59a91fa1012873bba697c8fe499781b28c1f787697c4`),
`artifacts/rslds-dual-calibration` and its `execution/results.json`,
retained seed-2101 adapter chains. Dataset identity for regenerated truth
verified against the recorded observation digest.

Development studies (scratchpad, session 34ed63d0): `coverage_study.py`
(seeds 5000 to 5499 at 1,280 steps; 5000 to 5299 at 2,560 and 5,120),
`qulu_dev.py` (seeds 4201, 4202, 4203) and `qulu_slds_dev.py` (seed
4201), each at 48 outer iterations with the frozen configuration
otherwise, per-family residual attribution, and float64 enabled as in the
frozen runner; `ident_limit.py` (posterior-draw evaluation of the
registered recovery metric on seed-2101 retained chains).

Key development numbers: coverage 456/500 at 1,280 steps, 297/300 at
2,560, 300/300 at 5,120. Outer iterations to convergence: rSLDS 12, 16,
12 on seeds 4201 to 4203; same-K SLDS control 22 on seed 4201, against
the frozen budget of 16.
