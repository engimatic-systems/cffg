# Reference-Adapter Mixing Review

Reviewer: independent-reviewer (Claude, 20 August 2026 session)

Ready for comparison: yes

## Scope

`cff-l5ag`

## Observed evidence

All evidence below comes from the immutable cff-oyie artifact
(`artifacts/rslds-confirmatory`, results.json SHA-256
`eaf4bd…697c4`, experiment commit `6c0437a` in the cff-oyie worktree,
CFFG commit `a686faa`), from the frozen calibration artifact
(`artifacts/rslds-dual-calibration`), and from read-only analysis of the
retained chain draws. Analysis scripts and their JSON outputs are listed at
the end of this report. Development-seed experiments use fresh seeds (4201,
chain seeds 4201xx) that no freeze or final decision ever used.

### 1. Which variables fail, arm by arm

I recomputed split R-hat and effective sample size (ESS) from the retained
`diagnostic_draws` of all 10 fitted arms with the registered code
(`diagnostics.py`). The recomputed worst values match the recorded values
exactly, so the registered arithmetic is correct.

The 43 monitored variables are: log likelihood per step (variable 0), 12
sorted recurrence-weight norms, 12 sorted recurrence offsets, 8 sorted
dynamics-covariance eigenvalues, 10 sorted emission-covariance eigenvalues.

The failures split cleanly by family:

- classic (6 arms): only the 24 recurrence variables fail. Log likelihood
  mixes extremely well in every classic arm (R-hat 1.002 to 1.009, ESS
  2,000 to 2,460; chain means agree to 4 significant figures). Dynamics and
  emission eigenvalues pass everywhere.
- heterogeneous (4 arms): the recurrence variables fail and, in addition,
  log likelihood itself fails badly (R-hat 2.3 to 7.0, ESS 4 to 5), along
  with 4 to 8 dynamics-covariance eigenvalues.

Worst per arm (recomputed, equal to recorded): classic R-hat 1.67 to 2.14,
ESS 5.4 to 7.1; heterogeneous R-hat 3.2 to 9.6, ESS 4.1 to 4.4.

### 2. Label switching versus genuinely different posterior regions

I matched regime sequences across chains with the best regime permutation
(Hungarian assignment on the confusion matrix of final-draw sequences):

- classic, shared tier: all 4 chains carry the same labels (identity
  agreement at least 0.997). No label switching at all, yet the recurrence
  variables still fail. Label symmetry cannot be the cause here.
- classic, native tier: chains sit in different label orders (identity
  agreement as low as 0.000) but agree almost perfectly after the best
  permutation (0.996 to 0.999). These are pure label-switched copies of one
  posterior mode.
- heterogeneous: even after the best permutation, cross-chain agreement
  drops to 0.75 to 0.87 in the worst pairs. Chains disagree about the
  segmentation itself, not just its labels.

The heterogeneous chains occupy two qualitatively different, internally
stable modes. On seed 2101 (both tiers):

- a collapse mode (train log likelihood 26.52 per step): one regime has
  exactly zero occupancy, its dynamics matrix drifts unstable (largest
  eigenvalue modulus 1.4 to 3.8), and the gates saturate (weight norms up
  to 126)
- a full mode (27.00 per step, about 615 nats better in total): all 4
  regimes alive (smallest at 10.5% occupancy), soft gates (weight norms
  0.3 to 1.6), stable dynamics

On 2101 shared, chains split 2 and 2 between these modes even though all 4
chains started from the identical mapped initialization. On 2103 native,
two chains sat in a third and fourth region and were still drifting
downward at the end of sampling. Segment-mean log-likelihood traces show
flat plateaus per chain (except the still-drifting ones), so these are
genuine distinct modes, not slow warm-up towards one mode.

The collapse mode is an absorbing trap for Gibbs: once a regime is empty
and the saturated gates route all probability away from it, no later sweep
can repopulate it.

### 3. Are the registered diagnostics valid for the sampled representation

The arithmetic is correct (recomputed values match). The monitored
variables are not valid summaries of what the posterior identifies:

- sorting the flattened recurrence norms and offsets does not make them
  label-invariant. The decision-list (stick-breaking) parameterization is
  order-dependent: relabeling regimes changes the gate parameter values,
  not just their order. The classic native tier shows this directly: pure
  label permutations of one mode still fail all 24 recurrence monitors.
- more fundamentally, the classic posterior does not identify the
  recurrence parameters at all. The classic generator's true gates are
  saturated (weights are 100 times unit vectors; offsets about plus or
  minus 200). Any weight vector beyond saturation fits the data equally
  well, so the posterior is a wide plateau whose only constraint is the
  vague prior (variance 1,000, from `vague_rslds_fit_priors` with
  `recurrence_strength=1e-3`). Retained draws confirm this: weight norms
  wander between about 20 and 100 within a chain (state magnitudes are
  about 5), offsets reach 130, and the within-chain autocorrelation time is
  roughly 350 to 1,000 iterations.
- monitoring identified functionals does not rescue the gate either. I
  recomputed mixing on label-canonicalized transition probabilities
  evaluated at the sampler's own states (both all-source and
  realized-source-row versions). Classic arms still fail (worst R-hat 1.5
  to 1.7, ESS 6 to 9), because near-boundary and counterfactual-source
  probabilities inherit the plateau: the decision boundary position wanders
  freely inside the data margin without any likelihood cost.

So for the classic family the gate is unsatisfiable at any feasible chain
length under the frozen model: it demands ESS 50 and R-hat 1.05 on
posterior directions the likelihood does not constrain. For the
heterogeneous family the same monitors additionally pick up a real failure
(distinct modes) that valid diagnostics should and would still flag.

### 4. Which update family causes poor movement

The recurrence update: Polya-Gamma-augmented Bayesian logistic regression
per source and gate (`paper_adapter.py`). In the saturated regime this
update is a known slow random walk: with logits of order 100, the
Polya-Gamma auxiliaries concentrate near zero, the conditional Gaussian for
the weights becomes wide and prior-dominated, and successive draws diffuse
along the plateau. This interacts with the label update in a lock-in loop:
saturated gates freeze the segmentation (within-chain label flip fraction
is about 0.002 in classic), and the frozen segmentation keeps the logistic
problem separable. All other updates move well: log likelihood, latent
states, segmentation, dynamics and emission all mix cleanly in the classic
arms.

In the heterogeneous family the same mechanism creates the collapse trap
described above, and the slow-moving direction becomes a mode barrier the
sampler cannot cross in 2,000 iterations.

### 5. The freeze already contained this evidence

This failure was visible before the final run:

- every one of the 16 calibration adapter arms failed mixing, even under
  the looser calibration gates (R-hat at most 1.2, ESS at least 10):
  observed worst R-hat 9.21, worst ESS 4.12
  (`artifacts/rslds-dual-calibration/execution/results.json`)
- the budget rule projected 2,430 draws from the calibration ESS but the
  2,000-iteration cap truncated the budget to 400 warm-up plus 1,600 draws
- that projection was invalid anyway: ESS scales with draws only if chains
  explore one mode; with between-chain R-hat of 9 the deficit is modal, and
  more draws do not close it

The final failure therefore confirmed calibration behavior rather than
revealing anything new. The freeze process let a failing gate through
because calibration-phase eligibility does not require the mixing gate to
pass (`confirmatory.py`, `reference_eligible` when `phase == "calibration"`).

### 6. Development-seed pilot: moderate prior restores identification

Design: heterogeneous development dataset (fresh seed 4201, 1,280 training
steps), native-style adapter chains, frozen chain budget (4 chains, 400
warm-up, 1,600 draws), two arms that differ only in the shared registered
prior scale: frozen `recurrence_strength=1e-3` (prior variance 1,000)
against a moderate `recurrence_strength=1.0` (prior variance 1).

Results (all 8 chains completed):

- frozen prior: gate fails (worst R-hat 3.71, minimum ESS 4.3). One chain
  found the full 4-regime mode (log likelihood 26.97 per step, soft
  gates), one collapsed a regime with saturated gates (maximum norm 122),
  and 2 sat in partially saturated intermediate modes (norms 33 to 47)
- moderate prior: saturation disappears everywhere (maximum gate norm 3.7
  against 122) and mode discovery improves to 3 of 4 chains in the full
  mode, with the 3 good chains being exact label permutations of one
  another. One chain still entered the collapse mode — with soft,
  non-saturated gates, so the trap is segmentation-level, not only
  prior-level. The pooled gate still fails (worst R-hat 3.37)
- within the good mode (3 chains, moderate prior): the label-invariant
  registered monitors (log likelihood plus 18 covariance eigenvalues)
  come to worst R-hat 1.065 and minimum ESS 31 against thresholds 1.05
  and 50, on the frozen 1,600-draw budget that the calibration projection
  already said needed about 2,430 draws. Canonicalized realized-state
  transition functionals still mix slowly (worst R-hat 2.0), because the
  rare sources' gate posteriors at 1,280 steps are wide and the sampler
  crosses them slowly

Reading: the pilot confirms the plateau mechanism (the prior change
removes saturation and improves mode discovery) and simultaneously shows
that no single lever passes the heterogeneous gate at the frozen budget.
The prior, label canonicalization, collapse handling, and an honest
budget (or a larger data scale, see the heterogeneous review) are each
necessary. For the classic family the plateau is the entire failure, so
the prior plus canonicalization plus the existing budget is the predicted
fix; that prediction still needs a classic-scale development run, which I
did not execute in this session (about 55 minutes per arm).

## Competing explanations

1. Label or decision-list symmetry across chains inflates pooled
   diagnostics. True for the classic native tier only. Falsified as the
   full explanation by the classic shared tier (identical labels, still
   fails) and by the heterogeneous log-likelihood failures (label moves
   cannot change the likelihood).
2. The monitored recurrence variables are unidentified under the frozen
   vague prior on near-deterministic data, so the sampler wanders a
   plateau and the gate can never pass. Supported by the truth parameters
   (saturated by construction), the prior scale, the norm traces, the
   autocorrelation times, and by the failure of even canonicalized
   functional monitors. This is the main classic-family explanation.
3. The heterogeneous posterior is genuinely multimodal (regime-collapse
   trap against full-occupancy mode), and 400 warm-up iterations cannot
   cross between modes. Supported by stable per-chain log-likelihood
   plateaus 615 nats apart, zero-occupancy regimes with unstable dynamics
   in the collapse mode, and identical-initialization chains splitting
   between modes. This failure is real and valid diagnostics would still
   report it.
4. Diagnostic arithmetic or budget mistakes. Falsified for the arithmetic
   (exact recomputation matches). Partially true for the budget: the frozen
   budget contradicted its own projection rule (2,430 draws needed, 1,600
   given) and the projection premise was invalid under modal failure.

## Discriminating checks

Checks I ran (all read-only on the artifact, or on fresh development
seeds):

1. Per-variable R-hat and ESS by variable family across all 10 arms
   (separates explanation families by which monitors fail).
2. Chain-mean-centered R-hat and ESS (separates between-chain offsets from
   within-chain drift; classic arms still fail after centering, so drift is
   real).
3. Cross-chain regime-sequence agreement under identity and best
   permutation (separates label switching from segmentation disagreement).
4. Per-chain log-likelihood plateau analysis in fifths (separates distinct
   modes from unfinished warm-up).
5. Mode characterization: occupancy, gate norms, dynamics eigenvalues per
   chain (identifies the collapse trap).
6. Canonicalized functional-monitor recomputation (tests whether
   canonicalization alone could ever satisfy the gate; it cannot for
   classic).
7. Development-seed prior pilot (see section 6; the smallest experiment
   that separates explanation 2 from a sampler defect: if the plateau is
   the cause, a moderate registered prior restores mixing without touching
   the sampler code).

## Findings

1. The mixing gate failed for two different reasons, one per family.
   Classic: the frozen model monitors unidentified parameter directions;
   the sampler and the data are fine, and every identified quantity mixes
   well. Heterogeneous: the posterior is genuinely multimodal at 1,280
   steps, with an absorbing regime-collapse mode, and the frozen chain
   protocol cannot bridge or even detect-and-classify the modes.
2. The paper-equation adapter implements its model correctly as far as
   this evidence reaches: the density oracles pass, and chains agree
   perfectly on identified functionals in the classic family. There is no
   evidence of a sampler bug.
3. The registered 43-variable monitor set conflates three different
   things: label order (native tier), unidentified plateau directions
   (both families), and real modal disagreement (heterogeneous). Only the
   third is a legitimate reason to withhold a comparison.
4. The freeze rules converted known-failing calibration evidence into an
   unattainable registered gate: calibration observed the same failure,
   the budget cap contradicted the projection rule, and calibration-phase
   eligibility ignored the mixing gate.

## Recommended repair and validation

1. Change the shared registered prior scale, not the sampler. Set a
   moderate recurrence prior (development calibration should pick
   `recurrence_strength` in about the 0.1 to 1.0 range) in
   `vague_rslds_fit_priors`, which both CFFG and the adapter consume. This
   makes the transition family identified, removes the plateau, removes
   the saturation lock-in, and keeps the same-model property of the
   comparison because both arms share the prior. The classic truth has
   saturated gates, so development seeds must confirm that the softer
   fitted gates still meet the classic task-adequacy outcomes (rollout
   periodicity); logits of order 10 to 25 at state scale 5 should remain
   effectively deterministic.
2. Canonicalize the native-tier label order at initialization. Order the
   initial clusters by a deterministic observation-only rule (for example
   first-visit time) before building the model, so all chains start in one
   label order. The classic shared tier shows chains then stay in one
   order. This is an initialization choice, not truth leakage, and it
   removes the pure-permutation component of the failure without hiding
   real modes.
3. Keep R-hat and ESS thresholds unchanged, but monitor identified,
   canonicalized quantities: log likelihood per step, dynamics and
   emission eigenvalues, and transition probabilities at realized states
   under the moderate prior (which the pilot must confirm are then
   identified). Do not pool draws across chains that valid diagnostics
   place in different modes; report modal disagreement as its own failure.
4. Fix the freeze rules: calibration must pass the final-phase mixing gate
   (scaled to its budget) before a freeze; a budget cap below the
   projection rule's requirement is a freeze failure, not a truncation;
   and an ESS projection is only valid when R-hat is already near 1.
5. For the heterogeneous family, add a registered mode-handling rule:
   either longer warm-up with over-dispersed starts and a collapse
   detector (zero-occupancy regime with saturated gates is mechanically
   detectable), or an explicit protocol decision that collapse-mode chains
   are a reported failure of the arm. Do not silently average across
   modes.
6. Give the adapter an honest chain budget, and expect the heterogeneous
   scale increase recommended in my heterogeneous review to do part of
   this work. That review finds 1,280 steps too short on independent
   grounds (coverage reliability 91%, recovery caps at the
   identifiability limit) and recommends 5,120 steps. A 4-times longer
   path sharpens the segmentation posterior, which is what makes the
   collapse mode competitive in the first place, so the two repairs
   reinforce each other. Re-derive the draw budget from the projection
   rule at the chosen scale, and remove the 2,000-iteration cap or treat
   a binding cap as a freeze failure.
7. Validation before any fresh freeze: on at least 3 fresh development
   seeds per family, all 4 chains meet the registered thresholds on the
   revised monitors with margin (worst R-hat at most 1.02, ESS at least
   200 under the calibration budget), zero collapse-mode chains in the
   heterogeneous family, oracles unchanged and passing, and classic
   task-adequacy outcomes unchanged within their frozen tolerances. Only
   then freeze fresh final seeds (criterion 6 of cff-l5ag).

## Risks and disconfirming evidence

- The heterogeneous pilot already shows the moderate prior alone does not
  pass the pooled gate at the frozen budget: the collapse trap and the
  budget shortfall remain. If the combined repair (prior plus
  canonicalized initialization plus collapse handling plus the projected
  budget) still fails on development seeds, a sampler-level repair (for
  example marginal or blocked updates for the recurrence family, or
  tempered chains) becomes the leading candidate.
- The classic-family prediction (prior plus canonicalization suffices at
  the existing budget) is untested at the 10,000-step scale in this
  session. A classic development run is the first validation step; if the
  24 recurrence monitors still fail there under the moderate prior with
  identical labels, the plateau explanation is incomplete.
- A moderate prior changes the probability model. If classic rollout
  periodicity degrades under the softer fitted gates, the prior scale must
  rise until task adequacy returns, and there may be no scale that both
  mixes and stays adequate; that would falsify repair 1 and push towards
  monitored-variable reform (repair 3) plus a sampler upgrade.
- The heterogeneous mode gap (615 nats) could in principle be a
  likelihood-computation defect rather than true multimodality. The
  internal consistency of each mode (occupancy, gates, dynamics all
  coherent) argues against this, but an exact-oracle likelihood check on
  one draw from each mode would settle it.
- My mode analysis subsampled draws (every 40th); very fast within-chain
  mode switches would be invisible. The flip-fraction and ESS numbers make
  this unlikely.

## Confidence

- High: the failure decomposition (classic identified-functional agreement
  against heterogeneous modal disagreement), the plateau mechanism, the
  collapse trap, the calibration-already-failed finding, and that the
  diagnostics arithmetic is correct.
- Medium: that a moderate shared prior alone restores mixing at the frozen
  budgets (pilot evidence at development scale; classic full scale not yet
  rerun).
- Lower: the exact prior scale to register, which needs the development
  calibration study described above.

## Evidence log

Immutable inputs: `artifacts/rslds-confirmatory` (results.json SHA-256
`eaf4bdccb6b2d9ecf35f59a91fa1012873bba697c8fe499781b28c1f787697c4`),
`artifacts/rslds-dual-calibration`, retained chain NPZ files under
`<family>/<seed>/replicate-0/attempt-*/<tier>/paper_adapter/`.

Analysis scripts (scratchpad, session 34ed63d0): `adapter_diag.py`
(per-variable R-hat/ESS, chain-centered variants), `label_analysis.py`
(cross-chain permutation matching), `functional_diag.py` and
`realized_diag.py` (canonicalized functional monitors), plus inline
per-mode characterization. Pilot: `pilot_het.py` with dev seed 4201 and
chain seeds 420110 to 420113, run through the pinned legacy runtime
(`uv run --project reference/legacy-runtime cffg-rslds-comparison-chain
--warmup 400 --draws 1600 --thin 1`).
