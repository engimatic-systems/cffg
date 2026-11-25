---
id: chr-vyg1
status: closed
deps: [chr-ozmk]
links: []
created: 2026-11-25T06:21:25Z
type: feature
priority: 1
assignee: coopwilliams
parent: cff-mjku
---
# Condition interior Gaussian equality taps exactly

## Background in Plain English — WHY

A continuous observation can fix a whole vector at a junction shared by several factors. The compiler previously rejected a DATA port there. Adding a sensor with arbitrary positive noise changes the physical model. A measured value must reach every factor that uses that vector, and the evidence score must include the density of the measured value.

## Outcomes / Acceptance Criteria — WHAT

1. A whole-vector observation substitutes into every ordinary Gaussian prior and affine conditional sharing the observed junction. When both ends of a conditional are fixed it contributes the exact negative log density.
2. The ledger excludes the entropy of fixed coordinates and preserves the entropy of uncertain coordinates. Every fixed copy returns the same mean and exactly zero covariance; unfixed sensor ports retain their predictive cavity.
3. Two distinct latent observations can coexist. Binding None to either restores the distribution conditional on the remaining evidence. Local statistics, both schedules, per-lane execution and persisted template reconstruction agree with independent dense conditioning.
4. Reject duplicate observations of one junction, unsupported incident families and invalid values.
5. Untapped graphs retain ordinary routing, state and bundle representation.

## Methodological Suggestions — HOW

Change paths are public model edit and runtime numerical or status invariant. Resolve static equality-connected variables during compilation, and pass the existing state observation vector and mask to affected native Gaussian groups. Verify flooding, flat/registerized scan, per-stage and batched execution, per-node statistics, full/selected marginals and entropy corrections. Include both floating precisions, weak or uninformative messages, mixed coordinate units and a 32-slice chain. Compare explicitly conditioned left and right submodels.

## Future work

This admission covers built-in ordinary priors and affine conditionals. A measured subset or a noisy observation uses an affine emission. Nonlinear or external endpoints would need their own substitution and evidence rules. The parent owns the aggregate suite and dashboard.

## Definitions

A tap exposes a variable through equality. An equality-connected variable includes all leg copies forced to agree. A cavity is the prediction at a sensor before that sensor's own evidence. BFE is the Bethe free energy; on these trees it equals the negative log evidence.

## Qualification

Independent dense and forward-filter/backward-smoother references agreed in both precisions. The bound 32-slice evidence was 5.83856028527 nats. The explicitly split left/right graph and all factor and edge-entropy terms agreed to roundoff. Timed-default representation control on the 6-slice untapped example: state 2,464 bytes, template and schedule fingerprints, state leaves before and after solving, BFE and compiled StableHLO for the sweep and ledger were identical with and without tap support atop the corrected affine rules. A separate 2-slice f64 fixture produced exact-point NLL 8.87165092619 with variance zero. Giving that observation noise variance .01 produced NLL 8.51538303741 with latent variance .00929479691.

Raw evidence under ~/agent-share/cffg-hierarchical-runtime/gaussian-tap includes 9fcc47a-a3e9c88z/qualification.json, the per-factor split ledger, final-no-tap-{after,base}.json, probe.json, and reproduction scripts. Focused execution tests cover stats and individual Colony masks, duplicate/unsupported endpoints and transactional invalid-value checks.
