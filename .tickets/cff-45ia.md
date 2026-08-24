---
id: cff-45ia
status: in_progress
deps: [cff-9w2d]
links: []
created: 2026-08-24T06:00:41Z
type: task
priority: 1
assignee: coopwilliams
parent: cff-3v7q
tags: [rslds, validation, periodicity, experiment]
---
# Run the paired D10 classic periodicity cohort

## Background in Plain English — WHY

The current classic NASCAR baseline is not robust across declared rollout streams. At recurrence prior strength 0.001, the native seed-5000 fit passed only two of four 10,000-step periodicity rollouts. A previous screen declared four prior strengths before execution: 0.001, 0.1, 0.3, and 1.0. Strength 1.0 was the only candidate whose shared and native seed-5000 fits passed all eight rollouts while preserving the registered prediction, regime-recovery, transition-recovery, and boundedness outcomes.

That screen used the older D9 engine and retained no strength-1.0 fit checkpoint. The current engine uses D10 structured hybrid inference. We therefore cannot reuse the old fit or compare a new D10 strength-1.0 fit against an old D9 baseline. This ticket runs a prospective paired D10 cohort in which recurrence prior strength is the only intended model difference.

## Outcomes / Acceptance Criteria — WHAT

1. Before data generation, one immutable registration pins current reviewed D10 CFFG semantics, Python/NumPy/JAX/x64 identity, the canonical generator identity, family scale, split, priors 0.001 and 1.0, data and fit replicates, both initialization tiers, stochastic streams, numerical budgets, metrics, thresholds, output layout, and stop rules.
2. The declared cohort is exactly three classic development data seeds 5000–5002, fit replicates 0 and 1, both initialization tiers, and both prior strengths. Every fitted coordinate is crossed with four fixed 10,000-step rollout streams. This is 24 fits and 96 rollout cells.
3. Each prior pair uses byte-identical observations and matched tier-specific initialization and fit stream coordinates. The recurrence prior strength is the only intended model input that differs inside a pair.
4. The funnel runs only seed 5000, fit replicate 0, both tiers, and both priors. Expansion is authorized only when all four fits and terminal solves converge with complete current work evidence, every strength-1.0 rollout is bounded and exceeds its tier floor, strength 1.0 has held-out transition RMSE at most 0.15, and its paired changes meet both preservation gates: held-out mean predictive log likelihood is no more than 0.1 worse and label-aligned regime accuracy differs by no more than 0.02 from strength 0.001.
5. Expansion runs every remaining declared coordinate without ranking, dropping, replacing, or rerunning a scientific miss. It stops on an infrastructure or provenance failure but preserves completed scientific misses and continues the declared cohort when the artifact remains interpretable.
6. The complete strength-1.0 cohort passes only if every one of its 48 rollout cells is bounded and exceeds the registered tier floor, every fit and terminal solve converges, every held-out transition RMSE is at most 0.15, and every paired predictive and regime-accuracy comparison meets the preservation gates.
7. The result reports fit-cell and rollout-stream variation separately for each tier and prior. It does not call one successful seed, fit replicate, or rollout stream robust.
8. The canonical generator hash and all historical cff-oyie, prior-screen, classic-scale, and replicate-0 periodicity artifacts remain byte-for-byte unchanged. Historical D9 checkpoints are evidence inputs only and never resume a D10 fit.
9. The runner is append-only and resumable at the smallest fit coordinate. Root, attempt, fit, parameter, rollout, and aggregate manifests authenticate exact identities and reject missing, moved, duplicated, or tampered evidence before resume.
10. Float32 compact scoring, exact non-finite failure, constant trajectories, unbounded trajectories, and 10,000-step rollout execution retain the numerical-robustness behavior already required by cff-3v7q.
11. No adapter chain, heterogeneous dataset, calibration seed, final seed, confirmatory conclusion, production CFFG behavior change, or post-outcome threshold change occurs in this ticket.
12. A complete pass supplies cff-3v7q with the prospective evidence needed to choose prior strength 1.0 for a later classic parity calibration and freeze. Any miss remains a development result and blocks that choice without erasing the narrower architecture implementation evidence.

## Methodological Suggestions — HOW

These suggestions are optional. The implementer may choose another sound method.

- Build the runner beside the selected-scale and classic-scale development runners and reuse their append-only ledger, manifest, work-accounting, and historical-artifact validators.
- Use the current matched-SLDS coordinate evidence helpers only where they apply to CFFG fit work. No SLDS control is required here.
- Reuse the registered periodicity scoring and four rollout streams from the prior screen.
- Use common held-out particle streams inside each prior pair so Monte Carlo variation does not obscure the preservation comparison.
- Store parameter checkpoints for every converged fit so rollout evidence can be replayed without refitting.

## Future work

- cff-3v7q owns the final periodicity decision after this cohort.
- A passed candidate still needs fresh adapter calibration under the same chosen prior before the classic parity freeze if the comparison requires a shared prior.
- The future classic calibration and confirmatory run must use newly frozen seeds. This ticket does not authorize them.
- Source-heterogeneous recovery remains owned by cff-0oqo and is not affected by a classic-only prior decision.

## Definitions

- **D10:** The current structured hybrid inference coordinate with one categorical regime chain, one Gaussian Markov latent trajectory, one shared objective, and four-family learning statistics.
- **Fit replicate:** A separately initialized fit on the same observed dataset, using a declared stochastic stream.
- **Rollout stream:** A declared random-number stream used to generate a trajectory from one completed fitted model.
- **Paired comparison:** A comparison in which the dataset, tier, stochastic coordinates, and evaluation streams match and only the declared prior strength changes.
- **Preservation gate:** A bound that prevents improved periodicity from being purchased by an unacceptable loss on prediction or regime recovery.
- **Funnel:** The smallest predeclared subset that must pass before the remaining cohort may execute.

## Notes

Architecture handoff: the experiment-layer fit, checkpoint, periodicity, held-out metric, work-accounting, and artifact-reporting boundaries apply. Core CFFG compiler, engine, D10 learning, and persistence are consumed at one pinned reviewed commit and are otherwise unchanged. Adapter, heterogeneous-generator, calibration, and final-execution surfaces are deliberately excluded and owned by downstream tickets.

Hostile-input decisions: exact zero support remains owned by D10; non-finite or improper fit evidence fails loudly; float32 rollout guards scale with dtype; constant and unbounded trajectories cannot pass; the complete runner must remain linear in trajectory length and must not retain all 96 rollouts simultaneously.

Legend: prerequisite --> dependent

```text
[x] declared four-prior seed-5000 screen
      |
      +--> [ ] cff-45ia paired current-D10 cohort (this ticket)
                  |
                  +--> [ ] cff-3v7q classic periodicity decision
                              |
                              +--> [ ] fresh classic calibration and freeze
                                          |
                                          +--> [ ] classic parity evidence
```

**2026-08-24T07:14:15Z**

The registered D10 funnel stopped on its first declared arm before producing scientific gate evidence. Execution used experiment commit 3ab57172, immutable registration commit f12a2ea, and CFFG 4b9f9e3. The only attempted coordinate was seed 5000, fit replicate 0, shared-initialization tier, prior strength 0.001. It ran from 2026-08-24T06:59:37Z to 07:10:48Z (671 seconds). The last live sample was 7,315,840 KiB RSS; exact peak RSS was not serialized because the arm record was never reached.

The fit returned far enough to enter post-fit recovery metrics. `aligned_secondary` then raised `LinAlgError: SVD did not converge in Linear Least Squares`, with LAPACK DLASCL illegal-value warnings. No fit checkpoint, phase-work record, rollout, paired score, or funnel gate was written, so this is an infrastructure/diagnostic stop rather than evidence that either prior passed or failed. Expansion is not authorized and no rerun occurred.

Experiment artifact commit e019bf4 preserves the immutable partial result. Its registration, root manifest, attempt manifest, and append-only ledger authenticate against the registered plan and execution identity. The ledger is exactly started then failed; the root aggregate contains zero completed arms. The settled-artifact validator correctly rejects it as an unsettled failed attempt rather than silently promoting or rerunning it. Results SHA-256 is 5a4417ba261700319ce0780676c47a3a10072547795f8149f9c49d126e9ecfd4; root manifest SHA-256 is 75315b5b1c904d441582f2c815c6fc6312199d3caf16ac7c64687135c6512fe0.

Resume handoff: before any new registration or fit, retain the coherent fit/status/work and terminal-belief finiteness evidence before derived metrics. The metric boundary must distinguish a non-finite terminal posterior from an ill-conditioned but finite alignment and fail with a named diagnostic. Because that repair changes registered execution code and the failed attempt has no reusable fit checkpoint, it cannot resume or relabel this artifact. cff-45ia remains in progress and cff-3v7q remains blocked from selecting prior 1.0 prospectively.
