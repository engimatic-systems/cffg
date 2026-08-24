---
id: cff-45ia
status: in_progress
deps: []
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
