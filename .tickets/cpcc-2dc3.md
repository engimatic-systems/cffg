---
id: cpcc-2dc3
status: closed
deps: [cpc-4ntg]
links: [cff-wt1c]
created: 2026-11-23T01:05:08Z
type: task
priority: 2
assignee: coopwilliams
parent: cff-xa6z
---
# Compare four planners on the reactor with online heat-transfer learning

## Background in Plain English — WHY
A gentler heat-transfer learner passed the stationary tracking allowance with the CFFG controller. We now need to know how the same estimate performs when each library plans actions from it. The fixed-model comparison could not answer this.

## Outcomes / Acceptance Criteria — WHAT
1. Freeze the same learner, model, information available to each controller, actuator limits and one-second decision deadline before opening fresh seeds.
2. Run CFFG, RxInfer, NumPyro and GTSAM for ten paired seeds (6601–6610). For each library run all 599 decisions in both the stationary and hidden-change conditions with the fixed, original (q=0.012), selected (q=0.003) and informed coefficient.
3. Compare selected-controller concentration tracking during the hidden drop. Primary measure is the mean paired per-seed difference in RMSE, CFFG minus peer. Report descriptive 95% whole-seed paired intervals, the absolute magnitudes, and all ten seeds. Report own stationary cost against a 1% tolerance and changed-period advantage of learning over fixed. Include original versus selected recovery, applied actions held on failure or timeout, and setup versus warmed decision latency.
4. Verify all adapters against affine ground truth and verify live parameter delivery against fresh models on development inputs. Preserve source and dependency digests, raw trajectories and a readable HTML report.

## Methodological Suggestions — HOW
Reuse the existing plant, estimator, masks, noise stream, latency boundary and native inference methods. Initialize each new episode from the same starting estimate. Shared code for the estimator isolates the choice of planner. Use the comparison machinery from cpc-4ntg and the four-library timing study. Architecture change-path index, runtime numerical or status invariant row, benchmark consumers only. Engine rules, objective, serialization, solver schedules, JIT/vmap production paths and shared dependencies are N/A.

## Future work
cff-wt1c owns broader comparisons. Independent native parameter-learning algorithms and response to other change schedules would require distinct experiments.

## Definitions
UA measures heat transfer. q is the assumed per-transition standard deviation of log UA. Root mean square error (RMSE) measures the size of concentration deviations from their target. Informed controllers receive true UA before acting. A paired seed supplies the same noise draws to every controller.


## Notes

**2026-11-23T01:34:03Z**

Completed the frozen comparison on fresh seeds 6601–6610. Study revision 1b6e2eb. Protocol SHA-256 01edc36391b6a2c4d161785be0b25cf894a5631ac3c0ebecc82ad66a49477535. Apple M2 Max, JAX CPU float64, one measured worker at a time. All 320 episodes completed, 191,680 decisions, 19 min 11 s of measured runs. CFFG tree 4935e93, RxInfer 3.6.0 / Julia 1.10.11, NumPyro 0.22.0, GTSAM 4.3.0.

Primary result for q=0.003. Mean changed-period concentration RMSE was 0.003264905 mol/L for CFFG and RxInfer, 0.003303294 for NumPyro, 0.003303295 for GTSAM. The mean paired CFFG minus RxInfer difference was 2.48e-16 mol/L (95% descriptive seed-bootstrap interval −3.71e-15 to +3.81e-15). The mean CFFG difference against each of the other two was −3.839e-5 mol/L (−4.632e-5 to −3.063e-5), or 1.162% lower. CFFG had lower error on all ten seeds against each. Learning reduced changed-period RMSE versus each planner’s own frozen model by a median 57.55–58.01%. Stationary penalties for CFFG / RxInfer / NumPyro / GTSAM were +0.398%, +0.398%, +0.347%, +0.069%, all within 1%. Original learner penalties were +2.886%, +2.886%, +2.475%, +2.190%. The model and estimator were shared across planners, with independent closed-loop histories.

Warm selected decision median / p95 in ms: CFFG 7.62 / 7.83, RxInfer 0.52 / 0.55, NumPyro 5.80 / 7.47, GTSAM 0.74 / 0.94. Zero deadline misses and zero selected or original learner failures. Eight GTSAM native LM damping-limit holds at tick 2 of the fixed and informed cases on two seeds; other controllers recorded none. Kept all seeds. These holds affected GTSAM’s stationary reference error. A supplementary comparison restricting both it and NumPyro to the other eight seeds gives +0.33244% for each.

Validation included affine dense oracles, changed-UA vs fresh models, A/B/A restoration, scheduled plant and causal noise/reward replay, x64 forecast/update replay, native status witnesses, matched-prefix checks and all seed/trace hashes. The initial run stopped before the first plant episode because an output directory was absent. The pre-episode logs are retained in ~/agent-share/pcgym-continual-comparison-setup-011007; the corrected run qualified all four with a new freeze. Offline exact-action verification exposed NumPyro roundoff on 16/60 matched prefixes, first-action differences near machine precision and maximum normalized command difference 1.63e-10. The audit reports bounded and bit-identical matches separately. Endpoints, design, recorded data and ranking are unchanged. Independent recomputation from every raw trace confirmed all four stationary penalties and changed-period gains, paired primary differences, all three bootstrap intervals and reliability counts.

Reproduce on this machine: uv run --frozen --no-sync python tools/pcgym_timing/continual_comparison.py run --output NEW_OUTPUT, then continual_comparison_analysis.py and continual_comparison_report.py --output NEW_OUTPUT from the same tools directory. Actual HTML, raw data and machine-readable figures: ~/agent-share/pcgym-continual-comparison/index.html. The optional benchmark wrappers and report are isolated from production; validated using the four live oracles, three peer-specific development checks, continual_estimator_check, module compilation, artifact validation, tk dep cycle and git diff --check. Full production short suite was outside this isolated tooling gate. cff-wt1c retains the broader planner comparisons.
