---
id: cpt-j81k
status: closed
deps: [cpt-ydse]
links: [cff-wt1c]
created: 2026-11-22T21:28:35Z
type: task
priority: 1
assignee: coopwilliams
parent: cff-xa6z
tags: [pcgym, benchmark]
---
# Compare four fixed-model reactor controllers on thirty paired seeds

## Background in Plain English — WHY
A one-seed pilot verified four implementations and found two pairs of nearly identical returns. Cooper wants to know whether the control and forecast differences repeat across independently drawn measurement noise.

## Outcomes / Acceptance Criteria — WHAT
1. Complete 59-step and 599-step control for all four frozen controllers on 30 new paired seeds under the one-second warmed deadline.
2. Score twelve matched 60-step forecasts per seed for each controller, retaining every failure and pairing the future commands and random particle draws.
3. Publish seed-level differences, uncertainty, failures, tracking, latency and forecast accuracy in a local HTML report with raw evidence.

## Methodological Suggestions — HOW
Reuse pilot reference gates and frozen solvers. Primary outcome is the CFFG minus peer summed return on 599 decisions. Use seed-level paired bootstrap intervals and disclose numerical ties. Architecture path is external runtime and numerical readout consumers. Engine/JIT, vmap, data serialization and production model math are unaffected. Sparse support, dtype expansion and higher-dimensional modelling are outside this fixed float64 fixture.

## Future work
cff-wt1c owns broader model, learning and runtime comparisons.

## Definitions
A paired seed supplies matching measurement noise to independent controllers. A forecast origin is a point where observations stop and prediction begins. An interval across resampled seeds measures uncertainty due to this source of variation.


## Notes

**2026-11-22T21:33:11Z**

The scored protocol was frozen at 15457bf before measuring the first held-out return. Seeds 1001 through 1030 had no role in selecting the adapters. Each seed has eight episodes and four forecast batches, run serially in fresh processes. Short and long horizons reuse the same seed. Arm order rotates across seeds and phases. Use the one-second warmed deadline, held actions on failure, unchanged clipping and all archived outcomes. Forecast truth and particle seeds are respectively 100000 + control seed and 200000 + control seed. The fixed open-loop commands give the same physical truth across seeds. Forecast uncertainty is across measurement and particle draws.
Primary comparison is mean paired 599-decision return (CFFG minus each peer), with 20000 paired seed bootstrap draws, 95% pointwise percentile intervals, and numerical ties at absolute return differences up to 1e-8. Short return and forecasts are secondary. Manifest, qualification and per-seed raw data are in ~/agent-share/pcgym-paired-30. Launch command from the worktree is uv run --frozen --no-sync python tools/pcgym_timing/run_paired.py --output ~/agent-share/pcgym-paired-30 --peer-root ~/agent-share/pcgym-four-library-timing. All four fresh qualification workers passed. The measurement session is live.

**2026-11-22T22:25:59Z**

Completed 30 paired seeds (1001–1030), 240 episodes and 120 forecast batches on an Apple M2 Max. All 78,960 decisions succeeded under the one-second warmed deadline with zero holds. All 1,440 forecast origins scored. The 360 measured workers took 52.91 minutes, and the campaign including qualification, truth generation and orchestration took 54.97 minutes.

Primary 599-step CFFG minus RxInfer mean return was -1.38e-12, a numerical tie on every seed. The paired advantage over NumPyro was +0.624514 (95% pointwise bootstrap interval 0.614081–0.634889), positive on 30/30. Against GTSAM the rounded result and sign counts were the same. Average long-run concentration RMSE was 0.00378544 mol/L for CFFG/RxInfer and 0.00392072 for NumPyro/GTSAM. On the secondary 59-decision schedule CFFG outscored each mode-fitting peer on 16/30 seeds (+0.000873, interval -0.002502 to 0.004346). The long schedule contains additional switches and holds. Long-run warmed decision medians were CFFG 7.20 ms, RxInfer 0.445 ms, NumPyro 5.62 ms, GTSAM 0.656 ms. Median launch to a warmed controller was 3.43, 14.65, 9.46 and 2.33 seconds respectively.

Each method's launch-to-horizon forecast used 360 matched origins over a single fixed physical truth path. At 60 steps CFFG RMSE was 0.000943 mol/L and 0.01824 K, compared with 0.02224 mol/L and 2.571 K for the last observed value. All four methods covered every truth in the nominal 90% intervals for both variables at this lead. All methods shared the propagator. Strong evidence applies to sustained fixed-model control on this reactor under repeated measurement noise. Parameter drift and continual learning are separate experiments.

Source-level work included the serial resumable seeded launcher, whole-seed resampling and HTML report. Independent qualification and provenance verified 394 workers, including all 12 affine checks. The draw hashes matched at 1,440 forecasts. Seeded noise replay and cross-arm equality were within 2.23e-16. Directly reconstructed -1000 times squared concentration error from every recorded state and target, with maximum per-tick reward difference 2.78e-17. The largest CFFG/RxInfer applied coolant difference was 6.32e-10 K across both lengths and all paired seeds.

Evidence: ~/agent-share/pcgym-paired-30/index.html, summary.json, paired-seeds.json, protocol.json, provenance.json and per-seed raw output. Use paired_provenance.py --output ROOT, paired_analysis.py ROOT, then paired_report.py --output ROOT. Additional focused validation included independent synthetic bootstrap/analysis tests, forecast_check, compileall, dependency cycle and diff checks. The isolated-tooling gate applies to these optional external benchmark and reporting drivers. Runtime behavior, shared dependencies and the test suite are unaffected.
