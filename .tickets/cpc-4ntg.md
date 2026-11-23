---
id: cpc-4ntg
status: closed
deps: [cpc-7tbv]
links: []
created: 2026-11-22T23:47:56Z
type: task
priority: 2
assignee: coopwilliams
parent: cff-xa6z
tags: [pcgym, continual-learning]
---
# Reduce stationary tracking cost while retaining recovery after a hidden reactor change

## Background in Plain English — WHY
In cpc-7tbv, a scalar learner fed updated heat transfer into the certified CFFG controller. During a hidden 3.7% drop, learning reduced concentration tracking error on all five development seeds. Median improvement was 57.40%. On a stationary plant whose starting estimate was already correct, leaving this update active raised mean tracking error by 2.80%. The prespecified budget was 2%. This was the only missed criterion.

## Outcomes / Acceptance Criteria — WHAT
1. Select and freeze a candidate update rule on development data with a stated assumption about how heat transfer changes.
2. On fresh paired validation seeds, measure stationary tracking cost and post-change tracking and recovery with the same estimator, controls, deadline, reference planner and causal observation boundary.
3. Include the original learner, frozen and informed controls. Report whether the original scaling criteria are met and whether recovery slowed relative to the retained learner.
4. Record a go or hold decision and preserve raw traces, scoring windows and reproducible results.

## Methodological Suggestions — HOW
A lower fixed rate of parameter motion is a simple comparison. A model with a stable regime and an occasional change is a possible alternative when persistent innovations justify adapting faster. Compare observable predictions and recovery after both the drop and return. Reuse the measured pilot and its audited noise/command ordering. These suggestions permit another justified estimator. This is initially an isolated experiment consumer.

## Future work
cff-wt1c owns broader comparisons. A passing development screen would support a held-out study. Generic particle learning and API design retain their separate scopes.

## Definitions
An innovation is the difference between an observation and its prediction made before that observation. Stationary tracking cost is the change in concentration error from updating a parameter when the true plant stays fixed. A paired seed supplies matched noise to each controller.

## Notes

**2026-11-23T00:37:38Z**

Pre-validation design on the paired pilot branch. Cooper requested a 1% stationary tracking-error allowance and validation on fresh seeds including the original learner. Primary limit is 1% extra full-episode mean concentration RMSE on the stationary plant versus the correctly calibrated frozen controller. Report 2% and 3% as sensitivity limits on identical outcomes. The solver, one-second warmed deadline, initial parameter uncertainty, noise, change schedule, simulator and actions stay matched to cpc-7tbv. Assume the log heat-transfer coefficient normally moves more slowly than in the original learner; only its per-step random-walk standard deviation changes. On tape 6400 and paired development seeds 6401–6405 evaluate q=.010, .006, .003 and .0015, compared with original q=.012. Each candidate is run closed loop, using the archived pilot as its paired baseline. No validation output is generated until all four have been evaluated and one selected.

Keep prior pilot criteria at their original definitions (wins at least 80% of seeds). Add paired candidate minus original median recovery delay at most +5 ticks after the drop and after the restoration, with every candidate recovery reached, and retain at least 90% of the original median paired changed-period tracking advantage over frozen. These are engineering margins set before this run. A 5-tick window is the existing recovery confirmation length. Rank by fewest failed primary criteria, then smallest positive stationary penalty excess over 1%, then largest q, favoring the smallest adjustment among passing alternatives. Even if none pass development, freeze the highest-ranked candidate as diagnostic and disclose it. Freeze all rule and evaluation inputs and the development results before opening fresh tape 6500 and validation seeds 6501–6510. All four comparators run fresh with matched noise and their own trajectories. Root of archived raw evidence will be ~/agent-share/pcgym-one-percent. Isolated external experiment consumers; production solver, JIT/vmap dispatch, serialized model and shared dependencies N/A.

**2026-11-23T00:54:05Z**

All eight criteria passed on ten fresh paired control seeds. Of four development settings, only q=0.003 passed all eight on the five existing seeds. Its stationary penalty was 0.388%; q=0.006 and 0.010 exceeded 1%, and q=0.0015 added seven median paired recovery ticks after each change. The freeze at revision d563f35 (manifest SHA256 ceaa0b125cb2b88631e5d2492d0121a1556cf7beafa7ec00287573454879db95) preceded opening fresh tape 6500 and seeds 6501–6510. All four controls ran on both stationary and switching plants.

Fresh stationary concentration RMSE rose 0.103% for the candidate (descriptive paired 95% interval −0.066% to +0.264%) and 2.338% for the original (+1.865% to +2.777%), versus the correctly specified fixed model. Median changed-period reductions were 58.114% and 57.304%, respectively; informed reached 59.011%. Both learners beat frozen on all ten seeds. The candidate retained 101.41% of the original gain and took one extra tick in the median paired comparison at each switch (limit +5); both recovered every time. Settled parameter MAE fell from 1,850 fixed to 292 original to 95.4 selected. Shared switching tape temperature RMSE was 0.754 K fixed, 0.427 K original, 0.380 K selected and 0.343 K informed. 47,920 held-out control decisions, zero failures, holds or deadline misses. Verdict stays GO at 1%, 2%, and 3%. This qualifies the benchmark estimator feeding the CFFG planner for this plant, change size and noise. Use q=0.003 as the frozen CFFG setting for the next matched comparison.

Observed wall times were 292.5 s for the 40 development episodes and tape replays, and 524.1 s for the 80 validation episodes and fresh tape. Reproduce by running one_percent_run.py development, freeze and validation with --root NEW_DIRECTORY and the original pilot archive. Then run one_percent_analysis.py and one_percent_report.py with --output NEW_DIRECTORY. All tools are under tools/pcgym_timing. Actual report, protocol, hashes, traces and per-seed results are in ~/agent-share/pcgym-one-percent/index.html and its linked files.

Focused checks for the isolated experiment harness and offline report passed. Ran uv run --frozen --no-sync python -m pcgym_timing.continual_estimator_check with PYTHONPATH=tools, CPU/x64; py_compile/compileall, git diff --check and tk dep cycle. Checked all trace digests and native rewards independently, reproduced arm-level tracking and paired recovery from raw traces, bootstrap penalty intervals and source hashes at the freeze. All 30 worker environments matched the original pilot in module versions, hardware and common file hashes. All 11 HTML navigation, image and evidence links resolved. Production src/cffg, shared dependencies and repository test selection are unchanged. The full short suite was not run for the isolated experiment.
