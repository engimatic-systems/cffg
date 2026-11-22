---
id: cpc-4ntg
status: open
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
