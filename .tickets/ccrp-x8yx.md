---
id: ccrp-x8yx
status: open
deps: [cns-t0b9]
links: []
created: 2026-11-25T19:04:56Z
type: task
priority: 2
assignee: coopwilliams
parent: cff-cynx
---
# Test learned movement sequences on aliased navigation routes

## Background in Plain English

In the same-exits navigation control, two junctions and their immediate neighbors share labels. The appearance model and the model that split only the junction both reached zero of six goals. The observed tile map reached all six. A short sequence of observations and achieved movements may distinguish visits that look alike on a single step. RGM Algorithm 1 motivates grouping repeated sequences into events.

## Outcomes / Acceptance Criteria

1. Freeze how candidate sequences are proposed from exploration data, which observations they use, their duration and the evaluation episodes.
2. Compare a one-step place model, a history-conditioned candidate and the same-information tile-map planner. Keep exact local movement available to every arm.
3. Score each frozen candidate on identical new traversals using both next-observation and multi-step prediction. Report closed-loop route success and actions separately.
4. Include matching immediate exits, successful versus blocked moves, and unseen layouts. Report coverage of novel sequences, dictionary size, compile time and memory.
5. Use an independent small oracle and distinguish uncertainty about a place from uncertainty about the agent's movement.

## Methodological Suggestions

Start with two-step sequences. Use only information available before the decision. Hold the path length, fit data and navigation budget fixed while testing the effect of a richer place representation. Architecture change path for a bounded experiment is the public model and inference interface. A production structure edit or checkpoint migration needs a separate review of the duplicated execution surfaces.

## Future work

The general operator and serialized proposal interface belong to cff-cynx. Consider sparse storage after measuring the cost of a larger learned model.

## Definitions

A history-conditioned place assignment can depend on recent observations or applied movements. An RGM path is a latent transition pattern. A route is the agent's executed sequence of moves.


## Notes

**2026-11-25T19:05:03Z**

Prerequisite --> dependent. [closed] cff-du6m --> [pilot] cns-t0b9 --> [next] ccrp-x8yx. Parent cff-cynx owns general proposal operators. The immediate follow-up can use a controlled fixture and the already available engine.
