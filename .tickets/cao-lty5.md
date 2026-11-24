---
id: cao-lty5
status: in_progress
deps: [cao-m63a]
links: []
created: 2026-11-24T05:32:40Z
type: feature
priority: 2
assignee: coopwilliams
parent: car-mps6
tags: [axiom]
---
# Construct recurrent interaction evidence from observed slot histories

## Background in Plain English — WHY

The interaction learner sees the previous scene and applied command. It learns which motion class and reward followed. Make that evidence reproducible before coupling the learner to pixels.

## Outcomes / Acceptance Criteria — WHAT

1. Build mixed Gaussian and categorical records from supplied slot state, identity labels and masks. Emit seven default or five tuned features. Freeze the missing-neighbor distance explicitly because the release draws that constant during tracing.
2. Implement centered rectangular and fixed-radius sampled-ellipse contacts, including projected motion, tracked preference and the proper static/background rules. Test neighbors with a scalar oracle, no neighbor, equal identities, dormant objects and a contact first appearing after forward motion.
3. Train with the executed action, observed switch and reward. Prediction omits both unknown outcomes. Preserve release-specific handling of an equal neighbor identity and the tuned default-motion training veto. Validate shape, empty support, dtype and finite values, both eager and JIT. No stochastic solver or learned parameters here.

## Methodological Suggestions — HOW

Reference b99f625 axiom/models/rmm.py lines 145–316, 319–428 and 521–547, axiom/infer.py lines 544–562 and defaults.py lines 446–482. Use the public MixedBatch representation. Runtime numerical or status invariant row covers host/JIT, whole scene and single-object reference. Message schedules and BFE are N/A.

## Future work

car-mps6 supplies learned identity and motion features and the resulting reward update. Pixel inference and control remain downstream.

## Definitions

A neighbor is another eligible slot close enough to influence motion. A categorical head is one discrete observation field.


## Notes

**2026-11-24T05:32:53Z**

Independent fixtures use the motion child handoff and existing MixedBatch. No live learner or campaign is required.
