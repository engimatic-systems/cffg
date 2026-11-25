---
id: chr-000i
status: in_progress
deps: []
links: [chr-kzwu]
created: 2026-11-25T06:27:52Z
type: task
priority: 2
assignee: coopwilliams
parent: cff-pyy9
---
# Check exact ties in the registered slow-regime maze trace

## Background in Plain English

The registered slow-regime maze chooses the lowest arm index on a tie. Finite-precision filtering gives two slightly different numbers at some mathematically equal arm probabilities. An exact replay can distinguish a rounding tie from a decision that disagrees with the model.

## Outcomes / Acceptance Criteria

1. A separate command reads the frozen trial trace, regenerates the paired tape, and uses integer odds to select each arm. It reports both exact and archived success and identifies every differing decision.
2. The check binds its output to the raw trace digest. It preserves the registered observations, model, thresholds and measured actions.
3. A multi-tick same-clock reference independently checks state correlations and per-tick evidence after repeated slow transitions.

## Methodological Suggestions

Use the rational probabilities in the registered configuration. Multiply the two integer weights by observation likelihood numerators and apply the slow transition on its declared clock. A common divisor can keep the integers small. Compare the joint categorical filter with a single ordinary Markov chain over (regime, fast state) pairs for the same-clock test.

## Future work

The parent cff-pyy9 owns interpretation, shared documentation and the final suite.

## Definitions

An odds pair contains unnormalized integer weights for the left and right regimes. A maximum a posteriori decision chooses the state with the larger probability after observing the cue. A tie means the two exact weights are identical.


## Notes

**2026-11-25T06:27:57Z**

Architecture index: Runtime numerical or status invariant is exercised at the public existing hierarchy boundary by a stronger independent same-clock oracle. No engine, kernel, schedule, checkpoint or IR changes. The post-hoc command is isolated analysis of a saved trace; it runs no measured policies. Parent owns the final combined default suite.
