---
id: chr-000i
status: closed
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

**2026-11-25T06:34:00Z**

Checked 6,144 immutable native decisions from 16 registered seeds and regenerated the cue tape. Exact integer replay finds ten ties and seven changed actions, all within period zero. Native reward count was 5,311 and the exact lowest-index count was 5,308. Independent flat and history-using flat counts remained 4,423 and 5,207. The check rejected a tampered recorded action. The parent ticket holds the seven coordinates and both rates. tests/test_hierarchy.py::test_repeated_one_clock_matches_flattened_joint_hmm independently reduces the hierarchy to one dense ordinary Markov process on the joint state for eight consecutive observations; parent and child clocks, joint beliefs, conditional evidence and rollover prediction agree in float32 and float64. Both final hierarchy files passed, 13 tests in 13.31 seconds, via uv run --frozen --no-sync pytest -q tests/test_hierarchy.py tests/test_hierarchy_checkpoint.py. This focused correctness run overlapped other correctness workers. The parent owns the combined short gate.
