---
id: cml-xdw1
status: in_progress
deps: []
links: []
created: 2026-11-28T04:28:22Z
type: feature
priority: 2
assignee: coopwilliams
parent: cff-cj5h
---
# Specify and qualify seeded selection among exactly tied environment actions

## Background in Plain English — WHY

The current RollingAgent MAP choice uses numpy.argmax, which always picks the first index of a tie. Criterion 73 of cff-cj5h calls for seeded uniform choice among exactly tied maxima when select="map". Changing how ties consume the action RNG can change reproduced trajectories.

## Outcomes / Acceptance Criteria — WHAT

1. Document the requested select="map" tie semantics and the effect on old trajectories. Unique MAP winners retain their action, and deterministic graph assignment stays separate.
2. Cover cff-cj5h criteria 59–80, 90 and 93 under the approved policy. Preserve the posterior-sampling contract, exact (one ULP is different) maxima, trusted rejection of invalid probability vectors, unbiased seeded choice, and current/legacy checkpoint continuation.
3. Apply the agreed semantics to environment action selectors in the listed harnesses. Preserve deterministic decoding, DELTA projection, CSP decimation and numerical optimizer choices.
4. Check 4,096 draws in a four-way tie, 128 equal maxima in float32, malformed probabilities, reproducible seeds and checkpoint restores.

## Methodological Suggestions — HOW

Use the Learnable or checkpoint feature and Runtime numerical or status invariant change paths for RNG ownership and current/legacy state. Audit agent.py and each action-executing simulator. A shared host action helper can concentrate the exact comparison and RNG lifecycle.

## Future work

No near-tie threshold, discrete solver change or stochastic inference schedule is assigned.

## Definitions

MAP selects an action with the highest posterior probability. One ULP is one representable floating-point step. An RNG is a random number generator. CSP is constraint satisfaction.


## Notes

**2026-11-28T04:36:10Z**

Plan. The action-RNG owner selects among literal maxima only, drawing no random value for a unique winner. RollingAgent and the three named experiment executors share a host NumPy helper. DecisionPlan.action remains the deterministic representative for display. For actual cue episodes, cache conditional inference by the executed first action and reading. Current checkpoints already carry rng_state; legacy restoration leaves the target seed state in place. Preserve both. Validate uniform four-way draws, 128 float32 ties, a one-ULP near tie, malformed/absent mass, exact cold continuation, and representative versus executed cue histories. Numeric solver, Colony, decoding, CSP and DELTA schedules are N/A.

**2026-11-28T05:37:19Z**

Implemented uniform selection over the exact maximum set. A unique maximum leaves the RNG cursor untouched. RollingAgent uses its checkpointed generator; each experiment uses a seeded child generator so that tie draws cannot consume world randomness. The cue harness caches each reachable first-action/observation pair and uses the conditional for the action actually taken. Its displayed action remains a deterministic representative. The state format is unchanged. The final combined tree includes cml-6ba0 and removes the completed ergonomics todo note. Locked focused verification: 12 checks in tests/test_action_selection.py, test_model_lint.py and test_readme.py; four real simulations from test_tmaze_learning.py, test_behavioral_epistemics.py and test_hybrid_epistemics.py all passed in 23.16 s, including the original T-maze learning/ablation assertions. Full short gate and PR still pending.
