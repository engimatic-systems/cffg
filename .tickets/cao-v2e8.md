---
id: cao-v2e8
status: closed
deps: [cao-z3o7]
links: []
created: 2026-11-24T15:00:44Z
type: bug
priority: 1
assignee: coopwilliams
parent: car-zieb
tags: [axiom, evaluation]
---
# Learn from Hunt transitions when the native reward exceeds the three reward bins

## Background in Plain English — WHY

Hunt can award two points or deduct two points in one step. AXIOM's interaction model has three reward bins for minus one, zero and one. The authors encode any other integer as three zero counts. The current port rejects the entire frame. This stops a valid game and loses the observations of motion and contact.

## Outcomes / Acceptance Criteria — WHAT

1. A real transition with either sign of two advances the public scene exactly once. The reward contributes zero to the assignment score and adds no count to any reward bin.
2. Every eligible component retains the released weight, prior discount and write gates for both reward and observed fields. Unrelated missing fields and future prediction records are ineligible for learning.
3. Rewards in the three bins retain their original meaning. Nonfinite or fractional rewards fail atomically. A large integer outside those bins follows the same zero-count rule in single and double precision.
4. The campaign passes through and scores the original signed reward. Only the model's categorical encoding changes.

## Methodological Suggestions — HOW

Use the Runtime numerical or status invariant and Learnable or checkpoint feature rows in the architecture index. Check the host and compiled frame learners, the interaction evidence builder, public scene observation and the campaign boundary. Graph schedules, graph serialization and checkpoint layout are N/A because this changes an observation value inside the existing representation. Compare two equally plausible components with opposing learned reward counts for an independent calculation of the expected weights and discounted counts. Use released AXIOM revision b99f625, models/rmm.py and vi/models/hybrid_mixture_model.py, as the reference. Verify raw signed scoring with the campaign owners.

## Future work

The actual campaign and its aggregation belong to cao-srin. The actor input guard belongs to cao-gdyw.

## Definitions

A reward bin counts occurrences of one reward value. An assignment weight is the inferred probability that a component explains the observation. The prior discount scales old counts above their starting values when the model writes an update. A zero-count observation adds no occurrence to a reward bin. A write gate decides whether to publish a candidate component update.


## Notes

**2026-11-24T15:04:25Z**

Reproduced the cause with one native Hunt step per sign. Two simultaneous item pickups yield +2 and two obstacle hits yield -2. AXIOM b99f625 models/rmm.py line 423 uses a three-way one-hot encoding. vi/conjugate/multinomial.py line 141 scores the all-zero vector at zero. The hybrid learner still applies the shared assignment and write gate to that row. The CFFG interaction boundary now admits finite integral rewards, represents an observed all-zero reward field explicitly to the learner and requires a realized motion switch. In two equally likely components with prior reward counts [1,1,1], previous counts [9,1,1] and [1,1,9] become [5,1,1] and [1,1,5] for either reward. The observed action and motion each gain 0.5 in both components. The public RGB scene accepted +2 and -2 on consecutive ticks and rolled back malformed rewards. The actor owner verified raw input pass-through. The report owner decoded a 10,000-row synthetic native-format tape containing signed double rewards, including at termination, and the frozen scorer preserved all values.

Focused validation used uv run --frozen --no-sync pytest -q on tests/test_axiom_interaction.py, tests/test_axiom_mixture.py and tests/test_axiom_observation.py (14 passed, one slow case deselected). Additional selected tests covered the causal learned rollout and both precisions of the discounted component update (3 passed), and the double-precision dormant-component projection (1 passed). No game evaluation or full suite was run for this ticket. The campaign owner handles the final branch gate and execution.
