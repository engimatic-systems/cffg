---
id: cao-m63a
status: closed
deps: []
links: []
created: 2026-11-24T05:25:59Z
type: feature
priority: 2
assignee: coopwilliams
parent: car-mps6
tags: [axiom]
---
# Acquire fixed affine motion experts from tracked object transitions

## Background in Plain English — WHY

The planner can use a fixed table of affine motion rules. A tracked object supplies a before/after pair from which to select a rule or initialize an unused slot. This slice of car-mps6 can be qualified on independent histories before learned neighbors or pixel slots are available.

## Outcomes / Acceptance Criteria — WHAT

1. Keep a fixed table of six-coordinate affine maps and an occupancy mask. Initialize the four released velocity-model experts in order. Score occupied predictions against each observed successor with the isotropic Gaussian likelihood, acquire a rule on a strict threshold miss, then choose the first maximum after rescoring.
2. Test displacement-based velocity bias, coefficient clipping and constant-target teleport birth. The first vacant slot receives a nonzero new expert. A fully clipped zero map leaves the slot vacant. A full table preserves every map and still supplies the best supported switch.
3. Consume slots in their recorded order with tracked/update masks. Hand the executed action, previous state and inferred switch to the interaction learner. Show the result of ordered births, matching repeated motion, disappearing slots and read-only forward application on independently specified histories.
4. Preserve shape under JIT. Cover float32/64, a 500-map table, exact ties and threshold boundaries. Reject invalid inputs and empty occupancy without partial writes, including a broken active score. Inactive experts have zero support. The maps have no learned covariance or iterative solve.

## Methodological Suggestions — HOW

Use pinned release b99f625, axiom/models/tmm.py lines 66–164, 220–245, 248–288, 380–412 and 472–514; defaults.py lines 311–320; axiom/infer.py lines 294–319, 341–356 and 405–437. Use the default six-coordinate velocity model and derive independent scalar expectations. Architecture row “Runtime numerical or status invariant”: compare eager/JIT and single-slot/ordered-frame behavior. Message schedules, free energy and learned hybrid component transactions are N/A.

## Future work

car-mps6 owns the recurrent interaction features, reward learning and reductions. Perception supplies observed slot positions and the tracking mask. Feed the resulting motion matrices to the car-ne0a planner.

## Definitions

An affine expert predicts a state by multiplying the previous state by a matrix and adding an offset. A switch is the integer index selected for the observed transition.


## Notes

**2026-11-24T05:31:22Z**

Implemented initial_motion, observed_state, motion_step and acquire_motion in cffg.experiments.axiom_motion. The JAX frame scan processes eligible slots in order, returns absolute Gaussian likelihoods and post-insertion switches, and retains the previous bank on invalid frames. At distance exactly the observation teleport cutoff, a velocity expert can be born while an older expert wins the rescoring tie. A clipped all-zero teleport stays vacant. Seven focused tests passed with uv run --frozen --no-sync pytest -q tests/test_axiom_motion.py (4.60s), including float32/64, 500 rows, eager/JIT, true boundary ties, a repeat after restoration, nonfinite inputs and full capacity. Independent analytical histories only. The parent owns the full gate, dashboard entry and shared docs. The default six-coordinate unit-frame velocity model is covered. Interaction feature and bank learning remain with car-mps6.
