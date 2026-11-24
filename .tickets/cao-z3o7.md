---
id: cao-z3o7
status: closed
deps: [cao-0cla, car-9ccf, cao-lty5, cao-24tn, cao-0pbk]
links: []
created: 2026-11-24T05:32:40Z
type: feature
priority: 2
assignee: coopwilliams
parent: car-mps6
tags: [axiom]
---
# Apply released hybrid-mixture history and write rules to one or more observations

## Background in Plain English — WHY

The released iMM and rMM discount old evidence by the current responsibility. The library also supports ordinary additive learning. Expose the specific released rule to both integrations without changing additive updates.

## Outcomes / Acceptance Criteria — WHAT

1. Score occupied components using Gaussian and observed categorical evidence. Use likelihood-only soft assignment on a threshold match, first dormant one-hot on novelty, and row zero on a full novel bank. A full-bank write keeps the live logical identity.
2. Form each proposal from the reset prior, new weighted moments and responsibility-weighted old excess natural statistics when the winning old count exceeds one. Write whole rows only for responsibility above .25 and proposed alpha below 32. Check Dirichlet heads and all Gaussian moments with independent scalar calculations, including the 2.1 to 1.6 example.
3. Expose a host transaction and fixed-shape ordered-device scan with shared semantics, empty-head identity learning, immutable inputs and exact restoration. Test reset priors, invalid records, disabled updates, empty active bank, both dtypes and slot-capacity failures. The general additive update remains available.
4. Publish the scene training/prediction boundary and link the feature builder; test repeated observations with updated priors. Score predictions with unknown reward/switch omitted. Record further prerequisites for live integration.

## Methodological Suggestions — HOW

Use b99f625 axiom/models/utils_hybrid.py lines 127–155 and axiom/vi/models/hybrid_mixture_model.py lines 109–170,229–238. Coordinate logical slot/generation writes with the component lifecycle owner. Architecture change paths Learnable or checkpoint feature and Runtime numerical or status invariant: compare checkpoint, host/JIT and ordered updates. Flooding and scheduled inference are N/A.

## Future work

car-mps6 owns model-generated pseudo-observations, merge proposals and score acceptance. The perception parent supplies the real object stream.

## Definitions

Excess statistics are learned posterior natural parameters minus their reset-prior values. A write gate is the test deciding which proposed component rows replace existing rows.


## Notes

**2026-11-24T05:32:53Z**

The feature child supplies scene integration, the mixed learner supplies absolute scoring, and the lifecycle child supplies immutable host identity transactions. Device acquisition preserves the same layout.

**2026-11-24T05:49:04Z**

Implemented immutable host learn_observation and compiled ordered learn_frame. Full-bank novelty selects row zero and preserves its identity. Confirmed both dtypes give two 1.6 counts from 2.1 under equal responsibilities; ordinary additive learning gives 2.6. Strict quarter and 32 cutoffs, winner history, all heads, empty support, restart, rollback and the independent 32-slot statistics oracle passed. Real rMM updates consume observed reward and switch; contextual queries withhold them. All 10 focused tests passed after applying release_view during scoring (25.50s): uv run --frozen --no-sync pytest -q tests/test_axiom_mixture.py tests/test_axiom_interaction.py. Dependencies 848355c and d1b1953 are in the shared branch. Parent owns the final suite, documentation and dashboard.
