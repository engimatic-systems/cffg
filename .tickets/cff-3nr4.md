---
id: cff-3nr4
status: open
deps: []
links: []
created: 2026-09-25T03:46:26Z
type: task
priority: 2
assignee: coopwilliams
parent: cff-xa6z
tags: [pcgym]
---
# Specify serialized continuous and hybrid rolling-agent boundaries

## Background in Plain English

AgentSpec and RollingAgent currently require categorical action and belief boundaries. Continuous/hybrid experiments therefore implement observation handling, belief transfer, action readout and restart separately. A facade needs a clear boundary contract before runtime code.

Evidence comes from the PCGym report at `/Users/ravd/agent-share/PCGymCFFG/` and experiment code at `/Users/ravd/experiments/cffg-testing/pc-gym/00/cffg/`, reviewed against CFFG `fce60bd`. Copy only the minimal reproducible inputs into maintained fixtures; do not add external training artifacts or pickle dependencies.

## Outcomes / Acceptance Criteria

1. The design specifies Gaussian boundary moments, required cross-correlations, hybrid regime/state approximations and supported action readouts.

2. Observation timing, goal preview, actual executed-action feedback, learning cadence and failed-plan policy are explicit.

3. Template identity, numeric data, learnable posterior, controller state and random-key ownership are separated for reconstruction and restart.

4. The v1 scope states which continuous/hybrid compositions are supported and rejects others rather than silently dropping correlations.

5. Existing categorical AgentSpec/rolling behavior remains compatible through an explicit schema/version decision.

6. A linear-Gaussian reference and small hybrid case define uninterrupted/resumed and manually driven/facade equivalence; the facade ticket receives acceptance tests.

## Methodological Suggestions

Review AgentSpec validation, BeliefBinding and serialized categorical rolling boundaries before adding a parallel controller system. Architecture: New IR field and Learnable or checkpoint feature. Pairs host/JIT readout/transfer, flooding/scheduled status policy, single/vmap per-agent isolation, current/legacy decode required. Address vacuous/improper boundaries, zero selector support, float32 and long horizons.

These are suggestions; another method is acceptable if it satisfies the outcomes. Work branches continue from the preceding validated PR in the PCGym chain, whose initial base is `cff-ucvx-readme-capability-map` (PR #161). Update the intended base before branching. Code changes require focused tests and one passing final short suite; documentation-only work validates its changed artifacts.

## Future work

- cff-wdhj — Implement the continuous and hybrid rolling-agent facade.

## Definitions

Rolling boundary: the belief carried from one planning window to the next. Cross-correlation: statistical dependence between coordinates. Cadence: when learning occurs. Schema: the structure/version of persisted data.

Shared terms: CFFG means constrained Forney-style factor graph; JAX is the numerical array/compiler library; dtype is a numeric storage type; free energy is the model's inference objective; host/JIT compares Python-side and compiled execution; single/vmap compares one instance with vectorized instances; N/A means not applicable.
