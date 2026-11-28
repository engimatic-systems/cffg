---
id: cff-3nr4
status: closed
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

## Notes

**2026-11-24T05:46:55Z**

Design base is current main (51ae6a7). Change paths are New IR field, Learnable or checkpoint feature, and Runtime numerical or status invariant. The public evidence covers AgentSpec JSON and hashes, model-local and cross-window validation, named parameter writes and marginals, observe/act cadence, model bundles and controller NPZ restore. For implementation verify host carry after ordinary and compiled Engine runs, synchronous/scheduled status gating, isolation of two agents, and current/legacy decoding. A compiled or vmapped stateful Python facade is outside this design. The new design subsection follows ARCHITECTURE §3c, leaving the shipped categorical contract in place.

**2026-11-24T05:49:57Z**

The contract is in docs/ARCHITECTURE.md §3c, “Continuous and hybrid rolling boundaries.” It specifies a full Gaussian state, raw mean or discrete first-action readout, exact applied affine input or observed action selector, transactional learning after perception, marked failure/hold, and versioned restart with a pending decision. A persistent hidden switch would need conditional state moments. Carrying one Gaussian and a separate regime marginal cannot distinguish swapped regime/state associations. The AXIOM pixel, bank and planner state stays with its application owners pending a separate API decision.

Question: can the shipped public facade actually restart from files? On main 51ae6a7 the categorical probe used two interpreters. The writer used cffg.RollingAgent with a learned B, sampled a proposal, observed an overridden action, and wrote a tick-1 NPZ. The reader used cffg.load_bundle for both windows, constructed with a different RNG seed, restored, and consumed the same two further transitions. Next actions [2, 0] and all 18 final checkpoint fields matched exactly at tick 3. Model bundles must use their own window's learning-source descriptors. Reproduce with JAX_ENABLE_X64=1 uv run --frozen --no-sync python ~/agent-share/cff-3nr4-design/restart_probe.py write, then read in a new interpreter. This verifies categorical continuation only. The Gaussian/hybrid witnesses are implementation acceptance cases.

Evidence for the design: the public AgentSpec validator and RollingAgent._with_params currently expect categorical priors; _agent_checkpoint already stages restored state before mutation. The PCGym common/graphs.py hybrid_plan uses the physical action as selector. tests/hybrid_planning_fixture.py documents the goal-dependent moment-collapse limitation. The separate AXIOM tracking worktree's component_bank.py records live generations and exact bank snapshots; a Gaussian marginal cannot reconstruct them. These source-level ownership checks support the scope. The test command selecting test_agent_spec_json_bundle_and_hash_round_trip, test_two_block_agent_updates_resets_and_checkpoints_exactly, test_improper_edge_readout_is_explicit_nan and test_serialization_round_trip_compiles_and_agrees passed the three short tests; the last is marked slow and was deselected. Design and ticket changes only; local integration owns the new §3c paragraph. Follow cff-wdhj for the implementation and parity gates.
