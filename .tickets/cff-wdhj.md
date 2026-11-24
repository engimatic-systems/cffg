---
id: cff-wdhj
status: open
deps: [cff-3nr4, cff-v6th, cff-ectw]
links: [cff-fj4v]
created: 2026-09-25T03:46:26Z
type: feature
priority: 2
assignee: coopwilliams
parent: cff-xa6z
tags: [pcgym]
---
# Implement the continuous and hybrid rolling-agent facade

## Background in Plain English

A reusable agent loop should let an experiment author specify a model and policy rather than reimplement four lifecycle variants. The boundary design defines what information may be carried and how failures and executed actions are handled.

Evidence comes from the PCGym report at `/Users/ravd/agent-share/PCGymCFFG/` and experiment code at `/Users/ravd/experiments/cffg-testing/pc-gym/00/cffg/`, reviewed against CFFG `fce60bd`. Copy only the minimal reproducible inputs into maintained fixtures; do not add external training artifacts or pickle dependencies.

## Outcomes / Acceptance Criteria

1. The public facade implements the approved continuous/hybrid boundary contract with explicit observation, learning, planning, action and checkpoint steps.

2. A manually driven Engine reference and the facade produce equivalent beliefs, objective values and actions on the linear-Gaussian fixture and supported hybrid case.

3. Learning consumes actual executed transitions once, including externally bounded or overridden actions, and does not treat predicted goals as evidence.

4. Nonconverged plans follow an explicit caller-selected policy; no silent stale action is presented as a successful plan.

5. Checkpoint/resume reproduces uninterrupted operation, including posterior learning and random-key progression.

6. Sparse selectors, missing data, improper boundaries, float32 and larger horizons have tested outcomes; existing categorical agents remain compatible.

7. README provides a concise end-to-end example with public imports and links to the capability/limitation contract.

## Methodological Suggestions

Prefer one lifecycle authority over copied experiment controllers. Architecture: New IR field and Learnable or checkpoint feature; Runtime numerical or status invariant for consumer behavior. Pairs host/JIT boundary transforms; flooding/scheduled status parity; single/vmap per-agent isolation; current/legacy checkpoints required.

These are suggestions; another method is acceptable if it satisfies the outcomes. Work branches continue from the preceding validated PR in the PCGym chain, whose initial base is `cff-ucvx-readme-capability-map` (PR #161). Update the intended base before branching. Code changes require focused tests and one passing final short suite; documentation-only work validates its changed artifacts.

## Future work

- cff-wt1c — Validate the PCGym follow-up chain with matched control and runtime comparisons.

## Definitions

Facade: a small public interface coordinating existing components. Predicted goal: a planning preference, not a newly observed transition. Override: an action executed instead of the agent's proposed action.

Shared terms: CFFG means constrained Forney-style factor graph; JAX is the numerical array/compiler library; dtype is a numeric storage type; free energy is the model's inference objective; host/JIT compares Python-side and compiled execution; single/vmap compares one instance with vectorized instances; N/A means not applicable.

## Notes

**2026-11-22T01:02:01Z**

The serialized boundary contract (cff-3nr4) remains a prerequisite. The public recipe collection (cff-fj4v) is related work that can run in parallel. The facade still owes its own public example and equivalence to the manual loop.

**2026-11-24T05:51:05Z**

Implementation assessment after cff-3nr4. The first public extension covers one proper vector Gaussian and either an affine continuous command or a discrete command selecting a Gaussian transition. The known executed switch is the PCGym hybrid use case. Model validation must reject a persistent hidden switch/state joint. Treating its two marginals as independent can change the next action. Both numerical prerequisites are closed. This ticket is ready on the design branch; implementation and a new public README example remain open.

Suggested order:

1. Add the opt-in declarations for the two windows to AgentSpec, validate and scope their references, preserve legacy JSON/hash bytes and test template/bundle round trips. Check the action coordinates, units/order, selector domain, preview DATA ports, and ownership of the mean/covariance and effective filter offset. A Gaussian action DATA port denotes a likelihood under today's Engine; an exactly known affine input uses b + B @ applied. Include B in the static digest even when no factor reads it directly. Use Engine.gaussian_marginal, set_params, bind_data and the existing solver's status and ledger.
2. Reuse the rolling agent's two Engines, converged filter statistics and Learnables. Expose preview, tagged decision, and feedback with the actual applied command through public methods. Discrete MAP/sample and a raw Gaussian mean suffice. A held command must report its reason. Check the last applied override and the very first hold. Inject a failed plan and a failed learning update. Check an isolated retry, repeated planning, wrong or duplicate tick and a missing reading against the one-commit cadence.
3. Add the typed controller checkpoint to the existing staged decoder. Bind both model identities, fixed data, configuration and immutable learner priors. Restore grouped pending information atomically at either proposal or post-feedback. Exercise both .cffgz files through the public API in a fresh interpreter. A small builder that loads the two files suffices. Compare an uninterrupted sequence to a restart at both cuts, including sampled actions and RNG advancement, the next learning update, objective, status and next belief. Reject a missing covariance, a reordered same-size coordinate layout, a different control map, and a checkpoint for another phase before changing live state. Preserve categorical v1 and its historical decoder tests.

Acceptance witnesses:

- Linear. Let the joint state be (x1,x2,u) with mean zero, Var(x1)=Var(x2)=Var(u)=1, Cov(x1,x2)=0.6 and u initially independent. Plan one step with x' = x + (1,0.5) u + w, Q=0.2 I, and a fresh independent unit-variance next control. Observe only x1' with variance 0.3. A target of 2 gives planned E[u]=0.8 and ledger -log N(2; 0, 2.5). In the goal-free filter apply -0.5 exactly and receive +0.5. Expect physical mean (0.3,0.15) and covariance [[0.24,0.12],[0.12,0.96]]. Changing just the preview must leave this filtered result identical. Compare both manual Engine and analytical values. Repeat with a declared filter-source learner active and check its posterior/next update against the manual loop.
- Hybrid. Give two affine motion branches distinct offsets, a categorical first selector, and use its executed index as filter DATA. Compare first q/action and ledger to the same manual Engine approximation, and the selected branch's filter to an ordinary Kalman step. Include zero weight in the plan, a valid external override, a missing sensor, and a hidden-switch/state request that must fail validation. Use the partial-goal fixture in tests/test_hybrid_planning.py to delimit any horizon claims.
- Both. Exercise float32/64, a horizon ten times the small fixture, proper and unresolved covariances, synchronous/scheduled result gating, and two independently interleaved agents. Compare host readouts from compiled execution with the same manual reference. Batched stateful facade execution is outside the contract.

Scope decision for AXIOM integration: a common belief/action contract does not itself checkpoint the pixel tracks, bank generation IDs, retained reduction samples or planner search state. An application-owned versioned envelope can bind their exact owner snapshots, keys and the data cursor to the same agent tick. Verify the next birth/reuse, reduction and planned action after restart on the AXIOM task. Ask Cooper before adding generic custom-state callbacks, model-bank migration, or an AXIOM rollout planner to the public facade. Those features are outside this ticket.
