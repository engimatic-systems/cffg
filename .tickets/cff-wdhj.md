---
id: cff-wdhj
status: closed
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

**2026-11-24T06:02:19Z**

First implementation slice on feature/cff-wdhj-gaussian-boundary. RollingAgent.from_gaussian_bundles opens a GaussianPlanSpec / GaussianFilterSpec pair with fixed affine parameters, one full Gaussian state and deterministic vector mean command. The public propose, observe, checkpoint and restore methods work before and after feedback. The filter consumes exactly b + B @ applied; rebinding a preview never updates the filter. A failed solve raises the Engine diagnostic. New-format restore stages the whole record, binds both model structures and blob digests plus compute settings, and preserves the pending decision. The README control-loop example and docs/ARCHITECTURE.md §3c describe this slice. New dashboard file is tests/test_gaussian_agent.py in M5.

Implementation dependencies left on this branch: none for the fixed affine slice. To finish the full ticket, extend the lifecycle to declared filter-source learning and exact next-update restart, then the known action-selector hybrid, selection RNG and marked hold policy. Qualify their schedule/agent-isolation and larger-horizon cases. Preserve existing categorical semantics. The cumulative short-suite gate belongs to the integrated head. A custom AXIOM checkpoint extension is a separate scope decision; the first slice does not depend on that decision.

**2026-11-24T06:03:20Z**

Focused validation at the proposed implementation bytes: all seven cases in tests/test_gaussian_agent.py passed, including the opt-in fresh-interpreter restart. The physical witness produced command 0.8, post-override mean (0.3,0.15) and off-diagonal covariance 0.12 in float32 and float64. A second interpreter loaded both .cffgz models and the pending .npz and reproduced the next mean, full covariance, command and free energy exactly. Hostile cases cover an unresolved float32 prior, a DATA-edge cavity masquerading as a posterior, a stale control matrix, damaged snapshot, injected budget exhaustion, invalid feedback, and duplicate feedback. The float32 FE error against the analytic normalizer was 1.50e-6; the test uses 16 machine epsilons relative. Also passed 65 existing short cases in agent_spec, serialize, ir, compose, bundle, and the public API authority/checkpoint envelope (1 deselected). No independent full gate or timing measurement. Diff hygiene and ticket dependency cycle checks clean.

**2026-11-24T06:09:02Z**

The cumulative branch renamed the covariance-rank helper. Integration first exposed five facade imports that failed. The agent now uses the shared ef.resolved_correlation entry point. The original unresolved float32 covariance regression and the combined public-agent and reducer tests passed on the integrated source.

**2026-11-28T04:44:37Z**

The hybrid and learned-affine slice continues from #208 (d07e8cf). The architecture change paths are New IR field, Learnable or checkpoint feature, and Runtime numerical or status invariant. Affected pairs are declaration validation/JSON/hash/scoping and bundle load, fixed-affine/extended v2 and unchanged categorical checkpoints, host carry over ordinary/scheduled Engine runs, and separately interleaved agents. The stateful Python controller has no vmapped execution surface. Factor registration, assembly and compiled message rules have no new owner or layout in this slice.

Can the filter learn from a known switch without dropping state/selector dependence? The Engine binds the applied categorical DATA as exactly one selector, uses a separate affine sensor and exposes the selected transition joint to component MNIW. See lifecycle.py bind_data, specialized/discovery.py hybrid endpoints and mixg.py component joint readout. This is a code-level finding independently checked against the single-transition Kalman and MNIW oracle. A hidden persistent regime would require a different carry.

Can the plan use the learner without consuming planning targets as evidence? Learnables.write and accumulate use compiled target roles and Engine source statistics separately (src/cffg/learn.py). The filter is the only update owner, and compatible planner declarations consume current rollover parameters. The controlled affine offset has a distinct per-transition writer, so the continuous fixture learns a separately owned sensor. This is a code-level finding checked against a conjugate oracle and exact restart.

**2026-11-28T04:45:30Z**

Focused gate passed 57 cases across tests/test_gaussian_agent.py, tests/test_hybrid_rolling_agent.py, tests/test_agent_spec.py, tests/test_agent.py, tests/test_bundle.py and tests/test_readme.py, including opt-in cases (uv run --frozen --no-sync pytest -q -o addopts='' -n 0). After adding the affine H=10 analytical witness, its complete file passed all 11 cases. The hybrid H=10 test restored both pending and committed cuts in a fresh interpreter, then produced identical complete archives. The two-type fixtures bind the actual override, skip missing-observation learning, qualify f32/f64 and reject unresolved boundaries. Retry after failed learning, held-command restart, synchronized duplicate feedback, malformed RNG and same-layout/different-prior rejection have independent checks. The final short dashboard gate follows on the committed code.

**2026-11-28T05:35:12Z**

Final short-suite gate passed on ba02eac with 2,844 passing tests, zero failing, and 443.282 seconds measured wall time. The dashboard command was run under the shared CPU lock. The runtime history and ticket closure are the only changes after the qualified code commit. The PR is separate and depends on #208. Hidden-regime state, vmapped controller execution and application-specific AXIOM checkpoints remain outside this facade.
