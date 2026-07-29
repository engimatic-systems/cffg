---
id: cff-nhxf
status: closed
deps: [cff-x0sv]
links: []
created: 2026-07-28T04:19:16Z
type: feature
priority: 2
assignee: coopwilliams
tags: [agent-interface, multi-chain, serialization]
---
# Serialized RollingAgent bindings for arbitrary rolling boundary state

## Plain English

This ticket lets a model tell `RollingAgent` where its actions, observations,
and persistent beliefs live. The declaration travels with the serialized
model, so the same agent runtime can drive one state block, several interacting
state blocks, or a joint boundary belief through model-local names and
bindings.

RollingAgent still assumes one latent belief array and literal build_rollout names: arrays d/e, factor prior, marginal u1, DATA ports act1/obs1, and posterior zm1. Declared learnables removed the learning special cases but did not create a model/agent interface. Introduce an additive Template-level AgentSpec, covered by canonical serialization and structure hashing, that declares action readout, action and observation DATA ports, and one or more belief blocks with their prior target, filter posterior readout, and optional episode-scale learnable. RollingAgent consumes this declaration instead of names. The single-chain public API and legacy checkpoints remain compatible.

## Design

AgentSpec is model structure, not an untracked Python builder sidecar: serialization remains the source of truth. Planning and perception templates each carry the bindings applicable to that window. RollingAgent stores beliefs by declared block; the existing belief array remains a compatibility view when exactly one block exists. Validation resolves every referenced factor, variable, port, and learnable and rejects wrong roles or cardinalities before compilation. No inference rule changes. Hostile-input plan: exact-zero block beliefs retain NEG support semantics; a two-block float32 cycle must remain finite; 10-100x scale is N/A to the interface but a 32x4 belief pair prevents accidental joint-state allocation.

The declaration must also make the receding-horizon boundary factorization
explicit. Coupled state blocks generally have a joint filtering posterior
`q(z1, ..., zn)` whose correlations go beyond the individual marginals. Each
boundary binding therefore declares whether it carries a joint belief over
several blocks or separate block marginals under an explicit mean-field
constraint already present in the Template. Its filter readout and next-window
prior target must agree in domain, shape, and factorization. The window
rollover therefore stays faithful to the CFFG's declared variational family
and makes every approximation explicit.

## Acceptance Criteria

Template AgentSpec round-trips through JSON/.cffgz and affects structure_hash. build_rollout declares bindings with no behavioral or checkpoint regression. RollingAgent contains no literal lookups for prior/u1/act1/obs1/zm1 or arrays d/e; action cardinality, binding, prior injection, posterior readout, and episode reset come from AgentSpec. A two-block synthetic model plans, observes, updates both beliefs and declared learnables, resets episode priors, and checkpoint/restores exactly; single-block belief/counts/b_counts compatibility remains. A correlated two-block fixture pins rollover semantics: a joint boundary stays joint, while separate boundary marginals require and preserve the Template's explicit mean-field constraint. Missing or inconsistent bindings fail validation with model-local errors. Exact-zero and float32 cases pass; no new factor type, compiler dispatch, or inference math.

## Notes

**2026-07-29T03:42:01Z**

Implemented serialized AgentSpec/BeliefBinding structure with joint versus explicit mean-field rollover, canonical JSON/.cffgz/hash support, validation, composition/plate handling, and build_rollout declarations. RollingAgent now resolves action, DATA ports, all boundary priors/posteriors, action cardinality, and episode learnables exclusively from AgentSpec; named multi-block beliefs, exact checkpoints (including RNG), and the one-block belief/counts/b_counts compatibility path pass. Acceptance coverage includes a renamed two-block model, both episode learnables, correlated joint rollover, missing/irrelevant mean-field constraints, exact-zero float32, and a factored 32x4 boundary. Final proposed-head dashboard suite: 1275 passed, 0 failed, 5 new; focused affected suite 87 passed; git diff --check, compileall, and tk dep cycle passed. Next ready M3-path ticket after this lands is cff-b47g; its builder must put AgentSpec on the complete host Template (not a plate fragment) and make every separate coupled boundary cite a mean-field constraint that touches its posterior.
