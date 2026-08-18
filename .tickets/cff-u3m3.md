---
id: cff-u3m3
status: closed
deps: [cff-epdp]
links: []
created: 2026-08-13T04:26:16Z
type: feature
priority: 1
assignee: coopwilliams
parent: cff-uy48
tags: [usability, public-api, release]
---
# Stabilize one newcomer path from model build through checkpoint restore

## Background in Plain English

The repository has real model builders, serialization, compilation, inference,
objective diagnostics, a rolling agent, and checkpoint restore. A newcomer must
still learn deep module paths to combine them. The top-level `cffg` package
exports almost nothing and retains a scaffold `hello()` function. README
examples therefore demonstrate internal capability without defining one small
supported facade.

This ticket establishes the minimum public path required by the release gate.
It does not freeze every pre-1.0 Python object or redesign the modeling API.

## Outcomes / Acceptance Criteria

1. `src/cffg/__init__.py` removes the scaffold `hello()` function and its stale
   export.
2. The top-level package exposes a deliberately small documented facade for
   the quickstart's model construction, compilation, inference, status,
   inspection, agent, and checkpoint operations.
3. Each top-level export has one canonical implementation owner. The facade
   re-exports it and does not add a second behavior implementation.
4. README contains one newcomer path that uses supported imports and performs
   these operations in order:
   4.1. build a serialized model template and parameter arrays;
   4.2. compile the template;
   4.3. initialize and bind data;
   4.4. run inference and require a successful status;
   4.5. inspect at least one posterior and the correctly named objective;
   4.6. construct the corresponding rolling agent;
   4.7. perform one environment step;
   4.8. write a checkpoint;
   4.9. restore into a fresh agent; and
   4.10. continue with the same belief, learned state, cadence, and random
   sequence.
5. An executable documentation test runs the complete path in a temporary
   directory using only the imports shown to the newcomer.
6. The example states whether its objective is exact BFE, exact GFE for a named
   discrete composite, or a declared bound. It does not label every
   `Engine.bfe` result exact.
7. The example fails loudly on a nonconverged result instead of inspecting or
   acting on it.
8. Checkpoint restoration either reproduces the documented continuation or
   raises a specific version, structure, static-parameter, runtime-configuration,
   field, or shape error. It cannot silently discard a known field.
9. The public facade policy states which names are supported before 1.0 and
   which deeper extension interfaces remain internal-unstable.
10. README and architecture descriptions of edit operations, bundle APIs, and
    checkpoint formats name the APIs that actually exist. Planned high-level
    edit bundles are not written as shipped functions.
11. The example runs in float32-capable JAX configuration without relying on
    the test suite's forced x64 setting.
12. The default short suite and the executable newcomer test pass after the
    final code change.

## Methodological Suggestions

These suggestions are optional. An implementer may choose another sound method
that satisfies the numbered outcomes with equivalent evidence.

- Prefer re-exports of stable existing types over a new facade class.
- Use a small discrete builder so the quickstart is fast and its objective has
  an independent exact interpretation.
- Keep the executable example in an existing test or README-snippet authority;
  do not create a second tutorial document.
- Use the **New IR field** and **Learnable or checkpoint feature** checklists
  only for surfaces the facade actually exposes. The ticket should not change
  serialized structure merely to shorten imports.

## Future work

An interactive GUI, a general sensor transport, arbitrary online optimization,
and a frozen external-kernel Python API remain outside this minimum path.
`cff-209l` owns external-kernel execution contracts, but not a frozen public
Python surface. The GUI, transport, optimizer, and API freeze have no assigned
implementation ticket.

## Definitions

- **Facade:** A small supported set of imports that delegates to existing
  implementation owners.
- **Fresh agent:** A newly constructed agent with the same model declaration as
  the checkpoint source and no copied in-memory runtime state.
- **Public API:** A documented callable or type that the project intends users
  to import directly.
- **Scaffold function:** Placeholder code created by a project template and not
  part of the intended product.

## Notes

**2026-08-18T22:09:19Z**

Implementation handoff, 2026-08-18. Applicable change path: Learnable or checkpoint feature. Authority remains AgentSpec/BeliefBinding and Learnables; this ticket adds a versioned RollingAgent envelope without changing serialized model IR. Required surfaces: current and legacy checkpoint decoding are both applicable; belief, learnable, goal, cadence, NumPy RNG, version, logical/physical field inventory, plan/filter structure and static-parameter hashes, runtime solver/action/JAX-precision configuration, and shape validation are applicable. Compiler/Arena execution changes, host/JIT rule pairs, flooding/scheduled pairs, Colony/vmap, and functional mu3 state are outside this sibling ticket; cff-in0q owns functional evaluator checkpoint state and cff-uy48 must integrate both envelope contracts. The facade re-exports canonical owners only. New IR fields, compose/edit codecs, and bundle schema migration do not apply. The plate schema is unchanged, while grounded parameters from every plate copy participate in the static-parameter digest. Existing structure_hash is reused and the digest follows the bundle's content-hashing boundary. Hostile checkpoint cases include unknown, missing, or duplicate fields; wrong structure, static parameters including plated or tied-reader data, configuration, belief/learnable/goal shape or value; malformed RNG/cadence; ambiguous or incomplete legacy spellings; and atomic restore into an already-used agent.

**2026-08-18T22:09:19Z**

Research distillation, 2026-08-18. Question: what is the smallest complete public path? Answer: a deterministic two-cell gridworld perception template saved and loaded as .cffgz compiles to exact tree BP in float32; bind STAY and observation 0, require CONVERGED, inspect the state posterior, and name Engine.bfe as exact BFE = -log evidence for this tree only. A horizon-one B-learning RollingAgent then demonstrates environment step, fresh restore, learned state/cadence/NumPy RNG continuation. Source: src/cffg/models/{gridworld,rollout}.py, bundle.py, compiler.py, engine.py, agent.py; confidence: high, executed in a fresh JAX_ENABLE_X64=0 process. Question: what facade/envelope gap remains? Answer: re-export CorridorEnv, Engine, RollingAgent, RunStatus, build_gridworld, compile_template, load_bundle, save_bundle, and existing set_params from their canonical modules; remove hello. The epdp checkpoint lacks version, structure binding, expected belief shape, unknown-field rejection, and atomic staging. Source: src/cffg/__init__.py and agent.py plus tests/test_agent*.py; confidence: high.

**2026-08-18T22:45:40Z**

Audit distillation, 2026-08-18. Question: which state outside beliefs and learned counts changes an exact RollingAgent continuation? Answer: plan/filter static parameter values, horizon, sweep budget, damping, action-selection mode, construction-time JAX precision, cadence, and the NumPy bit-generator state all affect continuation and are now envelope-bound. Grounded plate-copy parameters and blobs tied between a checkpoint-replaced reader and a static reader remain in the static digest. Legacy migration requires one unambiguous normalized belief, every declared learnable hyperparameter, and RNG state for stochastic selection. Duplicate physical NPZ members, complex learnable arrays, invalid goal/count state, and malformed or overflowing RNG state fail before mutation. Source: adversarial GPT-5.6-sol reproductions against src/cffg/agent.py and src/cffg/learn.py, encoded in tests/test_public_api.py; confidence: high, independently re-audited clean after the fixes.

**2026-08-18T22:56:56Z**

Completion handoff, 2026-08-18. The supported pre-1.0 facade and executable README path cover bundle build/load, compilation, DATA binding, converged exact-tree BFE inspection, rolling action/environment interaction, versioned checkpoint, fresh restore, and deterministic continuation in float32 and x64. The versioned envelope is bound to compact structure, grounded static parameters, solver/action settings, construction-time JAX precision, normalized beliefs, complete learnables, learned-goal state, cadence, and RNG; restore validates atomically and legacy migration is explicit. Two independent GPT-5.6-sol audits report no remaining actionable findings. Final default not-slow suite: 1376 passed, 39 deselected, 0 failed in 639.74 seconds; dashboard refreshed at /Users/ravd/agent-share/cffg-tests.html. Next: cff-uy48 integrates this sibling with cff-in0q functional mu3 checkpoint/runtime state, including mu3 configuration in the envelope.
