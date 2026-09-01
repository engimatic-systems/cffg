---
id: cff-cfhi
status: open
deps: [cff-7ozr]
links: []
created: 2026-09-01T02:20:19Z
type: feature
priority: 1
assignee: coopwilliams
external-ref: gh-85
---
# Clean-port newcomer API and strict runtime checkpoints

## Background in Plain English — WHY

Current users must assemble a basic workflow from deep module paths, and the
live `RollingAgent` checkpoint format does not bind every fact required for an
exact continuation. Old PR #85 added a small top-level facade, an executable
README path, versioned runtime identity, atomic restore, and strict malformed-
state rejection. That work is distinct from the CFFGZ resumable-learning
roadmap already on current `main`: this ticket concerns a running agent's
short-horizon state and stochastic continuation.

## Outcomes / Acceptance Criteria — WHAT

1. The top-level package exports one small supported pre-1.0 workflow from
   model construction and bundle load through compilation, inference, agent
   checkpoint, fresh restore, and continuation, without duplicating canonical
   implementations.
2. README contains an executable newcomer example that works in both JAX
   float32 and x64 modes and checks inference status before using beliefs,
   objective values, or actions.
3. New `RollingAgent` checkpoints use an explicit versioned envelope and exact
   logical and physical field inventory.
4. The envelope binds planning and filtering structure identity, grounded
   static parameters, runtime solver/action settings, construction-time JAX
   precision, boundary-belief shapes and values, complete learnables,
   learned-goal state, cadence, NumPy RNG, and functional-`mu3` configuration
   and stochastic continuation.
5. Restore rejects unknown, missing, duplicate, partial, ambiguous, malformed,
   incompatible, non-finite, or stale state before mutating any live field.
   Failed restore is atomic even for an agent that has already been used.
6. Static-parameter identity covers grounded plate copies and tied values read
   by both learned and static consumers.
7. Explicit legacy migration accepts only complete and unambiguous historical
   checkpoints and rejects state that cannot reproduce continuation.
8. Tests cover both deterministic and functional-`mu3` agents, float32 and x64,
   exact-zero beliefs, malformed inputs, incompatible structures and settings,
   already-used restore targets, and models at least ten times a minimal
   fixture where practical.
9. Architecture and README distinguish live `RollingAgent` runtime checkpoints
   from CFFGZ bundle learning-state persistence.
10. The implementation is a semantic port from old PR #85 adapted to the
    preceding clean-port layers and current-main recurrence checkpoint formats.

## Methodological Suggestions — HOW

These suggestions are optional. An implementer may choose another sound method.

- Adapt old commit `f7b584da` and resolve its overlap with the functional-`mu3`
  layer to the stricter combined invariant.
- Use the **Learnable or checkpoint feature** change-path row. Current,
  versioned, and legacy decoding all apply. Beliefs, learnables, goals, cadence,
  NumPy RNG, JAX key state, static grounded parameters, plates, and runtime
  configuration all participate. Compiler/rule execution, host/JIT,
  flooding/scheduled, and `Colony`/`vmap` do not change.
- Stage and validate every decoded field in temporary immutable data before
  committing any change to the agent.
- Move cohesive checkpoint encoding and validation into a private helper module
  so `agent.py` stays below the repository's file-size caution threshold.

## Future work

`cff-9tvt` adds post-integration cache corruption and staleness rejection and
reconciles the final bounded release text. CFFGZ persistence for complete
declared-learning state remains owned by `cff-cz6v` and its child tickets.

## Definitions

- **Atomic restore:** Restore either accepts all checkpoint state or changes no
  live agent state.
- **Facade:** A small public import surface that delegates to existing owners.
- **Grounded static parameter:** A concrete compiled parameter value, including
  plate-expanded and tied-reader values, that is not replaced by learning.
- **Runtime checkpoint:** A snapshot of one live `RollingAgent` continuation,
  distinct from a serialized model or reusable learning bundle.

## Notes

**2026-09-01T02:23:06Z**

Planning audit, 2026-08-31. Capability owners are the top-level facade delegates, RollingAgent checkpoint encoder/decoder, Learnables state validation, bundle identity helpers, and the named public/checkpoint tests and docs. Wiggle scan reviewed in context; retained candidates mean: canonical=existing implementation owner; integrate=combined deterministic and mu3 envelope; public=facade contract; reusable=CFFGZ bundle scope; implement=old-PR behavior or exported workflow; support=checkpoint/API promise; work=explicit roadmap distinction; complete=enumerated state inventory; remaining=named downstream hardening; check=status guard; exact=continuation/value identity; reproduce=continuation contract; sound=optional method quality; test=automated artifacts; validate=input staging; infer=runtime inference; learn/learnable=declared parameter state; small=bounded facade or model; may=optional method; explicit_owner/named_ticket_handoff=cff-9tvt or cff-cz6v; all/every/any/at_least/only=the locally enumerated fields, failures, modes, and migration boundary; current=main 5c3b06a, active format, or decoder; existing=delegated owner; read=compiled consumer access.
