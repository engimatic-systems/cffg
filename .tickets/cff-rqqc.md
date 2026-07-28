---
id: cff-rqqc
status: open
deps: [cff-iwgl]
links: [cff-wbhu, cff-iwgl, cff-sj1w]
created: 2026-07-28T18:06:35Z
type: feature
priority: 2
assignee: coopwilliams
---
# Colony: per-instance PRNG streams and mu3 stochastic status batching

## Plain English

This ticket lets a Colony batch models that use stochastic functional messages such as `mu3`. Each model instance receives its own reproducible random-number stream, particle state, and inference outcome, while the Colony still executes compatible graph structures together.

The capability is generic runtime plumbing, not a temporal or glue-port convention. A graph may use glue for time, hierarchy, space, or any other composition; Colony keys belong to graph instances and stochastic evaluator groups regardless of what the graph represents.

## Design

Add an explicit per-instance PRNG interface to Colony without introducing hidden global randomness. A mu3 instance must be prepared from a caller key, store the returned next key separately from the structure-cache key, reuse its fixed particles for the complete inference attempt, and refresh only through an explicit API. Refactor or wrap the mu3 coordinate update so compatible prepared states can execute under the existing structure-group vmap while ESS, log normalizers, and stochastic statuses remain instance-local. Low-ESS or divergent instances stop independently and do not poison converged peers in the same batch. Kill/spawn and mixed-structure regrouping preserve key-to-instance ownership.

Serialization remains the source of graph structure; PRNG keys, particles, weights, and next-key cursors are derived runtime state and never enter Template JSON or structure hashes. Supplying the same keys intentionally permits common-random-number experiments; ordinary callers can split one master key into distinct instance keys. No API may infer stochastic or temporal semantics from `PortRole.GLUE`.

Hostile-input plan: refuse a missing key for stochastic evaluators; preserve proper sparse Dirichlet proposal behavior and exact log-zero handling from cff-iwgl; isolate all-nonfinite weights and low ESS per instance; retain float32 behavior; exercise 32-128 instances and uncomfortable A dimensions without fixed literal numeric guards. Duplicate keys are valid explicit caller intent, not silently rewritten. Non-mu3 colonies remain key-free and bitwise-compatible.

## Acceptance Criteria

1. A homogeneous Colony of mu3 models with distinct explicit keys matches N independent Engine.prepare_mu3/run executions for particles, weighted moments, FE, ESS, RunStatus, and returned next keys. 2. Replaying the same instance keys is bitwise reproducible; refresh advances each key exactly once per evaluator group and no sweep or repeated run resamples implicitly. 3. Converged, budget-exhausted, and diverged instances coexist in one structure batch without cross-instance contamination; kill/spawn and regrouping preserve key ownership. 4. Mixed mu3/non-mu3 structures batch correctly, and existing deterministic Colony callers require no keys and retain parity. 5. Missing keys and stale base posteriors fail loudly; sparse/exact-zero support, improper proposals, float32, 32-128 instances, and uncomfortable observation/state scale satisfy the numerical robustness contract. 6. README/ARCHITECTURE/LIMITATIONS document per-instance stochastic state as graph-instance runtime state and explicitly preserve glue as a domain-agnostic composition boundary.


## Notes

**2026-07-28T18:06:47Z**

Dependency companion (Legend: prerequisite --> dependent):

[x] cff-sj1w Colony batching foundation
[x] cff-iwgl mu3 runtime + explicit-key evaluator
        |
        +--> [ ] cff-rqqc per-instance stochastic Colony runtime (ready)

cff-wbhu is linked parallel performance work, not a prerequisite: key/status correctness should not be coupled to the later lax.scan throughput refactor. This ticket is not on the cff-mwjo completion path; it removes the documented Colony integration limitation after the core M3 evaluator exists.
