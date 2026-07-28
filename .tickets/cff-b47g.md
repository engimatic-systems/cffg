---
id: cff-b47g
status: open
deps: [cff-nhxf]
links: [cff-mjku]
created: 2026-07-28T04:19:36Z
type: feature
priority: 2
assignee: coopwilliams
tags: [builder, multi-chain, active-inference]
---
# Factored rollout builder with explicit product selectors

## Plain English

This ticket adds a reusable builder for dynamic models whose hidden state is
split into named, interacting parts such as position, orientation, inventory,
or mode. Each part can keep a compact transition model while declared
couplings describe how the parts influence one another. The two-part corridor
is the first concrete model built with the general interface.

Add a reusable factored receding-horizon builder for states such as position x orientation or location x mode. The public contract accepts a collection of named state-block specifications and coupling specifications and supports any number of blocks. Each block has its own initial prior and transition block; a transition may select on action alone or on a deterministic Cartesian product of declared parent blocks and action. Observation likelihoods are declared over one block or a declared product. Parameters are tied across time by shared blob identity and learning uses declared LearnableSpec sources/targets. The first full fixture is the two-block factored corridor, after cff-nhxf supplies the multi-block agent contract.

## Design

The public API is topology-generic: it iterates over any number of state-block
and coupling declarations using the model's own block names. A two-block
corridor provides the detailed parity and learning oracle;
a three-block construction smoke test proves that the interface itself has no
two-block ceiling. `RollingAgent` consumes only the serialized boundary and
I/O bindings from cff-nhxf and remains independent of the generated interior
topology.

Use existing primitives exclusively. Construct product selectors with deterministic transition_mixture tables whose out state enumerates the declared Cartesian tuple, and synthesize the required ternary equality branches inside the builder so actions, state blocks, product selectors, transitions and observations each remain valid degree-two FFG edges. Keep this helper private to the builder and link the generic fan-out ergonomics ticket rather than expanding this ticket. Correctness oracle: flatten the two-block factored conditional and observation tensors into one explicit joint-state rollout at identical fixed parameters, then compare joint marginals and BFE. Tying is tested separately; removing tying is not the parity operation. Hostile-input plan: deterministic product tables exercise exact-zero support; float32 compile/run is required; a 32x4 model must not allocate learned parameters with the flat 128-state quadratic count.

## Acceptance Criteria

The builder emits planning and one-step perception templates with serialized AgentSpec bindings from a collection of named state-block and coupling specs. Action-only and product-selected block transitions compile and converge. A two-block model matches an independently flattened joint rollout in joint marginals and BFE for fixed parameters, and a three-block model compiles, runs and round-trips through the same builder API. The factored learned parameter count is strictly below the flat encoding, and statistics from one time member of a tied blob improve prediction at another without hand-written accumulation. Observation matrices can range from state-identifying to aliased through an explicit builder argument. Templates and AgentSpec serialize/hash round-trip. Exact-zero deterministic, float32, and 32x4 scale cases pass. No new FactorType or central compiler/engine dispatch.
