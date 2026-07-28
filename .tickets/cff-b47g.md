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
# Factored two-chain rollout with explicit product selectors

Add a reusable two-chain receding-horizon builder for states such as position x orientation or location x mode. Each chain has its own initial prior and transition block; a transition may select on action alone or on the deterministic Cartesian product of the other chain and action. Observation likelihood is declared over either one chain or their product. Parameters are tied across time by shared blob identity and learning uses declared LearnableSpec sources/targets. This is builder and wiring work only, after cff-nhxf supplies the multi-block agent contract.

## Design

Use existing primitives exclusively. Construct a product selector with a deterministic transition_mixture table whose out state enumerates the Cartesian pair, and synthesize the required ternary equality branches inside the builder so action, chain state, product selector, transition and observation each remain valid degree-two FFG edges. Do not register a pair factor or add compiler/engine branches. Keep this helper private to the builder and link the generic fan-out ergonomics ticket rather than expanding this ticket. Correctness oracle: flatten the factored conditional and observation tensors into one explicit joint-state rollout at identical fixed parameters, then compare joint marginals and BFE. Tying is tested separately; removing tying is not the parity operation. Hostile-input plan: deterministic product tables exercise exact-zero support; float32 compile/run is required; a 32x4 model must not allocate learned parameters with the flat 128-state quadratic count.

## Acceptance Criteria

The builder emits planning and one-step perception templates with serialized AgentSpec bindings and at least two declared belief blocks. Action-only and product-selected chain transitions compile and converge. A small model matches an independently flattened joint rollout in joint marginals and BFE for fixed parameters. The factored learned parameter count is strictly below the flat encoding, and statistics from one time member of a tied blob improve prediction at another without hand-written accumulation. Observation matrices can range from state-identifying to aliased through an explicit builder argument. Templates and AgentSpec serialize/hash round-trip. Exact-zero deterministic, float32, and 32x4 scale cases pass. No new FactorType or central compiler/engine dispatch.
