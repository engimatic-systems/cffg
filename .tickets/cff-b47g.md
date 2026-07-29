---
id: cff-b47g
status: closed
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

## Notes

**2026-07-29T18:10:11Z**

Implemented the topology-generic factored rollout builder and canonical position x mode corridor. StateBlockSpec/CouplingSpec/ObservationSpec lower ordered C-style products to deterministic transition_mixture chains and all fan-out to ternary equality chains; separate serialized boundaries use neutral explicit mean-field anchors on the complete host Template. Transition/initial/observation LearnableSpecs tie blobs across time with past-only sources. Coverage pins planning/filter AgentSpec and .cffgz/JSON/hash round trips, RollingAgent operation, action-only and product-selected convergence, independently flattened joint-marginal and BFE parity, a three-block construction, tied-statistic propagation, exact-zero product support, float32, and 32x4 factored parameter scale. README/ARCHITECTURE/LIMITATIONS were updated in place. Final proposed-head dashboard suite: 1281 passed, 0 failed, 1 new in 353.03s; focused affected suite: 52 passed; compileall, git diff --check, and tk dep cycle passed. Next M3 path ticket after this PR lands is cff-5ipi, which preregisters the experiment before any cff-2gev result is run. Gotcha: learned_a/learned_c/epistemic_b uncertainty composites are intentionally rejected by this builder until their separate follow-on work.
