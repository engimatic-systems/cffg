---
id: cff-cxuy
status: open
deps: [cff-x0sv]
links: [cff-cpxj, cff-mjku]
created: 2026-07-27T20:19:11Z
type: feature
priority: 2
assignee: coopwilliams
---
# Factored rollout builder: multi-chain latent state with tied transition blocks

build_rollout is single-chain: one latent variable per slice and one flat [U, K', K] transition. A state that is really a product of factors — position x orientation, position x mode, location x carried-item — has to be encoded as the joint, which multiplies the parameter count and destroys transfer. The true dynamics may be a handful of rules, but every (state, action) column is then learned independently, so nothing observed at one state informs another.

This is a builder plus learnables plumbing, not new engine machinery. The IR already carries the structure: transition_mixture's selector role takes one chain into another chain's transition, factors tie parameters by naming the same blob (build_rollout already ties Bs across every slice), and plates tie parameters by name across copies.

Scope: a builder taking chain specs (cardinality, transition parameterization, which chain selects into it) and an observation factor over the joint. Parameters tied across slices and across the replicated axis. DATA and GLUE ports keep build_rollout's receding-horizon convention so agent drivers work unchanged.

Friction to resolve: no factor conditions on two latent inputs at once except through the selector, so p(a' | a, b, u) needs (b, u) folded into one selector variable. Decide between a pair-construction factor and documenting the fold as the modeling recipe. Coordinate with cff-mjku.

Observability must be a declared parameter of the builder, not baked into a fixed observation matrix. Experiments need to sweep from state-identifying to heavily aliased sensors. That sweep is what turns the S1c optimistic-bound caveat into a measured number instead of a confound: the novelty channel scores information gain as if z' were revealed, which is tight under good sensors and loose under aliasing, and nobody has measured where it stops being usable.

Consumer: cff-2gev needs this. The flat single-chain model gives directed exploration nothing to exploit, because information gained at one state does not transfer to any other, so curiosity pays the full cost of leaving the goal and collects almost none of the benefit.

## Design

Framed as a modeling capability rather than one experiment's fixture. The parity check against the flat model is the correctness anchor: same joint distribution, fewer parameters, and the difference shows up only in what learning transfers.

## Acceptance Criteria

A two-chain factored model compiles and runs, and reproduces the equivalent flat model's marginals when the tying is removed. Parameter count for the factored form is strictly lower than the joint encoding, and learning transfers: counts accumulated at one member of a tied group measurably improve predictions at another. Observability is a builder parameter and a sweep from identifying to aliased sensors is expressible. Learning drives through the declared-learnables API with no hand-rolled accumulation.

