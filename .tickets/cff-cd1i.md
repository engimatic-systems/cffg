---
id: cff-cd1i
status: open
deps: []
links: []
created: 2026-07-08T22:00:30Z
type: feature
priority: 1
assignee: coopwilliams
---
# Dirichlet B-learning: conjugate transition updates + node-belief readout

Learn transition matrices online, the highest-value learning gap (GPT-5.5 feedback item 1, confirmed: RollingAgent is hard-coded around A-counts, agent.py learn_a). MATH (Fable, top level): per-action Dirichlet columns q(B_u) with conjugate VMP updates from the SMOOTHED PAIR BELIEF at the transition_mixture node — delta_counts_u[z',z] = q_a(z',z,u) summed over trial slices; planning slices then use E[log B_u] = digamma(alpha)-digamma(alpha0), exactly mirroring the A path in experiments/tmaze_learning.py. ENGINEERING GAP THIS EXPOSES: the engine only reads EDGE marginals; conjugate updates need NODE beliefs (the joint q_a at a factor). Add Engine.node_beliefs(state, factors) reading the gathered incoming messages + logf per group — generally useful (it is also what d/e learning and the FE decomposition diagnostics want). Then: build_rollout grows learned_b (counts blob 'Bc'), RollingAgent grows learn_b with count rollover + checkpoint, agent test recovers a known stochastic B from trials (gridworld with slip probability; note RockSample's B is deterministic 0/1 — B-learning targets stochastic/unknown dynamics like the glucose world model, cff-t2g3). GFE composites are untouched: B uncertainty enters planning through E[log B] only; the epistemic-value-of-B-learning channel (info gain about dynamics) is NOT included here — it needs a P-sub-style derivation, note it in LIMITATIONS when landing. DELEGATION: Fable derives update + node-belief semantics; Opus can mirror the tmaze_learning A-path plumbing and write the recovery test once node_beliefs exists.

## Acceptance Criteria

node_beliefs API with tests (pair belief on HMM matches forward-backward pairwise marginals); learned B converges to truth on a slip-gridworld across trials; existing A-learning tests untouched.

