---
id: cff-cd1i
status: closed
deps: []
links: []
created: 2026-07-08T22:00:30Z
type: feature
priority: 1
assignee: coopwilliams
---
# Dirichlet B-learning: conjugate transition updates + node-belief readout

Learn transition matrices online, the highest-value learning gap (GPT-5.5 feedback item 1, confirmed: RollingAgent is hard-coded around A-counts, agent.py learn_a). MATH (Fable, top level): per-action Dirichlet columns q(B_u) with conjugate VMP updates from the SMOOTHED PAIR BELIEF at the transition_mixture node — delta_counts_u[z',z] = q_a(z',z,u) summed over trial slices; planning slices then use E[log B_u] = digamma(alpha)-digamma(alpha0), exactly mirroring the A path in experiments/tmaze_learning.py. ENGINEERING GAP THIS EXPOSES: the engine only reads EDGE marginals; conjugate updates need NODE beliefs (the joint q_a at a factor). Add Engine.node_beliefs(state, factors) reading the gathered incoming messages + logf per group — generally useful (it is also what d/e learning and the FE decomposition diagnostics want). Then: build_rollout grows learned_b (counts blob 'Bc'), RollingAgent grows learn_b with count rollover + checkpoint, agent test recovers a known stochastic B from trials (gridworld with slip probability; B-learning targets stochastic/unknown dynamics like the glucose world model, cff-t2g3 — deterministic-dynamics tasks gain nothing from it). GFE composites are untouched: B uncertainty enters planning through E[log B] only; the epistemic-value-of-B-learning channel (info gain about dynamics) is NOT included here — it needs a P-sub-style derivation, note it in LIMITATIONS when landing. DELEGATION: Fable derives update + node-belief semantics; Opus can mirror the tmaze_learning A-path plumbing and write the recovery test once node_beliefs exists.

## Acceptance Criteria

node_beliefs API with tests (pair belief on HMM matches forward-backward pairwise marginals); learned B converges to truth on a slip-gridworld across trials; existing A-learning tests untouched.


## Notes

**2026-07-08T22:24:46Z**

LANDED (2026-07-08, branch b-learning, commits 0440ca5+b8b49cc). All acceptance criteria met: node_beliefs pair == brute-force enumeration on a 2-step chain at 1e-9; slip-corridor B recovered from 2500 random-walk transitions (worst column TV 0.095, threshold 0.12); A-learning tests untouched (178 green).

DEVIATION from plan: NO template/blob change was needed — 'learned_b'/'Bc' don't exist. B has no GFE composite, so uncertainty enters planning only through E[log B], which is a pure runtime params-dict swap on transition_mixture groups. The compiler's column-stochastic validation applies to the compile-time blob only; runtime swaps bypass it by design (params are inputs).

GOTCHAS: (1) the count increment uses the filter's ONE-SLICE pair belief — a fixed-lag or end-of-trial smoother would give lower-bias increments (same trade the A path makes; revisit in cff-x0sv); (2) early counts accumulate under a still-ignorant model (online-EM bootstrap bias) — recovery tightens with data, don't panic at TV~0.19 after a few hundred steps; (3) LIMITATIONS S1c: B uncertainty generates NO epistemic value yet — dynamics-curiosity needs a P-sub-style derivation, decision parked with cff-x0sv.
