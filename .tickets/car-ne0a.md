---
id: car-ne0a
status: in_progress
deps: [cff-lquu]
links: [cff-21vm, cff-pyy9]
created: 2026-11-23T21:48:38Z
type: feature
priority: 2
assignee: coopwilliams
parent: car-zieb
tags: [axiom]
---
# Plan with batched stochastic action sequences and AXIOM’s exploration bonus

## Background in Plain English — WHY

The released agent chooses an action by trying many possible strings of actions inside its learned world model. It favors predicted reward and encounters with poorly sampled interaction rules. This is a useful CFFG planning interface in its own right and can be developed against a tiny known world before object perception works.

## Outcomes / Acceptance Criteria — WHAT

1. Provide a supported action-conditioned forward sampler from a read-only belief/parameter snapshot with explicit random keys per policy and replicate. In the AXIOM model sample interaction component, conditional motion class and predicted object state/reward. Hypothetical experience leaves learned counts unchanged. Check exact probabilities on a small enumerable world and reproducibility across scalar/batched execution, checkpoint and first-action replan.
2. Implement the release’s score aggregation over objects and time. Include the reciprocal Dirichlet-count novelty term, low-match and out-of-frame rules, tracking masks and the reference 0.1 weight. Report reward and bonus separately. Hand-score at least one rare and one familiar component, a masked object and an out-of-frame prediction. The count bonus is a proxy; keep its identity distinct from measured information gain and from the library’s GFE.
3. Generate and refit a factorized categorical proposal with the release’s random/smoothed candidates, prior best, constant strings and elite-weighted histogram. At reference budget use 32 steps, 512 candidates and 3 stochastic rollouts per candidate. Test two policies with opposite order under discounted proposal refitting and undiscounted final selection. Execute the first move of the best undiscounted sequence and replan after the next real observation.
4. Qualify ties, zero supported actions, nonfinite scores, constant/zero reward and both dtypes. Record synchronized time and memory at the reference batch on identified hardware. Bound memory by batching if necessary and qualify numerical and sampling equivalence. Publish a reproducible no-bonus ablation switch.

## Methodological Suggestions — HOW

First qualify the policy search on injected predictive kernels with exact small-world answers. Then use the same sampler contract for live rMM/tMM snapshots in car-zieb. cff-lquu gives existing one-step hybrid readouts; cff-sj1w demonstrates batching at a different level. Review the “Runtime numerical or status invariant” path for status, host/compiled and single/batched results. Flooding and scheduled GFE comparisons are N/A for sampling fixed action strings. Benchmark counters must distinguish real from imagined steps.

## Future work

The separate cff-pyy9 roadmap branches a future decision on a hypothetical observation. AXIOM samples whole fixed action strings. Joint NIW parameter sampling (cff-21vm) and Colony optimization can be pursued on their own evidence.

## Definitions

A proposal is the distribution used to suggest action strings. An elite is a high-scoring candidate used to improve the proposal. Discounting gives less weight to later rewards. A random key makes one stream of draws reproducible.

## Notes

**2026-11-24T05:16:09Z**

Implement independent read-only sampled planning against an enumerable world. Architecture paths: Agent, planning, or hierarchy feature and Runtime numerical or status invariant. Check shared callback, scalar/batched trajectories, explicit RNG and cold/warm planning, checkpointed planner state, masks, invalid settings and reward aggregation in both precisions. CFFG message sweeps, GFE and template/bundle migration do not own this sampled objective. Object-model integration depends on the native pipeline.

**2026-11-24T05:30:00Z**

Release-budget planner probe completed with supplied two-object, three-interaction snapshot: 32×512×3=49,152 imagined steps and zero real steps. CPU float32 compile 0.732 s, one synchronized warmed decision 21.56 ms. Exact synthetic rollout distribution/score and selection checks plus optional no-bonus comparison passed. Live features and model binding belong to car-mps6. Do not extrapolate these numbers to a 32-object, 5,000-interaction bank or game reward. Shared suite pending.

**2026-11-24T06:02:02Z**

Planner qualification covers one compact live rMM/tMM snapshot, relative and absolute scores against a full-bank query, read-only updates, bitwise-matched keys, no-bonus and reversed discounted/undiscounted rankings. A 32-step ×512 strings ×3 draws synthetic two-slot/three-rule case ran once warm in 21.56 ms. A 5,000-slot bank with eight occupied rules and six tracked objects passed a separately qualified H8×16×3 development search; its 384 imagined transitions took 7.16–7.86 ms after setup. Neither measurement is a 10,000-step Gameworld score.
