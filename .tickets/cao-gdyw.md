---
id: cao-gdyw
status: closed
deps: [cao-urtu, cao-fs5s]
links: []
created: 2026-11-24T14:51:38Z
type: feature
priority: 2
assignee: coopwilliams
parent: car-zieb
tags: [axiom, evaluation]
---
# Run full-budget AXIOM decisions from an exactly restorable campaign actor

## Background

A 24-action smoke connected the scene learner and policy search. The campaign needs a persistent actor at the published budget, a single real-action clock and a checkpoint that can resume before or after a decision.

## Outcomes

Implement Actor(initial_rgb,game,actions,seed), choose, observe, exact state_dict/load_state_dict. Preserve the proposal, explicit planning key, pending action, model scene, and complete reduction history. A true reset consumes an observation with learning disabled and preserves the real clock. Run the scheduled reducer after each due real action. A rejected step preserves the preceding actor state.

Use all 512 policy strings, 32 imagined steps and 3 draws per string. Limit concurrent policies while retaining the keyed samples, separate reward and novelty, rankings, proposal and selected action. Qualify against existing unchunked output on a small learned development case, including zero tracking; check unequal score, checkpoint after choose, invalid inputs, and restart. Measure a first full-budget plan then warmed development ticks, peak workspace and trajectory occupancy. Bound memory as support grows and disclose workload-dependent latency.

## Method

Use the Runtime numerical or status invariant and Learnable or checkpoint feature paths. Validate compiled chunked against ordinary search, original against restored actor, including real and reset clocks. Flooding and graph message schedules do not participate. Model/OOM failures must not become a valid action. Keep the supported model learning and release score semantics intact.

## Future work

The parent launches and monitors the measured campaigns. This ticket owns the actor, its performance checks and downstream API handoff.

## Definitions

A pending action is one chosen by the actor whose observed transition has not yet arrived. A batch processes a subset of independent policy strings at a time. An exact restart restores every stored numerical word.


## Notes

**2026-11-24T14:51:44Z**

The campaign actor uses the seed-key scheme already qualified by the acted smoke. Gauss owns the reduction scheduler and archive; the actor will call those hooks. The parent owns remote or long measured runs.

**2026-11-24T15:10:06Z**

Actor and checkpoint contract landed in 27be7fb, followed by the populated full-budget search in e741943. The actor owns one explicit plan key, next proposal, pending command and full model/learner/reducer checkpoint. A real applied action advances the clock once; a reset preserves it. Optional audit_plan hashes the deterministic search result and next key. Empty tracking uses the unchanged zero-score proposal/refit. The untuned populated model shares likelihoods only for byte-identical features and categorical fields within one decision. The 64 MiB cap covers numeric lookup storage; distinct per-policy draws, all 32 steps and all 512×3 trajectories remain. Other temporary arrays, executable memory and Python keys consume additional memory.

From this worktree run JAX_PLATFORMS=cpu JAX_ENABLE_X64=0 uv run --frozen --no-sync python tools/axiom/evaluation_planning_check.py --output /tmp/trace-qualification.json and repeat with --acted --output /tmp/live-qualification.json. Both passed on the integrated production sources. Two to 28 live real rows and a 128-live repeated-prototype stress case gave zero differences in 540672 selected components and motion classes, identical candidates, winner, action and proposal support, and zero max absolute score or proposal drift. Warm 2/12/28/128-row canonical to cached seconds were .431/.039, .615/.138, .743/.146 and 1.469/.219. A separate 12-action development run gave one 3.73e-9 mean novelty difference without a change in ranking. These checks cover sparse, untuned models on CPU. All-live costs and campaign throughput are unmeasured.

Eight real Explode actions (game seed 270037, model seed 87037) succeeded at full budget. Tick-five real disk restart reproduced the full plan SHA b78b9019b48d9a76aad51dcc08d52f8061ab7074a2d85f54251de61f24bfb48f. Empty cold .312s, empty warm about .001s, first populated 2.155s, 13-row warm .122s. First observation 3.52s, later populated updates about .10s. Source hashes and raw traces in ~/agent-share/axiom-online-precision/full-budget-planner-qualification.json (SHA-256 6a1b22dc27484473786300af120c6fa8127df6e623c20f97be2c69c5332cb332). Parent owns the cumulative short suite and sealed campaign; no evaluation seeds used here.
