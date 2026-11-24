---
id: car-8rgl
status: in_progress
deps: []
links: [cff-du6m]
created: 2026-11-23T21:48:38Z
type: task
priority: 2
assignee: coopwilliams
parent: car-zieb
tags: [axiom]
---
# Freeze Gameworld inputs, AXIOM reference cases and scoring

## Background in Plain English — WHY

The paper and public release differ in a few decisions that change behavior. A game score alone cannot tell us whether a difference began in perception, learning or action selection. Freeze the rules and small reference traces before judging the CFFG implementation.

## Outcomes / Acceptance Criteria — WHAT

1. Pin the [arXiv v1 paper](https://arxiv.org/html/2505.24784v1), the [authors’ implementation](https://github.com/VersesTech/axiom/tree/b99f62599e73294639f1944827bf427909f6c698) and [Gameworld v0.1.0](https://github.com/VersesTech/gameworld/tree/85493c477004ecfd3299b0b053884f23d59a5ec5). List configuration, observation size and channel order, frame/action/reward timing, independent model/environment RNG handling and episode reset behavior. Keep the MIT game environment and the separately licensed academic research agent identifiable. Use an authorized reference run where available; record analytical fixtures and unavailable reference data separately.
2. Predeclare the executable-release target and any equation-faithful variants. Cover fixed vs prior-predictive birth thresholds (reuse cff-du6m AC 2–15), paper vs released identity threshold, mixing weights during assignment, slot growth/update gates, full-bank behavior, the recurrent merge score, the sign and scale of novelty, and discounted vs undiscounted action choice. Record the chosen tuned radii and feature set when the paper is ambiguous. Pin the chosen prior over components, per-module capacity and seeded motion rules. Each decision has a pinned source and a fixture that distinguishes alternatives.
3. Supply small replayable frame/action/reward and object-state tapes, independent scalar checks and, where permitted, reference intermediates with numerical tolerances. Cover background/static/moving objects, disappearance, a novel motion, repeated contact, positive and negative merge decisions, and two policies whose ranking reverses with discount. Specify which reference values are exact and which are random estimates.
4. Freeze the ten games and ten new seeds, selection and failure rules, benchmarks and uncertainty/margin rules before scoring. Capture raw per-step reward (the release’s cumulative CSV field describes a shorter window), build Table 1 totals by summing 10,000 rewards and reproduce a hand-computed rolling curve. Make comparable reset, perturbation and ablation protocols replayable. Map each table, figure and supplementary experiment to a fixture, campaign or explicit deferred result, including the rollout-budget sweep. Published means cannot be paired by seed with a new campaign.

## Methodological Suggestions — HOW

Build this as an experiment harness with tiny deterministic probes first. Store fixtures with original source, seed, shapes, tolerance and content hash. For pixel tests play recorded actions into Gameworld; for learner tests inject recorded slots. Privileged labels stay in the scorer. Architecture review covers serialization of fixtures and replay integrity; graph schedules and message rules are N/A.

## Future work

Full campaign execution belongs to car-zieb. Authorization to run or redistribute the authors’ agent is distinct from implementing CFFG independently. If a reference path is unavailable, scope parity claims to the evidence obtained.

## Definitions

A reference tape is a sequence of inputs held fixed for both implementations. An ablation reruns a system with one mechanism disabled. A practical margin is the largest score difference the study will call close enough.

## Notes

**2026-11-24T01:37:58Z**

First reference checkpoint, pinned release b99f625. Does a soft assignment simply accumulate observations? With mixture prior (0.1, 0.1), give each component two hard observations (x=1, category A). Both counts are 2.1. Feed x=3, category B, with equal responsibility, growth disabled. Release stores counts (1.6,1.6); the additive conjugate learner should return (2.6,2.6). For either component the release Gaussian mass/first/second sums become (1.5,2.5,5.5); additive gives (2.5,3.5,6.5). Category excess counts are (1,0.5) vs (2,0.5). Both pass the write gate. The released M-step weights the old excess statistics by the current responsibility when the winning component previously had alpha above one. Preserve both as separately named oracle cases. Source: pinned axiom/vi/models/hybrid_mixture_model.py lines 109–147 and 229–238. High confidence from a direct two-component development probe. Gameworld fixtures and full protocol still pending.

**2026-11-24T01:37:58Z**

Independent additive oracle. Identical active 2D Gaussian priors m=(1,-1), kappa=2, nu=4, psi=((2,1),(1,2)); category priors A=(2,3), B=(1,2,1). Record one: x=(2,0), zero observation covariance, A=0, B missing. Absolute mixed expected log likelihood -4.381065542. Record two: x=(0,0), covariance ((0.5,-0.25),(-0.25,0.5)), A=1, B=1; score -9.047732209. Fixed component weights (1/4,3/4,0) then (1/2,1/2,0) give first and second posterior mean (10,-8)/11 and (14,-8)/13, scales ((262,61),(61,246))/88 and ((362,107),(107,314))/104. Dormant component remains at prior. Independent intercept-only component-MNIW check agreed within 5.6e-7 in float32 and 1.4e-15 in float64. Use with source precision, mask and split-batch tests when work resumes.

**2026-11-24T03:49:54Z**

Completed the MIT Gameworld development tape and reward scorer at 7fa4f796740e0b365d2e0b1e70882265a843a30c on feature/axiom-gameworld-reference. The publishable evidence is under ~/agent-share/axiom-gameworld-reference. Use tapes/manifest.json for the source hashes, game/action seeds, integer action labels and 30 NPZ files. replay.json has every frame hash and the first observed perturbation differences. score.json has each raw-reward total, rolling means and terminal steps. check.json records the independent checks. The manifest SHA-256 is abca225803b4b79801c7fcf2a5823cc639237f12b707f779dea3f3815f3ed376.

Each of the ten games supplied 48 development decisions in its base, color and shape conditions. All 1,440 decisions and 1,473 native RGB frames replayed exactly in a fresh process, including three episode endings followed by explicit resets. A second seeded capture matched all tape array hashes. The simulator and the random policy have independent streams. The private simulator RNG starts before construction. The policy receives an isolated copy of pre-action pixels and the previous real reward, including on a reset after a terminal action. The first observation has no previous reward. The tape preserves the separate terminal and reset frames. For Gold and Hunt the implemented integer action order is left, right, up, down, stay.

The development perturbations switched after action 8. All twenty perturbed runs matched through that step, first changed the returned frame on action 9, and first changed a decision input on action 10. Explode shape changes physical bucket and bomb dimensions in the pinned game. With identical actions its reward first differed at action 46, yielding +2 over 48 steps versus +1 for the base and color runs. The full reference schedule specifies action 5,000. These measurements cover the early development switch only.

The focused checker recomputed raw returns and every eight-step rolling mean independently. The 48-step tapes have no complete 1,000-step window. A 1,001-reward analytic example verified the two consecutive 1,000-step windows and the total separately. Dirac's independent Explode golden matched both reset and first-step frame hashes. The checker also exercised terminal and truncated resets, caller RNG restoration, policy isolation and rejection of changed archives, digests and duplicated runs. This is isolated experiment tooling. No CFFG engine, learner, dependencies, numerical rules, test selection or pass/fail behavior changed. The full suite was outside this gate.

To reproduce from the named worktree, use the private Python 3.11.15 uv project and pinned MIT checkout. The game commit is 85493c477004ecfd3299b0b053884f23d59a5ec5. The overlay/uv.lock SHA-256 is 035d1f9b20a9d490a55f142c644cd27bb2789d03b0806523c1f85d64de720937.

```sh
root="$HOME/agent-share/axiom-gameworld-reference"
envrun() { uv run --project "$root/overlay" --frozen --no-sync python -B "$@"; }
envrun tools/axiom/gameworld_replay.py capture --checkout "$root/gameworld" --archive "$root/fresh-tapes"
envrun tools/axiom/gameworld_replay.py replay --checkout "$root/gameworld" --archive "$root/fresh-tapes" > "$root/fresh-replay.json"
envrun tools/axiom/gameworld_replay.py score --archive "$root/fresh-tapes" > "$root/fresh-score.json"
envrun tools/axiom/gameworld_replay_check.py --checkout "$root/gameworld" --archive "$root/fresh-tapes" --replay "$root/fresh-replay.json" --score "$root/fresh-score.json"
```

Pending at the parent online-learner accuracy stop: object-state and learner/planner intermediate fixtures, remaining paper/release decisions and tests, the paper result/ablation inventory and the predeclared evaluation seed, margin and failure protocol. The historical agent and held-out campaign were not run. Keep car-8rgl in progress.
