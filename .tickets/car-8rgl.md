---
id: car-8rgl
status: open
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
