---
id: cff-xa6z
status: open
deps: [cff-v6th, cff-04mb, cff-ectw, cff-qu2j, cff-7kpx, cff-toqa, cff-0kyl, cff-4yr6, cff-qtub, cff-3wsw, cff-5k0n, cff-csmz, cff-ees0, cff-wi2o, cff-iplr, cff-nd1q, cff-fl7h, cff-21vm, cff-fj4v, cff-3nr4, cff-wdhj, cff-wt1c, cff-rnn1, cff-98gj, cff-8jqy]
links: [cff-wbhu, cff-mjku, cff-3kdk]
created: 2026-09-25T03:43:40Z
type: epic
priority: 1
assignee: coopwilliams
parent: cff-ahps
tags: [pcgym, control]
---
# PCGym follow-up: reliable nonlinear and uncertainty-aware control

## Background in Plain English

The PCGym experiment exposed reproducible execution bugs, expensive host operations, and missing modeling capabilities after successful newcomer onboarding. Cooper requested all follow-ups as dependency-linked tickets and implementation as a PR chain. The goal is reliable nonlinear and uncertainty-aware control, not a blanket promise to beat PPO or to explain every reward gap.

The original evidence is at `/Users/ravd/agent-share/PCGymCFFG/`; experiment source is at `/Users/ravd/experiments/cffg-testing/pc-gym/00/cffg/`. The evaluated/current starting revision is `fce60bd`, the head of PR #161. Independent audit reproduced no-delta hardening/Colony nonconvergence, disconnected rendering, hybrid divergence with damped recovery, and a float32 Gaussian certificate failure. The exact reported nonlinear RK4 failure still needs a valid retained fixture.

## Outcomes / Acceptance Criteria

1. Every agreed bug, runtime issue and modeling capability has a scoped ticket with explicit prerequisites and an architecture/duplicated-surface handoff.
2. Existing readout and sweep-loop work is reused through cff-3kdk and cff-wbhu; fan-out ergonomics remains owned by cff-mjku.
3. Confirmed correctness fixes land before dependent performance or facade work; nonlinear admissibility, particle rejuvenation and action bounds pass their research/design gates before implementation.
4. Each implementation is an atomic ticket-referenced commit on a reviewable PR chain continuing PR #161, with required validation and linked handoff.
5. The dependency graph is acyclic and accompanied by a compact human-readable tree naming ready roots and blockers.
6. Matched final comparisons distinguish information access, model/controller choices, numerical behavior and runtime; no finite-bank or reward claim exceeds its evidence.

## Methodological Suggestions

Start with no-delta exact dispatch and failure attribution. In parallel independent work can capture numerical fixtures, specify bounds, measure training costs and expose the fixed-bank contract. Follow the architectural change-path row and hostile-input requirements recorded in each child. Keep derivations in DERIVATIONS, capability guidance in README, scope honesty in LIMITATIONS and work state in tickets.

No numerical change is made by this roadmap ticket. Host/JIT, flooding/scheduled, single/vmap and checkpoint pairs are N/A for ticket authoring; each implementation child records its affected pairs. Run `tk dep cycle` and `git diff --check` for this ticket-only PR. The literal dependency records are authoritative; serial PR ancestry does not invent engineering prerequisites.

## Future work

- cff-v6th — Preserve exact Gaussian inference when no delta constraint can harden.
- cff-04mb — Report originating factor and reason for inference failures.
- cff-ectw — Stabilize the reproduced hybrid planning failure without changing its model.
- cff-qu2j — Resolve float32 exact Gaussian-chain certification failures against an independent oracle.
- cff-7kpx — Capture and classify the RK4 nonlinear CSTR planning failure.
- cff-toqa — Implement the evidence-backed nonlinear planning admissibility contract.
- cff-0kyl — Render every disconnected graph component and clarify SVG entry points.
- cff-4yr6 — Batch observation binding while preserving partial DATA updates.
- cff-qtub — Reuse compiled structure when Colony spawns parameter-only variants.
- cff-3wsw — Expose physical-function point training without neural-specific registration folklore.
- cff-5k0n — Measure physical-mean training setup and derivative costs on a retained fixture.
- cff-csmz — Remove measured redundant work from physical-mean training.
- cff-ees0 — Expose exact finite parameter-hypothesis updates as a supported library feature.
- cff-wi2o — Specify valid resampling and rejuvenation for nonlinear parameter particles.
- cff-iplr — Implement the validated nonlinear parameter-particle learning contract.
- cff-nd1q — Derive a bounded continuous-decision contract with an independent planning oracle.
- cff-fl7h — Implement bound-aware continuous planning from the approved derivation.
- cff-21vm — Add reproducible joint NIW and MNIW posterior sampling.
- cff-fj4v — Publish continuous and hybrid control recipes covering experiment onboarding gaps.
- cff-3nr4 — Specify serialized continuous and hybrid rolling-agent boundaries.
- cff-wdhj — Implement the continuous and hybrid rolling-agent facade.
- cff-wt1c — Validate the PCGym follow-up chain with matched control and runtime comparisons.
- cff-3kdk — vectorized and selective public readouts.
- cff-wbhu — compiled sweep/controller execution.
- cff-mjku — existing fan-out/observation-branch ergonomics, linked context rather than a new gate on the PCGym comparison.
- cff-ahps — downstream factor-library roadmap.

## Definitions

- **PCGym:** the process-control benchmark used by the experiment.
- **PPO:** proximal policy optimization, the reinforcement-learning comparator.
- **PR chain:** pull requests whose branches build on preceding requests.
- **Research/design gate:** a prerequisite establishing supported mathematics and independent tests before implementation.
- **Finite bank:** a fixed set of parameter hypotheses, not an exact representation of every continuous parameter value.
- **Ablation:** a controlled comparison changing one component.

## Notes

**2026-09-25T03:47:40Z**

Ticket-authoring handoff: 22 child tickets created under cff-xa6z; cff-3kdk extended, cff-wbhu and cff-mjku reused. Base is PR #161, cff-ucvx-readme-capability-map. Implement first cff-v6th, then failure diagnostics; research gates remain real prerequisites.

Legend: prerequisite --> dependent. [R] ready; [B] blocked.
The independent [R] rows can proceed in parallel; all children gate cff-xa6z.

[R] cff-04mb diagnostics
  +--> [B] cff-7kpx nonlinear reproducer --> [B] cff-toqa nonlinear repair
  +--> [B] cff-ectw hybrid solve ---------+
[R] cff-v6th exact/Colony ---------------+--> [B] cff-wdhj controller
  +--> [B] cff-qtub cached spawning      |
[R] cff-3nr4 boundaries ----------------+
[R] cff-fj4v recipes -------------------+
[R] cff-ees0 fixed bank --> [B] cff-wi2o particle design --> [B] cff-iplr particles
[R] cff-nd1q bounds design --> [B] cff-fl7h bounded planning
[R] cff-5k0n profile --> [B] cff-csmz training optimization
[R] cff-qu2j float32; cff-4yr6 binding; cff-21vm sampling
[R] cff-3kdk readouts; cff-wbhu sweep execution (existing)
  all applicable leaves above --> [B] cff-wt1c matched comparisons
[R] cff-0kyl disconnected SVG; cff-3wsw physical training API
  plus every child above --> [B] cff-xa6z --> parent roadmap cff-ahps

cff-wt1c's exact fan-in is cff-toqa, cff-iplr, cff-fl7h, cff-wdhj, cff-csmz, cff-4yr6, cff-qtub, cff-21vm, cff-qu2j, cff-3kdk, cff-wbhu. cff-mjku remains linked fan-out work owned by the existing roadmap, not a new prerequisite of this comparison. The final arrow to cff-ahps denotes the parent roadmap contribution, not a newly added dependency edge.

Built from tk dep tree --full, tk ready and tk blocked; tk dep cycle reports no cycles. No implementation or numerical tests yet.

**2026-09-25T03:48:29Z**

Roadmap PR: https://github.com/engimatic-systems/cffg/pull/162 (base #161). Ticket-only validation passed: tk dep cycle and git diff --check. Next implementation ticket: cff-v6th. The epic remains open until its child outcomes and final matched comparison are complete.

**2026-09-25T04:19:55Z**

Validation follow-up: added cff-rnn1 after the first implementation suite exceeded the short-suite budget. Legend prerequisite --> dependent: [closed] cff-rnn1 (retain exhaustive gradient checks as opt-in slow) --> cff-v6th --> cff-qtub / cff-wdhj. The 22 original capability/fix tickets are unchanged; this is an additional test-budget child. tk dep cycle remains clean.
