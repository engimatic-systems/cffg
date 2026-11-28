---
id: car-zieb
status: open
deps: [car-kwjj, car-mps6, car-ne0a, cff-wdhj, cao-36dd, cao-lxmq]
links: [cff-xa6z, cff-cynx]
created: 2026-11-23T21:48:38Z
type: epic
priority: 2
assignee: coopwilliams
tags: [axiom, roadmap]
---
# Reproduce AXIOM from pixels to actions with supported CFFG features

## Background in Plain English — WHY

AXIOM learns from game pixels, the actions it actually took and the rewards that followed. It discovers objects and motion rules, uses them to imagine possible futures, then acts. We want to learn whether the same design works when CFFG owns the beliefs, updates and conditional predictions. Matching a return while quietly replacing one of those operations would leave the library question unanswered.

## Outcomes / Acceptance Criteria — WHAT

1. Run one complete game for 10,000 real steps as the integration pilot. Show a frame becoming object beliefs, an observed transition updating the world model, a proposed or accepted reduction, and imagined outcomes changing the action ranking. Use targeted fixtures to exercise rare branches. Check continuous execution against a resumed run, including a game reset, random draws and the next real action.
2. Use the supported rolling-agent boundary from cff-wdhj to connect the six deliverables below. Actual observations and executed actions enter learning once. Planning reads a snapshot. Game labels, true object coordinates and future outcomes are available only to evaluation. Execute the model through CFFG, independently of the authors’ solver.
3. Run the main untuned experiment on all ten Gameworld games, ten prespecified model seeds per game, 10,000 online steps per run. Retain learning across episode resets within a run and start fresh between runs. Report cumulative raw reward per seed, per-game mean and standard deviation, 1,000-step learning curves, failures, hardware and elapsed time. One arm is one million real interactions. Keep smoke results separate.
4. Test the paper’s explanatory claims separately: disable periodic merging of interaction rules in one run and disable the exploration bonus in another; run the tuned interaction-distance variant; introduce the specified visual perturbations at step 5,000 with matched unperturbed controls and show recovery. Identify privileged remapping wherever used. Archive representative slot, motion and interaction traces, the number of active parameters and update/planning timings.
5. Publish a claim table with the frozen protocol, raw results, uncertainty, reference value and verdict for each claim. Distinguish an implementation that ran, module parity and statistical agreement. Apply predeclared per-game practical margins. Compare against the published baseline aggregates as unpaired references; mark independently rerun baselines and any unattempted supplemental claims explicitly.

## Methodological Suggestions — HOW

Requirement ownership and roadmap position

| Deliverable | Owner | Existing work reused |
| --- | --- | --- |
| Reference fixtures, benchmark conditions and score | car-8rgl | Gameworld v0.1.0 and the paper’s Table 1 |
| One assignment updates continuous and categorical beliefs | car-9ccf | cff-lquu hybrid moments, cff-qg1d hybrid learning and cff-6tek Gaussian families |
| Maintain active components, preserve their identity, pool and reset their evidence | car-k6oi | cff-enhs / cff-jilz bundle and exact-resume foundations; cff-du6m AC 2–15 for the paper’s growth/reduction math |
| Pixels to object tracks and types (sMM/iMM) | car-kwjj | cff-cpxj plates; the weighted learner and component bank |
| Motion and interaction/reward prediction (tMM/rMM), including periodic pruning | car-mps6 | cff-pksr switching Gaussian factors; the same bank and learner |
| Simulate possible futures and select the first action | car-ne0a | cff-lquu one-step prediction; compare against exact enumerable worlds |
| Consume real experience once and preserve the live model across game resets | cff-3nr4 → cff-wdhj, then integration here | cff-enhs / cff-jilz persistence, extended by the bank and object tickets |

The existing structure-learning track cff-du6m → cff-cynx studies edits to an arbitrary graph. AXIOM mainly fills, combines and reuses rows in a model with predetermined capacity. The paper research is shared; general graph search can proceed on its own schedule. The PCGym control roadmap cff-xa6z and the contingent planner cff-pyy9 are parallel investigations. Start component-level pilots on small tapes while the public rolling interface is in development.

Use the “Learnable or checkpoint feature” and “New factor or rule family” change paths for integrated library behavior. Compare single/batched prediction, resumed/uninterrupted execution and host/compiled numerical status. An external reference runner consumes the same fixtures independently.

## Future work

Fresh BBF/Dreamer training, extended 100,000-step comparisons in later revisions and general-purpose architecture search have separate scopes. Record them as unreplicated when applicable. A claim of reproducing the whole paper must list every included and outstanding experiment.

## Definitions

A slot is a reusable hypothesis about one visible object. A component is a learned alternative such as a type or motion rule. A rollout is an imagined sequence of future actions and outcomes. A seed fixes the random initialization and draws for one run.

## Notes

**2026-11-23T21:51:25Z**

Roadmap handoff. The ticket graph was checked against tk dep tree --full, tk ready and tk blocked. All six new features/tasks are open. Closed foundations are omitted below.

```text
prerequisite --> dependent                         [R] ready now

[R] car-9ccf mixed evidence --> car-k6oi grow/merge components

    car-k6oi + [R] car-8rgl reference fixtures
        +--> car-kwjj pixels and identities ----+
        +--> car-mps6 motion and interactions --+--> car-zieb AXIOM

[R] car-ne0a sampled planner ---------------------> car-zieb
[R] cff-3nr4 rolling boundary --> cff-wdhj facade --> car-zieb
```

For the first library proof, give one ambiguous observation to two object types, verify every learned count, then grow and recycle a component without losing either identity. This unblocks both object-model branches. In parallel, build the reference traces and planner against an enumerable toy world. Each object branch has an injected-input fixture so neither waits for the other. The facade is an integration dependency because this epic promises a supported CFFG agent. A private paper prototype could wire its own observation loop.

Research record. Which existing tickets actually block the port? The hybrid conjugate/readout stack already supplies weighted moments; cff-3nr4 and cff-wdhj own public state carry. See their acceptance criteria and the capability map above. High confidence for the ownership mapping. Which structure learning is shared? cff-du6m AC 2–15 describe paper growth and merge. The released recurrent merge tests a limited reward/switch score on generated data and the released planner evaluates sampled action strings. See the [reduction implementation](https://github.com/VersesTech/axiom/blob/b99f62599e73294639f1944827bf427909f6c698/axiom/models/rmm.py#L839-L920) and [planner implementation](https://github.com/VersesTech/axiom/blob/b99f62599e73294639f1944827bf427909f6c698/axiom/planner.py#L127-L178). High confidence for these code paths; behavioral agreement and full-scale performance remain empirical.

**2026-11-24T01:37:58Z**

Work paused on the first accuracy checkpoint. car-9ccf now depends on cff-7t4j. The current float32 mixture readout can encode distinct true within-component spreads with identical public statistics. Reference work established separate additive and released-history update fixtures. No Gameworld outcomes have been measured.

**2026-11-24T03:34:42Z**

The centered mixture uncertainty repair passed. Implementation paused at Cooper’s performance checkpoint during the next learning milestone. Corpus, public readout, posterior and host-fold measurements are linked in the served ~/agent-share/axiom-centered-checkpoint report. Both API options are available for further testing. The one-game pilot remains downstream.

**2026-11-24T03:53:38Z**

Next accuracy checkpoint is cff-xfnx --> car-9ccf --> downstream component lifecycle/model work (arrow from prerequisite to dependent). At the new checkpoint, storing a small float32 change at each tick lost real learning; one batch and float64 recovered the analytic answer. A separate Gameworld tape capture/replay milestone is available in PR #201. End-to-end learning and game reward remain unmeasured.

**2026-11-24T06:04:21Z**

Road position after the precision gate. A supported typed agent restart now has explicit ownership in cao-lxmq. Existing categorical/one-Gaussian restart cannot restore AXIOM tracks, component banks, policy proposal or accumulated reduction samples. The fixed-action and small acting smoke can proceed with experiment-owned state. The full public-facade reproduction remains downstream. Legend: prerequisites --> dependents. Mixed learning + fixed-bank operations --> scene learning and rMM reduction; scene + sampled planner --> short acting smoke; scene + planner + reduction + rolling boundary --> cao-lxmq --> 10k integration, then preregistered million-step car-zieb study. Default compensated learning needs a human speed decision (caoc-0dvd).

**2026-11-24T06:44:11Z**

The public experiment now takes pixels through tracking and learning to actions. On 24 closed-loop Explode steps at a reduced search budget, all 24 actions were applied, the agent received zero reward, and an exact disk restart before action 12 reproduced that action, the plan and the next random key. The reducer also has independent positive/negative merge and 500-step cadence checks. A whole-agent continuation that includes its retained synthetic records still belongs to cao-lxmq. The short game establishes the interfaces and leaves the paper's returns unmeasured.

In the fixed-action Bounce development run, separate native initializations ended at 129 versus 112 interaction rules in float32 and float64. An exact shared start produced 129 each and matched all 1,536 tracking decisions. The authors' released command line defaults to float32 and also supports float64. The dtype of historical paper runs is unconfirmed.

Selected dependencies, prerequisite --> dependent. car-8rgl supplies the frozen protocol. The compute tradeoff is the P1 decision for Cooper.

    caoc-0dvd + cff-xfnx --> car-9ccf --> car-k6oi
    car-k6oi + car-8rgl --> car-kwjj + car-mps6 --> car-zieb
    car-ne0a + cao-fs5s + cff-wdhj --> cao-lxmq --> car-zieb

The reviewed measurements, scripts and raw records are linked from ~/agent-share/axiom-online-precision/index.html. The 10,000-step evaluation is pending.
