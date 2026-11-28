---
id: ccrp-x8yx
status: closed
deps: [cns-t0b9]
links: []
created: 2026-11-25T19:04:56Z
type: task
priority: 2
assignee: coopwilliams
parent: cff-cynx
---
# Test learned movement sequences on aliased navigation routes

## Background in Plain English

In the same-exits navigation control, two junctions and their immediate neighbors share labels. The appearance model and the model that split only the junction both reached zero of six goals. The observed tile map reached all six. A short sequence of observations and achieved movements may distinguish visits that look alike on a single step. RGM Algorithm 1 motivates grouping repeated sequences into events.

## Outcomes / Acceptance Criteria

1. Freeze how candidate sequences are proposed from exploration data, which observations they use, their duration and the evaluation episodes.
2. Compare a one-step place model, a history-conditioned candidate and the same-information tile-map planner. Keep exact local movement available to every arm.
3. Score each frozen candidate on identical new traversals using both next-observation and multi-step prediction. Report closed-loop route success and actions separately.
4. Include matching immediate exits, successful versus blocked moves, and unseen layouts. Report coverage of novel sequences, dictionary size, compile time and memory.
5. Use an independent small oracle and distinguish uncertainty about a place from uncertainty about the agent's movement.

## Methodological Suggestions

Start with two-step sequences. Use only information available before the decision. Hold the path length, fit data and navigation budget fixed while testing the effect of a richer place representation. Architecture change path for a bounded experiment is the public model and inference interface. A production structure edit or checkpoint migration needs a separate review of the duplicated execution surfaces.

## Future work

The general operator and serialized proposal interface belong to cff-cynx. Consider sparse storage after measuring the cost of a larger learned model.

## Definitions

A history-conditioned place assignment can depend on recent observations or applied movements. An RGM path is a latent transition pattern. A route is the agent's executed sequence of moves.


## Notes

**2026-11-25T19:05:03Z**

Prerequisite --> dependent. [closed] cff-du6m --> [pilot] cns-t0b9 --> [next] ccrp-x8yx. Parent cff-cynx owns general proposal operators. The immediate follow-up can use a controlled fixture and the already available engine.

**2026-11-28T13:52:33Z**

Bounded experiment on top of the published navigation pilot. Public model/inference interface: build a conditional categorical tree via the existing navigation evidence helper and use its native exact BFE. Movement encoding, history dictionaries, held-out generation, cost-to-go and replay records are experiment-owned. Message rules, duplicated JIT/host kernels, serialization schemas, checkpoint migration and production structural operators are N/A; numerical semantics do not change. The pilot and its archive stay intact. Freeze the concrete evaluation and route protocol before generating any held-out tape. Invalid/missing probability mass is rejected; both successful moves and literal blocked displacements are retained. Uniform prediction covers unknown rows. Native scoring and reference comparisons use float64 and dtype-relative tolerances; focused tests include f32 and exact movement reconstruction.

**2026-11-28T13:58:37Z**

Protocol froze at 5f1e370. Fit receipt 2640a7d316164b19e3dc4e476807a48345b99ff4cf42c2dbfe9ab3e0ca6ab7d0 binds source 33cac39, the runtime tree and 4,128 training transitions from 14 episodes. The two held-out layouts and command tapes were generated after this seal. The hand-enumerated synthetic oracle distinguished two visits with identical current symbol and command at horizons 1 and 3. Independently replayed the closed archive: 7,020 native arm scores, 54 routes, 920 actual actions; largest absolute discrepancy from the raw-tape NLL reconstruction 0. H3 on the two unseen layouts: history 1.85744, appearance 1.84176, tile map 0.76130 nats (764 windows). One-step endpoint losses were also higher with history on both unseen layouts. Both learned models reached 2/12 unseen goals; the fit-map BFS reached 8/12. There were 19 appearance states, 88 retained histories and 24 surveyed tiles; storage 15,915 / 62,086 / 5,246 bytes respectively. Cold template+Engine build 25.9 ms, first native event including lazy JIT 406.9 ms, process peak RSS 291,897,344 bytes. These clocks describe one development run. Actual gathered movement is exact; ambiguity is among places compatible with observed histories. Archive: ~/agent-share/navigation-histories; no evaluated model or score edits after the fit seal.

**2026-11-28T15:49:51Z**

The plain-language guide at http://localhost:8080/navigation-histories/index.html replayed sealed forecasts and actual failures at desktop/mobile widths. Joint source 132eb3e passed all 2,806 short cases; 510 opt-in, 4 warnings, 1,187.11 seconds pytest and 1,202.604 seconds dashboard wall. The 2.604-second overrun triggered the selection-only duplicate float64 mixture-rank change already qualified in d71952b. Its measured case duration was 2.755 seconds. The float32-vector0 test retains the full annihilated-direction assertions. Before/after collection confirmed exactly that one float64-vector1 case moved to opt-in, 2,805 short and 511 opt-in; estimated wall 1,199.849 seconds. Every navigation case stays in place. Reused the passing gate per AGENTS. No new simulation or short-suite run.
