---
id: chr-d760
status: closed
deps: [cff-mjku]
links: []
created: 2026-11-25T06:08:03Z
type: task
priority: 2
assignee: coopwilliams
parent: cff-pyy9
---
# Precommit a bounded policy for two future observations

## Background in Plain English — WHY

An agent that buys a cue can commit in advance to one response for each possible reading. The current rollout exposes action marginals and can replan after a reading. A precommitted policy must select its common first move before seeing the cue and execute its later choices without another solve.

## Outcomes / Acceptance Criteria — WHAT

1. A public builder expands one or two observation stages into an ordinary serializable CFFG with a common policy and a named action distribution at each history.
2. The native fixed-policy Bethe ledger gives the sum of the local negative log preferences weighted by their causal visitation probabilities.
3. A returned commitment executes every reachable history without inference and excludes impossible histories.
4. Independent depth-one and depth-two enumeration checks the scores and conditional marginals at float64 and float32 precision. Zero support, vacuous evidence and at least tenfold branch-entry scale are included.
5. A paid-cue commitment chooses compatible actions on both readings and reaches the analytic contingent payoff. A fixed sequence on the same task reaches the open-loop payoff.
6. JSON and CFFGZ reconstruction preserve named policy decisions. Over-depth and over-budget requests fail before expanding the candidate table.
7. Report native factor counts, compile and warm solve times and arena storage by depth.

## Methodological Suggestions — HOW

Use one categorical action prefix per observation history. Each child copies the chosen prefix and appends a uniform action. Expose each clamp with observe_interior and distribute earlier actions using balanced equality fan-out. Give each history a preference observation and a conditional action readout with a distinct absent state. A uniform policy prior permits a native MAP readout and a fixed-policy score with a known prior offset. Freeze enumeration tolerances before inspecting results: absolute probability and energy errors within 256 epsilon times the number of history nodes, with relative energy tolerance 256 epsilon. Exercise forwards and restarts through public bundle and Engine APIs.

Change-path index: Runtime numerical or status invariant uses existing categorical factors, exact two-pass Engine and native BFE/status. Flooding, colored, mixed, host/JIT support and single/vmap operators have no kernel edits. Learnable or checkpoint feature uses standard template/blob reconstruction. No added IR field, learning record, kernel, schedule or legacy migration.

## Future work

The parent epic owns hierarchical execution, shared documentation, dashboard and the cumulative suite. Deeper policy trees are outside the supported domain.

## Definitions

A history is the sequence of future observations available when choosing an action. A commitment maps every reachable history to its action. A branch mass is the probability of reaching a history under the externally chosen actions. Bethe free energy (BFE) is the native ledger score; on a tree its value is minus the log evidence.


## Notes

**2026-11-25T06:08:17Z**

Implementation handoff: charged likelihood at a reached history equals its preference raised to its reach probability. The absent-action category permits reading causal visitation on the selected commitment. Benchmark and admission include every expanded history.

**2026-11-25T06:11:28Z**

Design refinement: compile one action-prefix state per observation history and enforce earlier shared actions with copy-and-append transitions. This factors the complete-policy Gibbs distribution exactly. A normalized action/absent readout integrates out of MAP selection. Use native max-product on the reduced tree and native sum-product on the complete tree for scores and distributions. The largest hidden cardinality is the product of the stage action counts, independent of the number of observation histories. Open-loop construction ties histories at the same decision time. Count all dense source tables before allocating them.

**2026-11-25T06:29:57Z**

Pre-timing protocol: use one fresh CPU float32 process per depth with the 4-action, 7-reading paid-cue model (utility 4, cost 1, accuracy .9). Time host template construction, native preparation, first full synchronous solve and three warm solves. Verify all repeats equal the first policy. Count factors, scalar table entries over instances, source blob bytes, actual CFFGZ bytes, distinct live State buffer bytes at public Engine run/bind boundaries, and process high-water RSS including compilation. The buffer observation runs separately from the timers and excludes XLA temporaries. A deeper request or one exceeding 256 histories/1,000,000 entries is refused before generated array allocation. Both precision/domain refusal and the cancellation-free matched local cost have regressions. Parent owns the shared final suite.

**2026-11-25T06:33:35Z**

Independent CPU resource results: ~/agent-share/cffg-contingent-profile.gcvWXM/depth-{1,2}.json. Rerun tools/hierarchy/benchmark_contingent.py with JAX_ENABLE_X64=0 JAX_PLATFORMS=cpu and a fresh --output for each --depth. Both full commitments chose the cue. 8/57 histories, 47/341 factors, 1264/73392 scalar CPT entries; 6953/37907 compressed bundle bytes. Build 0.008/0.347 seconds, prepare 0.157/0.290 seconds, first full solve 3.645/24.416 seconds and warm medians 3.00/14.57 ms. Public-boundary state high-water 40368/1395120 bytes, process RSS 629358592/1637433344 bytes including JIT. Tests of the smaller oracle enumerate all policies independently; 57 histories use 64-state prefixes instead of a complete 4^57 policy axis. Timed source digests are in both reports.

**2026-11-25T06:34:51Z**

Final validation: uv run --frozen --no-sync pytest -q tests/test_contingent_policy.py passed 15 cases, including native f32/f64 enumerations, sparse and impossible support, tied actions, CFFGZ reconstruction, rare probabilities, budget-before-allocation, altered prior and mass rejection, and sub-ulp scalar loss. An ordinary HMM compiled, constructed, initialized and converged in a fresh Python process with an import hook forbidding every contingent module. The opt-in planner adds no kernel or eager executable to this ordinary path. D17/S4/README contain the derivation, API, measured costs and domain. Parent owns the cumulative suite/dashboard.
