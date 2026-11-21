---
id: cfp-rh8k
status: closed
deps: []
links: []
created: 2026-11-21T21:23:01Z
type: task
priority: 2
assignee: coopwilliams
---
# Measure and reduce first-plan latency on the full adaptive reactor

## Background in Plain English — WHY
The first reactor plan in PR 185 took 26.48 seconds even after repeated plans became fast. Cooper needs to know which use cases pay that cost and whether it can be reduced.

## Outcomes / Acceptance Criteria — WHAT
1. Attribute the measured cold plan cost to concrete operations on the unchanged benchmark.
2. Compare remedies in separate processes, distinguishing first-ever use, subsequent launches and warm decisions.
3. Check returned beliefs, energies, residuals, actions and warm performance before recommending a remedy.

## Methodological Suggestions — HOW
Use the full P4/H15 benchmark on the stack tip. Profile the compiler and test executable caching and cold compilation options. Architecture change path is runtime execution if any production change is needed. The measurement uses Engine on CPU and the existing independent certificate. Colony, training, checkpointing and factor rules are N/A for a tooling-only solution.

## Future work
Broader model qualification depends on the results.

## Definitions
A cold plan is the first decision in a process. A persistent compilation cache stores machine code on disk for compatible later runs.


## Notes

**2026-11-21T21:31:37Z**

Question: Is the first reactor decision inherently expensive? The stock PR185 timed boundary reproduced at 26.289 s on the M2 Max. JAX logged 21.282 s in back-end compilation, led by proposal (9.597 s) and post-root gradient diagnostic (5.451 s). Cache replay was 7.958 s. Compiler options alone reached 16.359 s. Source: ~/agent-share/cffg-cold-start/{empty.log,summary.json}. Confidence high for this frozen fixture; these are mostly single fresh-process measurements.

**2026-11-21T21:31:37Z**

The new directed reference solves the same forward-cavity Taylor system in two passes. Seven launches with JAX disk caching disabled: median 0.206 s for solve plus independent node/ledger checks, 0.648 s whole process via uv. All nine matched plans passed the original engine map, raw/canonical successor FE, finiteness, local-status, merit-gradient and positive marginal checks. Largest full root 7.38e-14. A stale plan was rejected. Full engine audit was timed separately. Reproduce from this worktree with uv run --frozen python tools/benchmark_cstr_chain.py; see ~/agent-share/cffg-cold-start/index.html, check_all.py and full-certificates.json. Isolation: new benchmark and derivation only, no runtime, dependency, test selection or reporting change; focused changed-artifact checks instead of short suite. Next: cfp-okq2 covers conservative production dispatch. No general nonlinear or controller reliability claim.
