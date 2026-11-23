---
id: cff-2q3i
status: closed
deps: []
links: []
created: 2026-09-25T09:55:24Z
type: chore
priority: 2
assignee: coopwilliams
parent: cff-u9un
tags: [testing, runtime]
---
# Keep hybrid planning validation within the short-suite budget

## Background in Plain English

The final passing hybrid-integration suite took 1,226.516 seconds, above the 1,200-second budget. The new initialization, rank, sparse-support and float32 checks must remain in the default gate. Several full serialization and standalone-example replays repeat model mathematics already covered by direct tests.

## Outcomes / Acceptance Criteria

1. Move only unchanged repeated serialization/example replays to explicit slow selection, preserving every assertion.
2. Keep all new hybrid numerical checks and existing independent mathematical, sparse-support, float32 and uncomfortable-scale regressions in the default suite.
3. Record measured per-test savings and an estimated remaining wall time below 1,200 seconds.
4. Validate changed-file collection, dependencies and whitespace; reuse the passing 2,565-test result under the marker-only budget exception.

## Methodological Suggestions

Another sound selection is acceptable. This affects test selection and uses the explicit AGENTS.md budget exception, not an isolated-tooling exemption. Do not change test mathematics, assertions, fixtures, or production behavior.

## Future work

- cff-u9un records and publishes this adjustment in a separate commit beside its integration result.

## Definitions

The short suite is the default not-slow pytest selection. An opt-in slow test runs when explicitly selected; its assertions are not deleted.


## Notes

**2026-09-25T09:56:03Z**

Measured final gate: 2,565 passed, 96 deselected; pytest 1,211.08 seconds, dashboard wall 1,226.516 seconds. Moved unchanged replays: nonlinear example 4.759s; linear example 4.487s; hybrid example 3.588s; discrete-rollout example 2.927s; neural bundle/schedule replay 7.359s; nonlinear JSON/bundle/edit replay 5.636s; Kalman serialization replay 4.872s. Total 33.628s; estimated remaining 1,192.888s. All new initialization and rank checks, existing independent numerical oracles, sparse/f32/scale tests, all static example-entrypoint checks and three standalone examples remain default. Preserved every assertion. Changed-file collection 33 selected / 8 deselected; git diff --check and tk dep cycle clean. Reused passing suite under explicit marker-only exception; no new full-suite run claimed. Complete.
