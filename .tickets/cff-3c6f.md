---
id: cff-3c6f
status: open
deps: [cff-ucvx]
links: []
created: 2026-09-22T19:16:00Z
type: epic
priority: 1
assignee: coopwilliams
---
# Epic: Make CFFG self-describing for experiment newcomers

## Background in Plain English — WHY

CFFG has a tested newcomer quickstart, but the rest of the usable library is hard to discover. The README names only part of the public facade. Mature model builders and experiment capabilities require deep import knowledge. Large model-specific walkthroughs obscure the high-level explanation. The limitations document also carries stale status text and one misleading moment-matching statement.

This epic makes the current library self-describing for an experiment agent. It does not add inference mathematics. It aligns public imports, runnable examples, limitations, installation guidance, and one concise capability-led README.

## Outcomes / Acceptance Criteria — WHAT

1. Child tickets define and implement the supported public import layers, runnable examples, current limitations, and the final README.
2. The final README contains one clear capability table covering every importable experiment feature and links each row to its explanation, API entry point, example when applicable, objective meaning, and main limitation.
3. Model-specific end-to-end programs live under examples/ instead of interrupting the README narrative.
4. A new experiment project can add the local CFFG checkout with one complete uv command and verify the import.
5. Canonical documentation and executable public-API tests agree at the proposed PR head.
6. The implementation lands as a linear PR chain above cff-1o72 so each review layer has one coherent purpose.

## Methodological Suggestions — HOW

These suggestions are optional. Another method is acceptable if it meets the outcomes.

- Land public-import changes first, examples second, limitations third, and the final README last.
- Preserve existing imports unless a break is unavoidable and explicitly reviewed.
- Treat tickets as historical evidence and current code/tests as the source for capability claims.
- Use the documentation-only validation exception where applicable. Run the normal short-suite gate for any layer that changes tested import or test-execution behavior.
- Applicable architecture change path: no exact runtime row applies. Each child names its public-documentation or facade authority and marks host/JIT, flooding/scheduled, single/vmap, and current/legacy pairs N/A unless it changes them.

## Future work

- Package publication and a real vendoring or release process are deferred. No ticket is created for that work in this epic.
- Stable pre-1.0-to-1.0 compatibility remains owned by cff-1xig for CFFGZ and by future release planning for Python imports.
- Mathematical features and performance gaps remain in their existing tickets.

## Definitions

- Capability table: A compact map from a supported task to its import path, status, outputs, objective meaning, example, and main limitation.
- Facade: A package import surface that presents supported names without requiring callers to know internal modules.
- PR chain: A sequence of pull requests where each branch is based on the preceding branch.
- Newcomer: An experiment-writing agent or engineer encountering CFFG for the first time.
- uv: The required Python package and environment manager for this repository.


## Notes

**2026-09-22T19:18:39Z**

Roadmap dependency snapshot, 2026-09-22. Built from tk dep tree --full, tk ready, and tk blocked.

Legend: prerequisite --> dependent

[cff-odag] facade (ready)
      |
      +--> [cff-rtuf] standalone examples
                |
                +--> [cff-rbsy] limitations reconciliation
                          |
                          +--> [cff-ucvx] README and uv install
                                    |
                                    +--> [cff-3c6f] newcomer epic

Ready root: cff-odag. Immediate blocker for every later layer: the preceding ticket in this linear review chain. Parallel work: none by design, because the final README must describe the exact facade, examples, and limitations already present below it. Downstream result: cff-3c6f closes only after the complete newcomer guide lands. tk dependency records remain authoritative. tk dep cycle reports no cycles.
