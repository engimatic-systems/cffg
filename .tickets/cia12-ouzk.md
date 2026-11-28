---
id: cia12-ouzk
status: in_progress
deps: []
links: []
created: 2026-11-28T03:08:59Z
type: chore
priority: 1
assignee: coopwilliams
tags: [integration, axiom]
---
# Integrate the AXIOM PR stack with the hierarchical inference runtime

## Background in Plain English — WHY
AXIOM pull requests 198 through 204 grew on an older revision. The current main branch also contains hierarchical planning and improved Gaussian readouts. Cooper asked to merge this stack before processing the ready queue.

## Outcomes / Acceptance Criteria — WHAT
1. Main includes the seven reviewed branch tips, including the Gameworld reference fork.
2. The Gaussian readout serves selected edges, interior point observations and centered learning statistics. The combined test collection retains both branches’ numerical checks.
3. The final integrated code passes the repository short suite and records the actual time. GitHub and the local commit graph confirm which PRs landed.

## Methodological Suggestions — HOW
Integrate from the published main tip. The architecture change paths concern numerical readouts and learnable statistics. Check scalar and grouped Gaussian reads, Engine and Colony forwarding, exact point readout and centered sufficient statistics. Observation masks and test selection have both changed on the two branches. The merge adds no new schema, planning algorithm or stochastic update. Exclude the unpublished navigation and Studio branches.

## Future work
The full AXIOM benchmark remains deferred per Cooper’s previous decision. Independent backlog work follows this merge.

## Definitions
A branch tip is the last commit on a branch. A centered statistic summarizes variation around a mean without subtracting two large raw moments.
