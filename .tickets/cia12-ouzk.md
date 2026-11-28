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

## Notes

**2026-11-28T03:45:11Z**

Integrated 198–204 at a58d91c with both parents and tested the final code at 04bc4c5. Grouped/selected Gaussian readouts and centered tap statistics agree with independent NIW/MNIW posteriors. A direct or equality-linked observation of a filter’s carried posterior now fails during validation. The standalone and bundle guards passed, including a valid noisy carry and both precisions. The full dashboard command passed 2,837 tests with 479 opt-in cases deselected. The inner wall was 1,250.052 seconds and outer wall 1,250.18 seconds. CPU was an Apple M2 Max. Timing evidence and all per-case durations are in ~/agent-share/axiom-integration-4RQ0bP/short-suite.log. All seven frozen PR heads are ancestors. Ready to publish after the short-suite budget adjustment.

**2026-11-28T03:53:59Z**

Short-suite budget handoff. Of the passing 2,837 selected cases, 31 replicated variants totaling 52.23 measured call seconds now run in the opt-in tier. Collection verifies the same 3,316 unique test IDs and 2,806 short plus 510 opt-in. Under the short-suite budget rule the passing result is reused; 1,197.822 seconds is an estimate, with 2.178 seconds of margin. The source and integration regressions are unchanged. The per-ID receipts and matched coverage are in ~/agent-share/axiom-integration-4RQ0bP/selection.json. cia12-ibdi owns the larger wall-time reduction.
