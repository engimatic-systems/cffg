---
id: cff-bpgm
status: closed
deps: []
links: [cff-cfqg, cff-uy48]
created: 2026-08-10T03:33:00Z
type: task
priority: 1
assignee: coopwilliams
tags: [audit, roadmap, paper-fidelity]
---
# Audit CFFG trajectory against vision and source papers

## Background in Plain English

Cooper needs to decide whether to keep merging the current pull-request stack
and continue the roadmap. He has little time for review and is concerned that
the project may have become hard to maintain, drifted from his original
product vision, or stopped faithfully implementing the two CFFG/LAIF papers.

This ticket records an independent survey of the repository at the head of
`agent/cff-nkbj-reference-blocker`. The survey compares the Obsidian vision,
both source papers, the as-built architecture, the open ticket graph, measured
performance evidence, and pull requests 55 through 62. It must report bad news
as plainly as good news. Reassurance is not an acceptance criterion.

## Outcomes / Acceptance Criteria

1. One phone-readable survey artifact compares the current repository with
   every numbered item in the Obsidian vision and the papers' central
   mathematical and behavioral claims.
2. The survey distinguishes exact implementations, declared approximations,
   project extensions, incomplete integrations, and unsupported claims. Each
   material conclusion cites repository or source evidence and states its
   confidence.
3. The survey assesses maintainability using concrete change paths, duplicated
   execution surfaces, module boundaries, public APIs, and test authorities.
4. The survey reports measured operational envelopes for compilation,
   inference, readout, planning, batching, learning, memory, and resource
   estimation without combining incomparable benchmark contracts.
5. The open ticket portfolio is reduced to a proposed near-term sequence. The
   recommendation names work to continue, reorder, park, re-scope, or add.
6. Pull requests 55 through 62 receive a dependency-ordered review guide that
   identifies merge blockers and the smallest useful human review surface.
7. The artifact is stored under `notes/` and linked into `~/agent-share/` for
   phone access.
8. Ticket and documentation-only validation includes a final wiggle-word scan,
   `tk dep cycle`, and `git diff --check`. The default test suite is not rerun
   because the survey changes no runtime code and the current code head already
   has a recorded passing short-suite result.

## Methodological Suggestions

These suggestions are optional. Another method is acceptable if it satisfies
the numbered outcomes.

- Read the vision and both papers before surveying the implementation.
- Split repository review among GPT-5.6-sol sub-agents using the change-path
  index in `docs/ARCHITECTURE.md`.
- Cross-check each sub-agent conclusion against primary repository evidence.
- Prefer one organized artifact over separate reports that repeat the same
  findings.
- Treat tickets as plans and tests or preserved external artifacts as evidence
  of implemented behavior.

## Future work

The survey may recommend new tickets or changes to existing tickets. It does
not implement those changes. Cooper's review of the survey determines which
recommendations become roadmap commitments.

## Definitions

- **Course correction:** A change in roadmap order, claim scope, or technical
  direction recommended by the evidence.
- **Paper fidelity:** Agreement between the declared CFFG model, constraints,
  message updates, free-energy ledger, and behavioral protocol and the two
  source papers. Similar end results alone do not establish paper fidelity.
- **Operational envelope:** The measured structures, sizes, hardware, latency,
  memory, and convergence conditions under which a capability is dependable.
- **Pull-request stack:** Pull requests whose base branch is the preceding pull
  request's head branch and which therefore must be reviewed and merged in
  dependency order.
- **Vision:** Cooper's project brief at
  `/Users/ravd/Documents/01-obsidian-vaults/agents/CFFG/0.0.md`.

## Notes

**2026-08-10T03:39:35Z**

Survey handoff and roadmap companion.

Legend: prerequisite --> dependent

[ ] cff-akmp (ready) --> [ ] cff-64ri --> [ ] cff-0l31 --> [ ] cff-oyie --> [ ] cff-nkbj
[x] cff-l7y4 -----------^              ^
[x] cff-garu --------------------------+
[x] cff-r6bl --------------------------+

Parallel selected ready roots:
[ ] cff-cj5h
[ ] cff-j7mm
[ ] cff-3kdk
[ ] cff-209l
[ ] cff-wbhu --> [ ] cff-5edo (immediate blocker: cff-wbhu)
[ ] new paper-faithfulness gate

Parked downstream roadmap work includes cff-0f3g, cff-o7hs, cff-1o72, cff-cynx, and cff-pyy9. The tk dependency records remain authoritative.

Retained wiggle-scan candidates and intended meanings: integrate means the implemented-to-public boundary classification; preserve means a retained external artifact; public means a documented or exported API; same means duplicate survey content; implement means repository code or named future work according to the sentence; approximate means a declared mathematical approximation; near means the next roadmap horizon; check means an automated command or a manual cross-check as named; converge means a runtime status or an operational convergence condition; estimate means the resource tool; evidence means a named source, test, experiment, formal argument, or provenance artifact; exact means no mathematical approximation for the named family; pass means recorded suite success; test means an automated artifact; validate means ticket or documentation acceptance validation; infer and learn are the surveyed runtime and parameter-workflow categories; performance means the named cold, warm, memory, and throughput phases; small means the minimum useful review surface; may states possibility or permission; every is bounded by the enumerated vision items; only is the documentation-only scope; current is pinned to the surveyed branch and head; existing means the named present tickets; read is a future requirement. Capability ownership was checked separately: cff-bpgm owns only the survey and phone-readable handoff. Cooper decides which recommended code or roadmap changes become commitments.

**2026-08-10T03:39:53Z**

Final scan additions after the roadmap companion: ready means dependency-ready according to tk; remaining means the selected open backlog after parking optional work; intended means the contextual meanings assigned in the prior scan note; explicit_owner means cff-bpgm ownership of the survey artifact; named_ticket_handoff means the concrete ticket IDs in the proposed roadmap. The 30 retained candidates now have a stated contextual meaning.

**2026-08-10T03:40:54Z**

Completed at commit 08c7a4a. The survey is notes/survey-2026-08-09/index.html. HTML parsing and internal-anchor checks passed. The final wiggle scan retained 30 contextually classified candidates and capability ownership was checked separately. tk dep cycle and git diff --check passed. The default short suite was not rerun because this ticket changes only ticket and survey documentation; the unchanged code head has a recorded 1,336-test passing short-suite result at ca295f9.

**2026-08-10T03:41:14Z**

Post-closure scan additions: complete means completion of this ticket scope at the cited commit; close means administrative ticket closure after the named validation passed. The prior scan notes classify the other 30 retained candidates.

**2026-08-13T04:40:34Z**

Course-correction follow-up, 2026-08-12: cff-uy48 now owns the separate paper-faithfulness and minimum-usable-release gate recommended by the survey. cff-cfqg remains the parallel engine-trustworthiness gate. cff-ahps now waits for both and closes on a finite extensible-kernel integration baseline.
