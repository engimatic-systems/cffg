---
id: cns-t0b9
status: closed
deps: [cff-du6m]
links: [ccrp-fuob]
created: 2026-11-25T18:24:33Z
type: task
priority: 2
assignee: coopwilliams
parent: cff-cynx
---
# Qualify learned places on a loop navigation test with a replay guide

## Background in Plain English

Two visually identical junctions can lead to different destinations. Cooper wants to see whether learning distinct places improves prediction and navigation using exact local motion.

## Outcomes / Acceptance Criteria

1. Freeze a bounded navigation fixture, proposal policy and separate fit, selection and final evaluation traversals before scoring.
2. Compare a fixed appearance model, a place-splitting candidate, and an observed tile-map navigator on common information. Include a negative control and compare the native ordinary tree energy to an independent predictive calculation.
3. Provide a reproducible, interactive HTML guide showing actual decisions, held-out evidence and return routes. Identify supplied geometry and learned connections.
4. Record the RGM ideas and their limits in the existing research ticket. Distinguish the pilot from the full structure-search module.

## Methodological Suggestions

Use supervised displacement from legal movement and build candidate assignments from the exploration tape. Never give a navigation policy the complete layout. Architecture change path is experiment tooling using the public Engine. No changes to host or JIT rules, schedules, Colony, checkpoints or codecs.

## Future work

The production edit/provenance interface, persistent belief migration and sparse scale engineering belong to later review of cff-cynx.

## Definitions

A place is a repeatable stop on a route. A proposal is an alternative assignment of observed stops to places. A held-out traversal is a reserved sequence used to score a frozen assignment.


## Notes

**2026-11-25T18:25:11Z**

Legend: prerequisite --> dependent. The ready research task cff-du6m --> this controlled navigation pilot. It also feeds cff-cynx, whose other prerequisite cff-izjd is closed. Protocol, fixture construction and report layout may proceed alongside the review. Evaluate the frozen model and claim the result after the evidence assumptions are checked. The full production structure-search API remains with the epic.

**2026-11-25T19:05:17Z**

Completed on source 7c8be735. Open http://localhost:8080/navigation-structure/pilot/index.html for the standalone visual guide. Two 24-tile layouts and their rotations use exact local displacement and a prespecified junction proposal. On the different-exits layout the reserved selection tape favored splitting by 1.243353 nats; the final 102 steps favored splitting by 2.486706 nats. Six of six routes succeeded with the split, zero with pooled appearance, and six with the observed tile map. The matching-exits control favored pooling by 4.046727 nats on selection and 8.093454 nats on final prediction. Both place policies failed all six routes there, while the tile map succeeded in six. Independent reconstruction verified all 72 routes and 1,248 native per-event score entries including rotations. The development and published cases/findings match exactly. Standalone desktop/mobile Chromium passed with network disabled. The separate 24/96/240-tile scale receipt checks dense shape growth and CPU timing; its largest split case stores 229,440 float64 CPT values for 480 traversed links. This pilot measures a bounded spatial proposal, not an RGM implementation. The full project short suite passed all 2,751 default tests (335 slow deselected), dashboard wall 1,159.265 seconds. Ready follow-up ccrp-x8yx owns learned sequence proposals and novel layouts. Ticket scan terms fixed/fit/learn refer to declared partitions and place assignments; readiness is ticket dependency status; exact refers to displacement or tree scoring. No repo runtime or numerical rule changed.
