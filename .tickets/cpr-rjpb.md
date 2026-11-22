---
id: cpr-rjpb
status: open
deps: []
links: []
created: 2026-11-22T02:01:05Z
type: bug
priority: 2
assignee: coopwilliams
tags: [premerge, api, diagnostics]
---
# Describe the max-product assignment score correctly in the public objective contract

## Background in Plain English — WHY

On a two-state categorical graph with two priors [0.2, 0.8], max-product returns the correct assignment score 0.4462871. Engine.objective_contract describes it as Bethe or generalised free energy even though Engine.bfe correctly rejects that use. A consumer inspecting the new metadata can misinterpret a valid score.

## Outcomes / Acceptance Criteria — WHAT

1. Public metadata names the assignment objective for max-product and agrees with objective() and the bfe() guard.
2. The ordinary sum-product and adaptive Gaussian objective descriptions continue to match their actual score. Exercise all three through public APIs.

## Methodological Suggestions — HOW

Use the Runtime numerical or status invariant row of docs/ARCHITECTURE.md for the host metadata/compiled objective pair. Flooding/scheduled, single/vmap and checkpoint decoding are N/A for this accessor. Inspect ledger.py contract and the Engine metadata and scoring accessors. A two-state graph gives the independent expected score −2 log(0.8). The change concerns metadata only.

## Future work

Independent of the closed-loop pilot. No change to max-product inference is needed for this finding.

## Definitions

Max-product searches for a most likely joint assignment. A Bethe or generalised free energy scores beliefs; the two numbers have different meanings.


## Notes

**2026-11-22T02:02:35Z**

The label first appears in 58185b4 on PR 156. Independently scoring the two equal priors gives 0.4462871 on its head and the stack tip. Fresh-process output and reproducer are in ~/agent-share/cffg-premerge-review/early/.
