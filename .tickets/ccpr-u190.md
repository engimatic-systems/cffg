---
id: ccpr-u190
status: open
deps: [ccpr-rl0k, cff-j7mm]
links: []
created: 2026-11-28T05:57:47Z
type: feature
priority: 3
assignee: coopwilliams
parent: cff-6s31
tags: [plates, compiler]
---
# Estimate plate storage from the source before allocating inference arrays

## Background in Plain English

The existing estimator measures arrays after an engine is built. A model author deciding whether a grid fits in memory needs an estimate before compiling it. Repeated shapes supply enough information to count the persistent arrays by name.

## Outcomes / Acceptance Criteria

1. A public estimate from a plate template counts persistent state and parameters by name before compiling an engine. Count Chain and balanced Broadcast connections without visiting every copy.
2. Count shared parameters once and private parameters once per copy. Reject invalid shapes and overflowing products before allocating an index table.
3. Match the compiled named-bytes contract from cff-j7mm on the 64-by-64 grid and mixture-bank fixtures. Check the corresponding nested model when the nesting path is available.
4. Identify the numeric dtype and each quantity being counted. Distinguish persistent array storage from transient compilation memory and elapsed time.

## Methodological Suggestions

Use the direct-address contract for grid dimensions and compare estimates with independently inventoried compiled leaves. Start with one level, then verify the nested comparison before parent closure.

## Future work

cff-6s31 owns the full cross-feature timing and parity report.

## Definitions

Persistent storage is the byte count of arrays that survive compilation and belong to an inference state or its parameters. A leaf is one array in that inventory.
