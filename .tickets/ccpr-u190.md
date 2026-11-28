---
id: ccpr-u190
status: closed
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

## Notes

**2026-11-28T07:24:08Z**

Change path: resource estimate and persistent State layout. Count a compact source fragment once with PlateGrid multiplicities; debit one joined edge and two consumed ports per Chain bond, and add the balanced equality tree for Broadcast. Check the estimator against cff-j7mm state_storage (two dtypes, tied pixel grid and private mixture bank). Treat known native row layouts and arbitrary external pack callbacks separately; logical source owners and repeated packed rows have different byte totals. Never run a pack callback or open blob files to obtain an estimate. Malformed shape or overflowing intermediate counts must fail before allocation. Execution schedules, numerical rules, FE and serialization N/A. Nested parity awaits ccpr-m0bu.

**2026-11-28T13:50:21Z**

The isolated one-level source is a8f1f68 after cff-j7mm. Fifteen focused tests passed (one deselected) in 15.98 s, including comparison of actual initialized State leaf byte counts on a 64x64 grid in both dtypes. With the #239 compiler available, this extension follows enclosing owner multiplicities, direct parent declarations, and generated DATA/GLUE ports. Verify two-level compilation against independent fixtures, including partial rerouting of conditioned rows. Parent is joining the disjoint generated-slot fix for one combined estimator gate.

**2026-11-28T15:16:57Z**

Complete on 8b826f9. The isolated one-level inventory passed 15 tests; the joined estimator, nested fixtures and compiled leaf checks passed 40 (one deselected). The canonical short suite passed 2,918 tests in 435.25 s. At 64x64 the initialized f64 State held 327,680 message and 294,912 packed bytes while the two logical f64 sources required 72 bytes. Implicit, outer-private and outer-tied calibration fixtures match every named initialized State leaf in both precisions. Arbitrary unclassified packers raise UnknownStorageLayout before invoking callbacks. The separate matched compile profile on the same head measured 0.737124 s warm median, 4,096 DATA ports and 40,960 message values. Both depth-two accounting criteria are covered.
