---
id: cao-dq3h
status: closed
deps: []
links: []
created: 2026-11-24T14:51:17Z
type: task
priority: 2
assignee: coopwilliams
parent: cao-srin
tags: [axiom, report]
---
# Score the frozen AXIOM campaign and publish progress by stage

## Background in Plain English — WHY

The campaign starts with one integration run, checks exact continuation on 20 short runs, then evaluates 100 new game and seed pairs. Readers need to distinguish a partial reward trace from the published ten-game comparison.

## Outcomes / Acceptance Criteria — WHAT

1. The standalone report shows the assigned and completed run counts and each stage's release status. It reads only released stages.
2. Each completed score is reproduced from post-action rewards, including terminal rewards. The published primary comparison requires every assigned run, its identity, its tape digest and its source pin.
3. The scorer preserves the frozen window definition, margin, uncertainty calculation and all failed or incomplete runs.
4. A local HTML report and machine-readable summary can be regenerated offline. Synthetic fixtures exercise stage locks, tampered data, partial files, full inventory and the hand-computed reward example.

## Methodological Suggestions — HOW

Reuse the reference protocol and its independently checked 1,001-action example. The architecture change-path index has no runtime change for an offline evidence reader. Engine, compiler, model, serialization and host/device execution surfaces are N/A.

## Future work

The campaign owner runs the actual games, signs the qualification receipt and archives completed results under cao-srin.

## Definitions

A return sums actual rewards over a run, including episode boundaries. A trailing mean averages the latest 1,000 actions. A seed fixes a model or game random stream.


## Notes

**2026-11-24T14:58:38Z**

Completed the standalone scorer. Each stage opens only when the previous receipt binds the source fingerprint and the complete set of verified result hashes. Every completed result binds its raw step tape. The report validates seeds, native applied commands and rewards, terminal/reset receipts, reduction clocks, the 500-action engineering continuation and the two archived reference pins. Only the final receipt plus all 100 primary results can yield a comparison. Setup, first action/update, subsequent medians, final worker wall time and resume tick are separate. The primary JSON also stores all 9,001 trailing 1,000-action means per seed and per-game curves across ten seeds.

Validated with a synthetic archive using the full 1 + 20 + 100 assignment and 1,030,000 action rows. All 100 primary returns were three greater than their respective printed target. The independent expectation was reproduced: macro difference +3, printed-target interval [3,3], population comparison inconclusive for Impact and within the margin for Drive. The partial 1/100 arm received no primary estimate. Stale receipts, swapped seeds, altered source or trace pins, wrong earned return, and a single missing continuation check all withheld the downstream read. The live partial-line case, zero-action failure, all report links, summary byte digest and unique row inventory passed. The independent hand test moved seven reward units out of the sliding window at action 1,001. A separate native-format 10,000-action Hunt trace retained +2 and −2 (including two terminal rewards); decoded array equality, raw total 0, and first/last window means +0.002/−0.002 all passed.

Run `uv run --frozen --no-sync python tools/axiom/evaluation_report.py ROOT` to write both artifacts. Validation used the synthetic archive in /tmp/axiom_evaluation_report_check.py, the five frozen reference tests via `uv run --frozen --no-sync python tools/axiom/reference_protocol_check.py`, both CLI spellings, and py_compile. The scorer reads offline experiment files. It leaves simulator, inference, execution, and test selection untouched. No campaign or short suite ran. cao-srin owns release receipts and the actual campaign verdict. The companion precision report now shows the approved default in both precisions, the speed option, and the existing optional exact format. The regenerated HTML matched the locally served bytes.
