---
id: cpsa-kghq
status: closed
deps: [ccpr-m0bu]
links: []
created: 2026-11-28T13:49:38Z
type: bug
priority: 2
assignee: coopwilliams
parent: cff-6s31
---
# Reject invalid generated plate addresses during compilation

A generated observation address can point one slot beyond the message array. A two-pixel test injected that bad address into the compiler layout. Compilation succeeded and building an Engine failed later. The compiler should identify the port or edge before publishing the arena.

## Design

Use the architecture change path for compiler layout and resolution. Check generated slots against their own discrete, Gaussian or exponential-family storage. Check an observation refers to its intended edge. Keep inference arithmetic, schedules, checkpoints, and valid indices intact. Engine and Colony share initial compilation; cached spawns reuse a previously checked layout.

## Acceptance Criteria

Negative, end-of-array, and valid-but-misdirected generated addresses are rejected during compilation with an identifying error. Valid compact and explicit layouts still agree, including both precision modes, nested plates, and the existing 64-by-64 compilation budget.


## Notes

**2026-11-28T13:51:48Z**

The original injected DATA slot 9 in an eight-slot, two-pixel graph survived compilation and raised IndexError during Engine construction. The compact compiler now checks its domain-specific slot ranges, endpoint mappings, and observed edge before resolving parameter data. Five focused corruption cases cover a negative slot, a slot past the table, an in-range slot belonging to a different port, a damaged gather, and an invalid edge address. All 32 initial focused cases passed; 29 cases on final code passed. Joined scale and suite qualification will be recorded with cff-6s31 and ccpr-u190.

**2026-11-28T15:16:57Z**

The guard in 83c7708 is in the joined source 8b826f9. Five injected corruptions and 29 public/flat/compact parity checks passed. Joined short suite: 2,918 passed in 435.25 s. Profile at the same head: 64x64 first build 0.76483 s, median of five subsequent builds 0.737124 s (target <=1.5 s), six identical index digests, empty copied-spec counter. The separate instrumented address check took 0.005082 s. Receipt ~/agent-share/plate-release-8b826f9-2gik0n/result-64.json, SHA256 1ef8078c414faab1f11f41fad1f20efd99ff7aa64c789026009d96734b2af6c0.
