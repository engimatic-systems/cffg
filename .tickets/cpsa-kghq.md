---
id: cpsa-kghq
status: in_progress
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
