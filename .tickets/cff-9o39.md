---
id: cff-9o39
status: open
deps: [cff-i00o]
links: []
created: 2026-08-17T20:52:04Z
type: task
priority: 1
assignee: coopwilliams
parent: cff-nkbj
tags: [rslds, validation, execution]
---
# Execute dual-family calibration and final coordinates

## Background in Plain English

The existing confirmatory runner assumes one generator, one flat seed map, one conclusion, and one checkpoint per integer seed. The dual-family protocol from cff-i00o needs explicit family-qualified execution so canonical and heterogeneous evidence cannot collide or overwrite each other. Long calibration also needs the same restart safety as final execution.

This ticket updates runner orchestration, controls, raw rows, aggregation, exception handling, and append-only checkpoints. It validates compact calibration-only fixtures and does not execute registered final seeds.

## Outcomes / Acceptance Criteria

1. Calibration and final orchestration iterate explicit family, seed, replicate, tier, chain, and control coordinates from the frozen protocol instead of hard-coded global constants.
2. Every raw row records family, generator descriptor, seed, split, observation hash, stochastic streams, and arm status. Both learned controls use the protocol's declared regime counts and the identical observations and split.
3. One arm exception produces a complete failed/not-ready record and does not prevent later required arms or coordinates from running.
4. Family-aware aggregation requires every registered family/seed/tier/control coordinate, recomputes all paired and absolute outcomes from raw rows, and emits separate classic, heterogeneous, and gated joint decisions.
5. Available secondary evidence remains reportable when another coordinate fails. A family failure does not replace or erase the other family's valid descriptive result.
6. Calibration and final execution checkpoint the smallest durable family/seed/replicate coordinate. Each completed coordinate is written once with protocol and preregistration identity, a result hash, and a complete raw-attempt manifest.
7. Interrupted attempts receive immutable attempt directories and terminal ledger states. Resume validates every completed payload and runs only missing registered coordinates without replacing prior attempts.
8. Family-qualified stochastic streams and artifact paths cannot collide. A changed family, seed, protocol, registration, generator, or raw payload stops resume before further work.
9. Reports record family-specific generator provenance, source commits, exact commands, timings, memory, hardware, and total runtime. Manifest validation covers every family-qualified raw payload and aggregate.
10. Compact tests cover interruption during each family, corruption, provenance mismatch, partial-family failure, control failure, aggregate recomputation, and a complete both-family calibration fixture.
11. No registered final seed is generated or fit by this ticket.

## Methodological Suggestions

These suggestions are optional. The implementer may choose another sound method that satisfies the outcomes.

- Use `family/seed/replicate/attempt-N` directories and keep the checkpoint envelope outside the immutable attempt.
- Add an attempt ledger rather than inferring interruption from a missing checkpoint.
- Preserve compatibility wrappers for historical single-family comparison and cold-benchmark paths.
- Split new tests out of the already large preregistration test file.

## Future work

cff-s6rs uses the runner for measured dual calibration and scale selection. cff-huxt freezes the resulting protocol. cff-oyie is the only final-execution owner.

## Definitions

- **Attempt ledger:** An append-only record that labels each execution attempt completed, failed, or interrupted while preserving its payload.
- **Durable coordinate:** The smallest unit whose completed result can be reused after process interruption.
- **Family-aware aggregate:** A result that retains separate decisions and evidence for each named dataset family.
