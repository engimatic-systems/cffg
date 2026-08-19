---
id: cff-wea9
status: open
deps: []
links: [cff-oyie]
created: 2026-08-19T17:12:38Z
type: bug
priority: 2
assignee: coopwilliams
parent: cff-nkbj
tags: [rslds, validation, reproducibility]
---
# Fail final execution before fitting when legacy reference sources are absent

## Background in Plain English

The dual-family runner validates its frozen code and provenance before it
creates a final dataset, but it does not verify that the ignored legacy source
checkouts needed by the Python 3.11 paper-adapter environment exist. A fresh
worktree can therefore pass preflight and spend hours fitting CFFG before the
first adapter command discovers a missing local package.

This happened in cff-oyie attempt-0001. After about 96 minutes, all four
shared-initialization adapter chains reported that
`.reference-sources/autoregressive` was absent. The attempt was interrupted,
the exact pinned sources were materialized with `reference/bootstrap.py`, the
38-test nested runtime suite passed, and the unchanged frozen command resumed
as attempt-0002. This is an execution-preflight defect, not a scientific
failure and not a reason to alter the frozen protocol.

## Outcomes / Acceptance Criteria

1. Final preflight rejects a missing, stale, or identity-mismatched
   `.reference-sources` tree before it generates final data, begins a
   coordinate attempt, or invokes any fitter.
2. Preflight derives the expected source identity from `reference/sources.json`
   and the registered patch hashes, then checks the generated
   `.cffg-reference-sources.json` marker and every locked local source path.
3. Preflight verifies that the locked nested environment can import or invoke
   the comparison-adapter entry point without fitting a dataset.
4. A complete pinned source tree and locked environment pass without modifying
   source checkouts, the registered protocol, stochastic streams, or scientific
   settings.
5. A regression uses a temporary repository with absent local sources and
   proves that no coordinate attempt, dataset, or result artifact is created.
6. The experiment README lists source bootstrap and locked environment sync as
   required preparation for final execution.

## Methodological Suggestions

- Reuse the identity logic in `reference/bootstrap.py` rather than maintaining
  a second list of commits and patch hashes.
- Keep materialization an explicit operator step. Preflight should diagnose the
  exact missing or mismatched input; it should not clone, patch, or synchronize
  an environment on behalf of the confirmatory command.
- Put the check in the final-runtime preflight that already validates revisions
  and frozen provenance so every final entry point receives the same barrier.

## Future Work

This ticket does not alter completed calibration evidence, rerun the
interrupted attempt, change append-only checkpoint policy, or tune a scientific
setting. cff-oyie owns the resumed frozen execution and preserves attempt-0001
as explicit interruption evidence.

## Definitions

- **Legacy source tree:** The ignored, patched checkouts materialized under
  `.reference-sources` from pinned commits, archives, and patch hashes.
- **Preflight:** Read-only validation that must pass before final data or fitting
  begins.

## Notes

Architecture handoff: the applicable `docs/ARCHITECTURE.md` boundary is the
sibling validation harness and isolated reference adapter. CFFG compiler, VM,
schedule, kernels, ledger, serialization, learning declarations, and factor
implementations are N/A.

Duplicated execution surfaces: both dataset families and every final entry
point use the same preflight; calibration source setup remains unchanged;
operator documentation and the final runner must agree on the bootstrap and
locked-sync sequence.

Hostile-input decisions: absent paths, a malformed marker, changed patches,
wrong commits, and an unusable environment fail loudly before final data.
Sparse probabilistic support, improper beliefs, float32 inference, and large
graph scale are N/A because this ticket validates filesystem and environment
identity without running inference.

Legend: prerequisite --> dependent

```text
[x] pinned sources + bootstrap identity
      |
      +--> [ ] cff-wea9 final preflight hardening (ready)

[x] cff-huxt freeze --> [ ] cff-oyie final run --> [ ] cff-nkbj epic
                              |
                              +--> [ ] cff-wea9 follow-up (does not block run)
```
