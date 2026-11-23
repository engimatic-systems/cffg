---
id: cfr-ex1c
status: open
deps: []
links: []
created: 2026-11-23T03:03:36Z
type: bug
priority: 2
assignee: coopwilliams
parent: cff-xa6z
tags: [premerge, pcgym, provenance]
---
# Require all frozen worker source hashes when auditing a relocated comparison

## Background in Plain English — WHY
The #192 verifier matches frozen source paths against its current checkout directory. Running the verifier from the #195 worktree selects zero of ten source names for the earlier study. The audit then accepts a worker manifest with a changed adapter digest. If the original checkout is absent, it also tries to open stale paths. Independently matching all ten expected source names against all 394 actual manifests found zero mismatches in the completed study.

## Outcomes / Acceptance Criteria — WHAT
1. Require a complete and unambiguous set of expected source identities before comparing workers.
2. A changed or absent worker digest fails the audit from both the original checkout and a relocated checkout with matching sources.
3. Report missing historical source bytes clearly. Keep support for the earlier archive format and its pinned revisions.

## Methodological Suggestions — HOW
Review tools/pcgym_timing/paired_provenance.py:20–24 and :47 plus run_paired.py:34. Use stable repo-relative identifiers and an explicit input location for historic sources. One in-memory mutation of cffg_adapter.py in a worker manifest is sufficient to exercise the false pass. Isolated benchmark audit; production engine, schedules, and serialization are outside this change.

## Future work
cpt-j81k holds the original thirty-seed comparison. No new measurements are required to demonstrate this provenance repair.

## Definitions
A digest is a fingerprint of file contents. A worker manifest records what code one benchmark process used.
