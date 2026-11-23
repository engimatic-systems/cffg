---
id: cfr-ex1c
status: closed
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

## Notes

**2026-11-23T03:11:12Z**

Change path: docs/ARCHITECTURE.md, Runtime numerical or status invariant, benchmark consumer only. The verifier will bind the ten worker filenames to the historical checkout recorded by PCGym, verify the complete frozen file inventory and the actual bytes, and compare every worker against the same named hashes. Explicit locations can supply relocated checkout, simulator and peer files. The measurement protocol and run_paired.py stay pinned. Engine, host/JIT, schedule, vmap, model math and serialization execution pairs are N/A. Validation uses an isolated archive fixture, worker mutation and a read-only check of the real #192 archive with a relocated historical source snapshot.

**2026-11-23T03:16:15Z**

The auditor binds the original absolute paths to checkout-relative identities using the archived reference path, and separately resolves peer and simulator files. It requires the full signature inventory, the native factor, the runtime inputs, and all ten frozen source digests from every worker. --source-root, --peer-root and --experiment-root accept relocated directories. --check-only reads the archived data without publishing a replacement provenance.json. The measurement recorder and its hash are unchanged.

Isolated tooling validation: uv run --frozen --no-sync python tools/pcgym_timing/paired_provenance_check.py passed four tests. They exercise the original recorder format, relocation with all three original directories gone, missing/changed worker and source bytes, incomplete and ambiguous frozen inventories and duplicate worker keys. No engine, model, measurement or suite behavior changes.

For the real #192 archive, set archive="$HOME/agent-share/pcgym-paired-30" and snapshot=$(mktemp -d). Run git archive 15457bfb87171adbada2da21fd7957725e679b38 tools/pcgym_timing tools/benchmark_cstr_chain.py tools/cstr_reactor_pilot.py tools/cstr_runtime_fixture.py uv.lock | tar -x -C "$snapshot", then uv run --frozen --no-sync python tools/pcgym_timing/paired_provenance.py --output "$archive" --source-root "$snapshot" --check-only. Verified 20 frozen files, 12 runtime files and 10 sources in all 394 worker manifests (3940 comparisons). The saved provenance SHA-256 stayed 8517d2d40588cb13fe9ffd920c0ce1952d1f473ef62225c5a567755ee2d5e3b8. Independently copied all three roots and repeated the audit, which rejected an in-memory change to cffg_adapter.py in seed-1001/cffg-short; auditing against the current branch bytes also failed on rxinfer_adapter.py as expected.
