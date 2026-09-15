---
id: cdc-c8kc
status: closed
deps: [cdc-bwta]
links: [cdc-ztuf, cdh-wi2b]
created: 2026-09-14T20:25:51Z
type: task
priority: 1
assignee: coopwilliams
tags: [testing, performance, tooling]
---
# Restore short-suite margin using measured execution costs

## Background in Plain English — WHY

The expanded Y15D short suite passed at 96276c8 with 2659 distinct tests, four workers and a fresh locked 1-GiB cache in 1199.035 wall seconds. It met Cooper's hard 1200-second limit with only 0.965 seconds to spare. Worker phase totals were about 975, 993, 977 and 1185 seconds, indicating a substantial scheduling imbalance. New D12 harness tests will need room. Cooper assigned short-suite performance implementation to root.

## Outcomes / Acceptance Criteria — WHAT

1. Use the retained per-file/per-worker evidence to select a bounded scheduling or ordinary performance improvement. Preserve every selected test/assertion, the not-slow selection, failure handling, no retries, cache isolation and hard 1200-second process-group deadline.
2. Keep implementation small and maintainable. Any cost-based ordering must be deterministic and handle missing/new test files safely; it cannot omit or duplicate work. Avoid fragile special cases keyed to individual test incidents.
3. After focused runner checks and independent review, run exactly one final default dashboard gate after the last relevant edit. Report measured wall time, margin, worker/cache conditions, one distinct call per selected test and worker balance. Demonstrate improved margin over the retained 1199.035-second baseline; preserve failures before fixes.
4. Record final implementation/source and evidence in this ticket and PR. Root implements this work. It does not change D12 numerical or recovery thresholds and cannot use the isolated-tooling exception because scheduling affects test execution.

## Methodological Suggestions — HOW

Start with /tmp/cff-y15d-short-suite-summary.json and the original worker JSONL named in cff-y15d's final note. The largest file costs were joint coordinate 587.98s, projection 522.85s, precision 493.60s and centering 438.27s. Measured work ordering may address the 200-second tail without increasing concurrent compiler memory. Preserve the existing fresh-cache and process ownership behavior.

## Future work

Proceed after the immediate bounded compilation-scaling evidence requested by Cooper and before relying on the short suite for further harness growth. Moderate numerical qualification and registered recovery remain separate.

## Definitions

- Margin: 1200 seconds minus measured wall time.
- Worker balance: distribution of executed test/setup/teardown seconds across pytest workers, not a numerical qualification measure.

## Change paths and hostile inputs

Runtime numerical or status invariant row for test execution/pass-fail reporting only. Affected surfaces: dashboard/direct suite invocation, worker/controller scheduling and artifacts. Production numerical host/JIT, single/vmap, compiler, lifecycle, serialization and learning are N/A. Applicable hostile cases are missing timing data, new files, timeout, crashed worker, incomplete results, artifact races and duplicate/missing execution.



## Notes

**2026-09-14T20:42:34Z**

Root read-only scheduling preflight after the Y15D gate (production/test code unchanged). Installed pytest-xdist3.8.0 LoadFileScheduling inherits LoadScopeScheduling. schedule() currently sorts file groups by descending TEST COUNT when loadscopereorder is true; --no-loadscope-reorder retains their collection order. _reschedule assigns another complete file when a worker has at most two pending tests. Sources: .venv/lib/python3.12/site-packages/xdist/scheduler/loadfile.py; loadscope.py:313,338,374; plugin.py:131,144. Existing tests/conftest.py has controller-owned phase timing and artifact cleanup, but no collection-order hook. tools/dashboard.py:169 controls the four-worker loadfile command and fresh cache.

Offline analysis grouped the retained 7977 setup/call/teardown records into 2659 distinct tests and their files, holding each observed phase duration fixed. A four-worker event-queue simulation with the installed low-water rule and current descending-count ordering reproduced the actual worker phase totals exactly: 993.466/974.803/977.019/1185.420 seconds. Reordering the same complete files by descending observed cost produced simulated totals 1032.674/1032.667/1032.668/1032.700 seconds, reducing the longest simulated phase sum by152.72 seconds. Artifact: /tmp/cdc-c8kc-scheduling-simulation.json. These are conditional scheduling-model results, NOT measured suite speed or promised wall time; actual cache/concurrent-compilation costs may change. No tests, compilation or solves ran.

A small candidate implementation is a deterministic collection-order hook using measured per-file costs plus --no-loadscope-reorder, while retaining xdist's existing complete-file scheduler and every discovered test. Avoid enumerating only known files on the pytest command line, which could silently miss new/nested discovery. Preserve within-file order; define safe behavior for absent/new/malformed timing data. Store only runtime-history evidence in the existing approved runtime-history mechanism, not repository session-state notes. Test actual scheduler/collection behavior on bounded synthetic cases and then use the required one fresh full gate after review. Root owns implementation after cdc-ogj9's immediate small measurement request.

Existing record_suite_runtime(tests, wall_seconds, pytest_code) in tools/dashboard.py:336 already receives each validated test's file and elapsed milliseconds. Per-file sums could be persisted as runtime-history metadata alongside the existing aggregate without adding another session-state file or needing an additional test run. Ordering is only a hint: preserve collection/selection and handle unknown filenames/costs safely. JUnit time precision and phase-timing sums differ slightly; disclose which source supplies ordering. A driver-provided cost mapping consumed by a deterministic conftest collection hook plus the existing --no-loadscope-reorder flag could avoid copying or subclassing xdist's scheduler. This remains a candidate, not implemented or qualified behavior.

**2026-09-14T21:05:09Z**

Prospective root implementation after cdc-ogj9 completed: keep four workers and xdist's existing complete-file scheduler, but order collected files by measured per-file cost and disable xdist's subsequent count-based reorder only when a valid cost map is available. A driver-provided map changes ordering only, never discovery/selection or within-file order. Preserve new/unknown files, with a deterministic conservative priority; missing usable history falls back to existing scheduling. Malformed mapping data must not omit tests or publish success. Persist per-file elapsed sums as optional metadata in the existing approved runtime history, selecting a recent complete passed not-slow result; failed/partial/incompatible results cannot replace good scheduling evidence. Seed the already measured96276c8 record from its retained phase log, without changing its wall time/count/result or presenting it as a new run. No dependency, numerical/source/recovery policy, worker count, cache/deadline, retry or artifact ownership change.

Architecture row checked: Runtime numerical or status invariant (dashboard/test consumers), docs/ARCHITECTURE.md:130. Applicable duplicated surfaces are dashboard/direct invocation and worker/controller collection/order/artifacts. Production runtime host/JIT, single/vmap, compiler, lifecycle, serialization and learning remain N/A. Meaningful focused checks will cover real collection/scheduler behavior, unchanged item inventory/order within files, missing/new/malformed cost data, successful-history selection/persistence, fresh cache and timeout/partial-report behavior. Commit implementation and pause for an independent reviewer before the one final dashboard gate. The retained scheduling simulation is guidance, not measured speed; final wall time and margin must be reported honestly.

**2026-09-14T21:12:45Z**

Root implementation is complete for review. tools/dashboard.py reads usable costs from a recent passed not-slow record, passes a checked map to workers, disables xdist's count reorder only when costs exist, and persists per-file sums after a passing full gate. The collection hook preserves all items and within-file order, prioritizes unknown files without floating-point sentinels, and rejects malformed direct maps. Missing usable history retains default file scheduling. Four workers, cache, budget, no-retry policy, discovery/selection and engine/numerical code are unchanged; AGENTS documents ordering.

Focused command after final edits: uv run pytest -q tests/test_dashboard.py --junitxml=/tmp/cdc-c8kc-focused-final.xml; 30 passed in1.26s, log /tmp/cdc-c8kc-focused-final.log. Checks exercise installed xdist's real file scheduler with complete item-inventory accounting, unknown-file priority/within-file order, malformed history/maps, driver flags, successful per-file persistence, cache isolation, deadline and report integrity. The first run had29 passes and one incomplete synthetic worker fixture (missing gateway identity); corrected before this final passing run; initial log /tmp/cdc-c8kc-focused.log retained. No full suite or numerical benchmark was run. Commit and pause for independent review; afterward root owns the single final four-worker default gate. The timing simulation remains an estimate; improved wall-time margin is not yet measured.

**2026-09-14T21:19:33Z**

Independent review of8bbf347 completed with no actionable findings; fresh persisted Codex CLI reviewer exited0, report /tmp/cdc-c8kc-review.txt. Its synthetic real four-worker run passed25 tests and verified identical worker collections, complete inventory, one execution per item, within-file order, unknown/nested files and slow exclusion. It independently matched the seeded costs against all7977 retained phase records and confirmed unchanged historical time/count/result and a clean tree. Its focused copy passed29 tests; the existing deadline test could not complete because that review sandbox blocked ps. Root's already completed unrestricted final focused run passed all30 tests, including the real deadline/child-process check, in1.26s. No source repair or unchanged test repetition is needed.

Proceed to exactly one final default dashboard gate at the unchanged implementation, four workers and a fresh locked1GiB cache, with measured file ordering and hard1200s deadline. Inspect only the two runtime wait fields before launch, announce and wait the full suggested budget quietly. Actual improved speed and margin remain unverified until this gate; preserve any failure before ordinary diagnosis. No numerical/source/recovery semantics changed.

**2026-09-14T21:41:20Z**

Final default gate passed at b6b355a (implementation 8bbf347): uv run python tools/dashboard.py --run-suite, 2668 passed, zero failures, 1073.76 pytest seconds and 1074.552 measured wall seconds. This is 17m54.552s, with 125.448s of margin under the unchanged hard 1200s deadline and a 124.483s reduction from the retained 1199.035s baseline. Four complete-file workers, measured ordering from 112 file costs, fresh locked 1GiB JAX cache, unchanged not-slow selection and no retries. The 16 warnings are existing gym/distutils deprecations repeated across workers. Root announced the budget and waited the full 1200s before inspecting results; no unchanged rerun is needed for publication.

Root audited all 8004 setup/call/teardown records: 2668 distinct call nodeids, exactly one call each, all phases passed, zero duplicates or missing selected calls. Worker phase totals gw0/gw1/gw2/gw3 were 1042.476/1042.276/1042.660/1042.417 seconds (spread 0.384s), compared with the prior roughly 211s spread. This is measured evidence for this machine/run, not a guarantee that future added tests fit; the enforced deadline remains necessary. No tests/assertions were removed and the nine added focused cases account for the full inventory increase from 2659 to 2668.

Artifacts: /tmp/cdc-c8kc-short-suite.log, /tmp/cdc-c8kc-short-suite-summary.json, and /var/folders/kg/pmjxhgy506n699dj2g96f2wh0000gn/T/tmpi8rnzt0c.timings.jsonl. tools/test-suite-runtimes.json records b6b355a with the measured full result and new JUnit-derived file costs; ~/agent-share/cffg-tests.html was refreshed by the same run. Independent review found no issues, root final focused checks passed all 30, and the review's real four-worker synthetic integration passed 25. Runtime history and this closure are the only post-gate changes.

All acceptance criteria are met. Close and publish a PR stacked on cdc-ogj9-compile-scaling / PR147. Next is cdc-ztuf's shared explicit-start entry and prospective moderate qualification harness, using a fresh implementation worker; cff-a3wz owns actual moderate solves/storage evidence. No recovery fits have run and no numerical, approximation, scope or threshold contract changed.
