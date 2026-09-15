---
id: cdh-wi2b
status: closed
deps: []
links: [cdc-ztuf, cdc-c8kc]
created: 2026-09-15T02:42:06Z
type: bug
priority: 1
assignee: coopwilliams
---
# Isolate native D12 checks from short-suite cache settings

## Background in Plain English — WHY

The first final short suite for cdc-ztuf finished within the 20-minute limit at source c8beaf4: 2,831 passed and one failed in 1,142.379 seconds. The native-process abstract-storage test rejected its runtime identity because the suite supplied two persistent-cache settings that survived into a child where caching is disabled. A read-only comparison isolated exactly jax_compilation_cache_max_size (1073741824 versus -1) and jax_persistent_cache_min_compile_time_secs (0.1 versus 1.0). No numerical solve failed. Root owns this repair to the short-suite environment interaction under Cooper's explicit instruction to handle suite work personally.

## Outcomes / Acceptance Criteria — WHAT

1. The native child removes the two inherited, inactive short-suite cache controls while retaining its explicit disabled-cache, CPU and float32 policy.
2. The existing native-process/abstract-storage regression reproduces the suite's inherited cache settings and passes against the complete frozen descriptor. Source, runtime, leaf and dtype validation remain strict.
3. The unexecuted moderate descriptor is refreshed prospectively only for its changed producer hash; all three inventories and runtime values remain identical. Its specification records the old hash and new producer-bound descriptor without changing cases, numerical allowances, limits or execution counts.
4. Historical tiny attempts 001–004, frozen tiny specification and cold-compilation archive remain immutable. No D12 math source, optimizer run, moderate storage compilation or recovery fit changes or runs.
5. After focused validation and independent review of the committed repair, one fresh final short suite passes within the existing 1,200-second cap with unchanged selection. Preserve the first failed gate as evidence.

## Methodological Suggestions — HOW

The observed leak is in tools/d12_compile_scaling.py:child_environment. Consider removing only the two demonstrated passive cache variables alongside the already removed cache directory. Inject the real suite values in the existing subprocess regression. Regenerate the descriptor using its existing abstract-only command and compare each inventory to the previous artifact before updating its hash in the unexecuted plan. An equally narrow solution meeting the criteria is welcome.

Architecture handoff: Runtime numerical or status invariant row, limited to benchmark/dashboard process setup and the abstract-inventory consumer. The native subprocess and suite launcher must agree on the declared disabled-cache policy. Host/JIT and single/vmap numerical implementations are checked N/A because no numerical function changes; production flooding/scheduled paths, lifecycle, serialization and current/legacy checkpoint decoding are N/A. Exact-zero support, improper messages and 10–100x numerical domains are N/A for environment cleanup; native float32 policy and all existing T32/T100/K20-d10 abstract inventories remain checked. This affects test execution, so the isolated-tooling exception does not replace the full gate.

## Future work

This repair blocks closure of cdc-ztuf. cff-a3wz retains all twelve moderate solver starts and actual iteration/storage qualification, followed by production integration and cff-x0if's three recovery fits. All registered recovery trials remain unused.

## Definitions

- Runtime identity: recorded library, interpreter and configuration values tied to a descriptor.
- Descriptor: the complete shapes, dtypes and byte counts of retained abstract arrays.
- Persistent cache: stored compiled programs reused by later processes; disabled in these qualification children.



## Notes

**2026-09-15T02:44:47Z**

Root implementation complete; paused for independent review before any fresh full-suite run. child_environment now removes only the two demonstrated inactive cache controls alongside its existing cache-directory removal. Explicit CPU, x64=false, restrictive float64 policy, disabled persistent caching, XLA-cache setting and logging stay unchanged. The existing native-process test injects the real suite values (1 GiB cache maximum, 0.1-second minimum compile time), so it reproduces the failure mechanism even outside the full suite. No descriptor comparison was weakened.

Prospective descriptor generation: uv run python tools/d12_moderate_layouts.py, exit 0, log /tmp/cdh-wi2b-layout-generation.log. A recursive comparison against /tmp/cdh-wi2b-layouts-before.json.gz found exactly four changed scalar fields: only the tools/d12_compile_scaling.py producer hash at the root and each of three inventory bindings. All 1026 leaves per shape, runtime values and complete-program lower/compile/execute counts (0/0/0) are identical. Comparison artifact /tmp/cdh-wi2b-descriptor-diff.json. New descriptor SHA256 1b98bcfa48a3aeb8d6de3a6ca14c2c963f5e900faf019faa3f831935244672f9; new UNEXECUTED moderate spec SHA256 3a6bab37fcbab2b3e3ffdac4bc6892f17e2be244c4f2e65754b4b3c3ee660c3b. The previous descriptor hash and reason are recorded in the plan. Numerical formulas, cases, budgets, thresholds and scope are unchanged.

Focused command: uv run pytest -q tests/test_d12_moderate.py tests/test_d12_moderate_integrity.py -k 'native_process_setup or source_and_runtime_binding or complete_inventory' --junitxml=/tmp/cdh-wi2b-focused.xml. Result: 8 passed, 97 deselected in 20.89 seconds; /tmp/cdh-wi2b-focused.log. This includes the native float32 three-shape inventory and rejection of altered source/runtime/worker identities. All 12 tiny checkpoint source hashes and immutable attempt004 SHA b127d35227a32851ccfcd7721cc54fdab09c8311c63b47f9f63547520dd9a900 still match; no candidate requalification is needed because none of its sources changed. git diff --check and tk dep cycle pass.

Dependency companion from tk dep tree --full cdc-ztuf, tk ready and tk blocked; complete output retained at /tmp/cdh-wi2b-dep-tree.txt. Legend: prerequisite --> dependent.

[ ] cdh-wi2b (ready; repair reviewed next, shared suite pending)
[ ] cdh-urn0 (reviewed; shared suite pending)
[ ] cdh-m75a (reviewed; shared suite pending)
    +--> [ ] cdc-ztuf (immediately blocked by these three gates)
             +--> [ ] cff-a3wz (moderate numerical/storage qualification)
                        |
[ ] cff-k6g5 (ready independently) --> [ ] cff-7sw5
                                           +--> [ ] remaining D12 integration
                                                    +--> [ ] cff-x0if (three recovery fits)
                                                             +--> [ ] cff-zov5

The tree abbreviates downstream dependencies; tk remains authoritative. Cooper's existing short-suite exception assigns this environment interaction repair to root; normal bounded D12 implementation remains delegated sequentially. No fresh short suite, moderate optimizer/storage compilation, source recovery fit or publication has run since this repair.

**2026-09-15T02:49:09Z**

Independent review completed after root committed and paused at ca02216cc3c8f3f320f890ee0bbd67ad13bac6f1: no findings. The reviewer independently compared the descriptor and plan with the preceding commit, reproduced the original helper's exact two mismatches, and checked fresh Python/JAX children with both actual suite settings and malformed inactive cache values. Both repaired children match the entire frozen trace configuration; parent and unrelated environment values stay intact. Twelve independent corruption probes across the three shapes still reject producer, cache-runtime and float64-leaf tampering. Every historical tiny report and all twelve executed-source hashes remain unchanged. No descriptor regeneration, D12 compilation, numerical solve, recovery fit or broad test occurred during review. Report /tmp/cdh-wi2b-independent-review.txt; executable audit and results /tmp/cdh-wi2b-independent-review.py and .json (0.881 seconds).

Root inspected and accepted the review, then notified the reviewer. All code is now final and paused. The fresh final suite is required because the environment helper and regression changed after the prior failed gate. Root inspected only latest.wall_seconds=1142.379 and suggested_wait_seconds=1200 before launch; the next command is uv run python tools/dashboard.py --run-suite, with the existing 1200-second hard limit, unchanged selection, and a full quiet suggested wait. Ticket remains in_progress until that gate passes. The first failure remains recorded at b0b5997 and in /tmp/cdc-ztuf-short-suite.log. New gate log: /tmp/cdh-wi2b-short-suite.log.

**2026-09-15T03:12:53Z**

Done: the two inactive inherited cache settings are removed, the native regression reproduces the real suite environment, and strict source/runtime/leaf checks remain intact. Independent review found no issues. All1026 leaves per shape and runtime values remain identical; only the producer hash changed in the unexecuted descriptor. The shared final short suite passed at 144fb21466894bfd91d2040ff2d5e1d003cc82e9 after all code and independent reviews: uv run python tools/dashboard.py --run-suite; 2832 passed, zero failures, pytest exit0, pytest1125.74 seconds and measured wall1126.534 seconds. The unchanged1200-second cap has73.466 seconds of margin. Four loadfile workers used an initially empty1GiB cache. Root waited quietly for the full suggested1200 seconds before retrieving the result. Phase audit verifies8496 phase records,2832 distinct call cases exactly once and no duplicate phase. Runtime history and ~/agent-share/cffg-tests.html were refreshed by this single passing run; no unchanged repeat is needed for publication. Log /tmp/cdh-wi2b-short-suite.log; summary /tmp/cdh-wi2b-short-suite-summary.json. The preceding failed gate remains recorded; the ordinary cache-environment defect was repaired and independently reviewed before this new run. Close this bounded repair; root proceeds with parent publication.
