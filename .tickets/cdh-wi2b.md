---
id: cdh-wi2b
status: in_progress
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


