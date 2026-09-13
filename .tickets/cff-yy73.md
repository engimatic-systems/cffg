---
id: cff-yy73
status: closed
deps: []
links: []
created: 2026-09-13T16:42:05Z
type: task
priority: 1
assignee: coopwilliams
tags: [testing, tooling, performance]
---
# Shorten default testing and define sustainable test coverage tiers

## Background in Plain English — WHY

The recent CFFGZ PR chain ran two independently valid PR-head gates, each taking about 20 minutes. Cooper wants shorter routine testing without losing important correctness checks, fewer redundant parameter combinations, and a clear protocol for assigning long-running regressions to PR-head validation.

## Outcomes / Acceptance Criteria — WHAT

1. An audit names concrete costly tests and parameter matrices using existing measured durations, distinguishes redundant cases from independent failure mechanisms, and records the coverage kept for each proposed reduction.
2. Developer and PR-head validation have explicit selections and commands. Every reported result identifies its selection, and runtime histories compare only matching selections.
3. The default developer selection keeps inexpensive independent correctness oracles, divergence/status checks, and representative sparse-support, float32 and scale regressions.
4. Expensive integration or expanded validation moved out of the default remains discoverable and is required at the documented PR boundary or named change paths.
5. Redundant combinations are reduced only with an explanation of which distinct mechanisms remain tested; expensive tests are not removed merely because they have rarely failed.
6. AGENTS.md documents test-writing obligations for parameter axes, fixture size, execution cost, oracle independence and suite placement. README gives the commands.
7. Changed selection/reporting tests and collection checks pass. Final validation reports the measured PR-gate duration and the deferred cases' contribution from that run. A separate unchanged developer suite is not run solely for benchmarking; its wall time is not inferred from testcase sums.

## Methodological Suggestions — HOW

The implementer may choose another sound method. Audit parametrization, costly inference/JIT setup and testing protocol independently using subagents. Use existing dashboard timings for discovery. Treat per-case duration sums as estimates, not fresh wall measurements. Select representative boundary cases for every real semantic axis and use focused expensive tests where the affected behavior requires them. Review tools/dashboard.py, pyproject.toml and tests/conftest.py together so completeness checks and history labels remain honest. Changes to discovery, selection, execution or pass/fail interpretation are not isolated tooling and require the full validation gate.

## Future work

cff-sten will examine shared preparation for constraint checkpoint migration and rejection tests, preserving their distinct formats and failure mechanisms. Keep numerical correctness obligations and the separate PR heads in the existing CFFGZ chain intact.

## Definitions

- Parameter matrix: combinations generated from pytest parameter axes or loops inside a test.
- Failure mechanism: a distinct way implementation can violate its contract, rather than a different numeric example of the same path.
- Developer selection: the relatively quick default checks used during implementation.
- PR-head gate: required validation of the final code proposed for merging.
- Oracle: an independent expected result used to detect implementation errors.


## Notes

**2026-09-13T16:46:56Z**

Planning and audit handoff: test-selection/reporting changes use the Runtime numerical or status invariant row as the nearest change path, with authority in pytest configuration, conftest collection and dashboard reporting. Engine/math/serialization implementations are unchanged (New IR field and rule changes N/A). Preserve host/JIT, flooding/scheduled, single/vmap, current/legacy checkpoint and numerical hostile-input coverage; current/legacy suite reports instead require explicit provenance so partial results cannot certify PR coverage. This is not isolated tooling.

Research question -> answer -> sources -> confidence:
- Which large parameter matrices are redundant costs? The 720 EF property cases contribute only about 0.05 s rounded time, while 17 mu3 checkpoint corruption/configuration cases spend 183.16 s repeatedly preparing agents. Keep their distinct rejection mechanisms and share immutable prepared checkpoints with fresh independent targets. Source: tests/test_ef_properties.py, tests/test_agent_mu3.py and shared dashboard; high confidence in duplication, fresh measurements pending.
- Which long cases can move to PR head? NASCAR finite-cycle, glucose cohort efficacy, tenfold fitter horizon, and the float64 large Gaussian execution duplicate total 100.410 s historical testcase time. Keep the float32 hostile-scale Gaussian, small independent FE/marginal oracles, replay and solver-status guards in default. Source: test_rslds_fit.py, test_glucose.py, test_gaussian_kernels.py and dashboard; high confidence, historical sums are not a new wall-time measurement.
- How can selections remain honest? Default pytest excludes slow and pr_head, while dashboard explicitly selects not slow and remains PR-only. Require JUnit selection/scope/count provenance in addition to inventory/freshness checks. Source: pyproject.toml, tests/conftest.py, tools/dashboard.py, tests/test_dashboard.py; high confidence.

Recorded timings came from dashboard commit 48dbc8b (the CFFGZ stack), whereas this new branch starts at main 070a15b. Later bundle-only tests are absent here; comparisons will state that scope difference. Kalman oracle tests repeat the same converged run four times and can share read-only state without dropping assertions.

**2026-09-13T16:53:38Z**

Implementation validation: all 17 checkpoint rejection mechanisms remain. Two independently prepared checkpoints have different particle values, JAX cursors and NumPy RNG states; only immutable archive bytes are shared, and every test gets a fresh target and array dictionaries. Atomicity checks additionally snapshot plan-result identity, cursor, epoch and pending flags. Focused checkpoint validation: 25 passed, 14 deselected in 60.52 s; the changed group consumed about 33.84 s including shared preparation, compared with 183.16 s in the historical dashboard (not a controlled same-head comparison). Shared read-only Kalman solve keeps all independent assertions: 11 passed in 15.46 s. Metadata/dashboard checks: 25 passed in 0.99 s. Collection: developer 2218 selected / 45 deselected; PR gate 2222 selected / 41 deselected. Final validation will run the required PR gate once and extract case durations from that same run; there is no second overlapping suite just for timing.

**2026-09-13T16:55:10Z**

Final review found and fixed a collection-provenance gap: overriding python_functions could omit PR regressions while retaining every module. Explicit discovery overrides and alternate config files now mark reports partial. Real subprocess cases cover the normal gate, developer selection, keyword filtering, file narrowing, explicit deselection, last-failed selection, discovery overrides and alternate config. Final focused metadata/dashboard validation: 27 passed in 1.28 s. Deferred prepared constraint checkpoint reuse is tracked in cff-sten; tk dep cycle passed.

**2026-09-13T17:23:15Z**

Completed implementation and final validation at code commit 149ff54. The developer selection is not slow and not pr_head; dashboard runs the complete not slow PR gate exactly once. The four PR-only cases all passed, contributing 103.626 s in that gate: NASCAR cycle 38.804 s, glucose cohort 33.993 s, large float64 Gaussian 17.044 s, tenfold fitter horizon 13.785 s. These remain mandatory at each PR head and must be included in focused runs when changing their behavior.

The 17 checkpoint rejection cases now contribute 34.014 s, including shared preparation, versus 183.158 s in the earlier shared dashboard. The four Kalman assertions together now contribute 4.203 s versus 14.606 s. No original assertion or invalid-input variant was removed; source and target checkpoint states are independent and atomicity assertions were strengthened.

Final command: uv run python tools/dashboard.py --run-suite. Result: 2224 passed, 41 slow cases deselected, four existing Gym deprecation warnings; pytest 988.85 s, measured subprocess wall time 1001.442 s (16m41s). tools/test-suite-runtimes.json and ~/agent-share/cffg-tests.html were refreshed. Prior tracked main gate: 2210 tests, 1163.362 s wall time (19m23s), the current run is about 162 s / 14% shorter. Compared b01edae..main source/tests/config: only runtime-history JSON differed, so this baseline has the same production/test source as our starting main. It remains a historical comparison, not a controlled repeated benchmark. The older 2270-test HTML includes the separate CFFGZ stack and is used only for named unchanged test-group comparisons.

No separate developer suite was run just for benchmarking. The 103.626 s deferred-case sum is not a measured developer wall-time saving. Default selection behavior and report rejection were verified using real tiny subprocess suites; broad coverage passed at the final code. AGENTS.md and README document per-axis coverage claims, independent oracles/execution surfaces, representative hostile inputs, safe fixture reuse, timing review from necessary runs, and one gate per independent PR head. Follow-up cff-sten is ready after this ticket closes. No engine or bundle code changed.
