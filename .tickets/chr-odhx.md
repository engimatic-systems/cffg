---
id: chr-odhx
status: closed
deps: []
links: [chr-y2wk]
created: 2026-11-25T06:19:17Z
type: bug
priority: 2
assignee: coopwilliams
---
# Restore frozen-chart coverage for the float32 adaptive reactor certificate

An opt-in slow reactor test raises IndexError before checking the returned state. The accelerated sigma-point solver builds its chart in adaptive_runtime, while the test observes only outer.build_outer_chart. The scientific check should still compare the actual accepted current and mapped state in the frozen solve coordinates.

## Design

Architecture change-path: runtime numerical/status invariant. Test only. Run pytest on tests/test_outer_failures.py -k float32_reactor_returned with -o addopts= and inspect which adaptive path provided the retained chart. Keep both root residual and stationary free-energy checks. Guard against an empty capture with an informative assertion.

## Acceptance Criteria

The original slow test reproduces at 51ae6a7. Cover the actual candidate production route and retain the independent returned-state checks at the 64-eps float32 tolerance. Do not replace the frozen chart with a new chart built from the terminal state. Runtime policy or numerical changes are separate work.


## Notes

**2026-11-25T06:19:36Z**

Question: does the fixture fail before the new compiled controller? Replayed unchanged 51ae6a7 src, examples, tools, tests and pyproject in a temporary directory with the same uv interpreter, -k float32_reactor_returned -o addopts=. It fails at tests/test_outer_failures.py:136 (empty captured charts), as on the feature branch. The regular short suite excludes this slow case.

**2026-11-28T03:59:19Z**

Architecture handoff: runtime numerical or status invariant. The test exercises the float32 adaptive host acceptance path for the flood schedule. The general trust solver has its own synthetic certificate tests. No production change to Engine, adaptive or trust solvers, compiled controllers, Colony, the ledger, or serialization. Parent owns the combined gate.

**2026-11-28T04:00:55Z**

Reproduced the empty capture on base a774a54 (IndexError in 11.01 s). The test body there matches the previously reproduced 51ae6a7 version. The test now observes outer_strategy.certify_adaptive_proposal after the real host acceptance check. It requires one accepted attempt, compares the returned coordinates exactly to that accepted point, then independently applies one undamped native map and evaluates the two energies. The root and relative ledger bounds still use 64 float32 eps. The test keeps its slow marker. No runtime change.

Reproduce from this worktree using UV_PROJECT_ENVIRONMENT=/Users/ravd/projects/cffg/.venv PYTHONPATH=src uv run --frozen --no-sync python -m pytest -o addopts= -q tests/test_outer_failures.py -k float32_reactor_returned. Passed in 11.93 s. Removing the -k selector ran all 31 outer-failure tests including the opt-in cases in 16.37 s. Validation used -p no:cacheprovider. Parent owns the combined suite and integration.
