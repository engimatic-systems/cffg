---
id: chr-odhx
status: open
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
