---
id: cff-314c
status: closed
deps: []
links: []
created: 2026-09-15T16:39:13Z
type: bug
priority: 2
assignee: coopwilliams
parent: cff-o7hs
---
# Keep core model imports independent of the runtime facade

## Background in Plain English — WHY

Importing cffg.core currently executes the top-level package imports, which load JAX and Equinox before core editing starts. Nonlinear and neural template contracts require core validation and serialization to work without either runtime dependency.

## Outcomes / Acceptance Criteria — WHAT

1. A fresh process imports core, validates a template, and round-trips JSON while imports of JAX and Equinox are explicitly forbidden.
2. Existing top-level public exports resolve to their original objects when requested, including ordinary Python imports and attribute access.
3. Public API tests pass and the documented core/runtime boundary remains accurate.

## Methodological Suggestions — HOW

- Use lazy top-level exports. This is an import-boundary refactor under the New IR field change-path prerequisite; numerical algorithms and model schemas are unchanged.
- Host/JIT, flooding/scheduled, single-instance/vmap and current/legacy numerical decoding are N/A; fresh-process import and public-export tests are affected. The full short-suite gate still applies to this shared package change.

## Future work

- cff-2dzo, cff-o7hs and cff-1o72 use the JAX-free core boundary for portable model declarations.

## Definitions

- Facade: the public names exposed by the top-level package.
- Lazy export: a public object imported only when a caller requests it.



## Notes

**2026-09-15T16:40:46Z**

Implemented lazy public exports with unchanged object identities and discovery through dir(cffg). A fresh-process regression blocks JAX, jaxlib and Equinox imports while core validates, serializes, loads and hashes a template. ARCHITECTURE.md records the boundary; the test is assigned to the dashboard. Focused validation: uv run pytest -q tests/test_core_imports.py tests/test_public_api.py tests/test_moment_matching_schema.py tests/test_serialize.py tests/test_viz_diagram.py — 46 passed in 18.32s. Full short-suite validation is still due at the parent implementation head after the remaining code changes. No numerical algorithms or model formats changed. Handoff: the core import prerequisite is implemented; cff-2dzo continues endpoint schema integration.
