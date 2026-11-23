---
id: cpr-pch9
status: closed
deps: []
links: []
created: 2026-11-22T02:01:05Z
type: bug
priority: 2
assignee: coopwilliams
tags: [premerge, api, compatibility]
---
# Restore core and models access after a plain package import

## Background in Plain English — WHY

On main, a fresh interpreter can import cffg and then access cffg.core.Template or cffg.models.rslds. The lazy facade introduced in PR 156 and present after PR 158 raises AttributeError for both until the subpackage is imported explicitly. This breaks existing callers even though all old names in __all__ remain present.

## Outcomes / Acceptance Criteria — WHAT

1. In separate fresh interpreters, import cffg followed by either of the former attribute access patterns resolves the expected object.
2. Explicit subpackage imports and the existing top-level export catalogue still resolve to their canonical objects. Preserve lazy loading for unrelated modules.

## Methodological Suggestions — HOW

There is no package facade row in the architecture change-path index. Check the package facade and its public API tests. Host/JIT, flooding/scheduled, single/vmap and checkpoint decoding are N/A. For compatibility, run a probe from a fresh interpreter or the installed wheel so unrelated test imports do not prepopulate package attributes. No numerical or persistence changes are involved.

## Future work

Independent of the closed-loop pilot. No wider expansion of the root package API is required.

## Definitions

A facade is a supported access point for public objects. Lazy loading imports a submodule only when someone uses it.


## Notes

**2026-11-22T02:02:35Z**

Attribution verified with fresh interpreters. Both attribute reads work on PR 155 and fail on PR 156, where 7ca753a introduces lazy root exports. PR 158 retains this behavior. See ~/agent-share/cffg-premerge-review/early/observed.txt.

**2026-11-23T03:10:32Z**

Reproduced both missing attributes in separate isolated Python processes before editing. The root facade now imports core and models on attribute access and lists both in dir(cffg). The existing wildcard export catalogue is preserved. The core subprocess also blocks JAX and Equinox and checks that models stays unloaded. Validation passed with uv run --frozen --no-sync python -I -m pytest -q tests/test_core_imports.py tests/test_public_api.py::test_top_level_facade_reexports_only_canonical_owners (4 tests). Parent owns the final short suite.
