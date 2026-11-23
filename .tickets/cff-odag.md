---
id: cff-odag
status: closed
deps: []
links: []
created: 2026-09-22T19:16:26Z
type: task
priority: 1
assignee: coopwilliams
parent: cff-3c6f
---
# Define one coherent public facade and model catalogue

## Background in Plain English — WHY

The package facade and its documentation disagree. The current top-level cffg package tests 37 exported names, while the README calls only 10 names supported. Mature builders such as the generic rollout, T-maze, and linear Gaussian state-space model also exist in code but are missing from the cffg.models facade. A newcomer cannot tell which imports are supported or find the built-in model catalogue through normal package discovery.

This ticket defines a coherent pre-1.0 import contract without removing working imports. It makes the package facades match the capabilities that later documentation and examples will advertise.

## Outcomes / Acceptance Criteria — WHAT

1. Existing top-level cffg imports remain compatible.
2. The supported top-level export inventory is declared once in code and is covered by an exact public-API test.
3. cffg.models exposes every supported built-in experiment builder through its package facade, including the generic rollout, gridworld, T-maze, linear Gaussian state-space model, factored rollout, and rSLDS paths.
4. Each public package facade used by the newcomer guide has an explicit __all__ and a short module description that states its scope.
5. Runtime result types returned by supported APIs have an obvious documented import path, even if they remain owned by their focused module.
6. Import tests prove that facade aliases retain identity with their canonical owners and that cffg.core remains independent of JAX, JAXlib, and Equinox.
7. No inference, numerical, serialization, checkpoint, or objective behavior changes.
8. The ticket handoff records the exact export decision that the README ticket must document.

## Methodological Suggestions — HOW

These suggestions are optional. Another sound design is acceptable.

- Keep cffg lazy so importing the facade does not eagerly import JAX.
- Prefer complete subpackage catalogues over adding every specialized constructor to the top-level namespace.
- Consider grouping the top-level export declaration by workflow, neural/runtime extension, and kernel extension while preserving the existing order expected by tests.
- Add focused tests for cffg.models and any other facade changed here.
- Applicable architecture change path: no exact row applies. Authority is each package __init__.py; derived surfaces are public API tests and the later README/examples tickets. Host/JIT, flooding/scheduled, single/vmap, and current/legacy checkpoint pairs are N/A because import aliases do not change execution.

## Future work

- The final README import map and capability table are owned by the downstream README ticket.
- A future 1.0 compatibility policy may narrow or version these imports. That is not part of this pre-1.0 repair.
- No other future work is identified.

## Definitions

- Canonical owner: The focused module where a class or function is implemented.
- Export inventory: The exact names listed in a package __all__.
- Identity: Two imports refer to the same Python object, not wrappers with similar behavior.
- Lazy import: Loading a name only when it is first requested.
- Linear Gaussian state-space model: A time-series model with linear transitions and Gaussian noise, also called an LGSSM.
- Public facade: The supported names available from a package such as cffg or cffg.models.


## Notes

**2026-09-22T19:49:53Z**

Completion handoff, 2026-09-22. Replaced the duplicated top-level export declarations with one ordered lazy name-to-owner mapping, preserving all 37 existing top-level imports. Expanded cffg.models into the complete supported model catalogue: generic rollout, gridworld and CorridorEnv, T-maze, LGSSM, factored rollout, and rSLDS construction/fitting/learning. Added an exact facade identity test. Export decision for downstream docs: cffg is the lazy common runtime and extension facade; cffg.core owns JAX-free graph construction; cffg.models is the complete built-in model catalogue; cffg.viz owns rendering. Runtime result types remain publicly importable from cffg.engine and canonically owned by cffg.runtime_types. No numerical, inference, objective, serialization, or checkpoint behavior changed.

Validation: uv run pytest -q tests/test_facades.py tests/test_public_api.py::test_top_level_facade_reexports_only_canonical_owners tests/test_core_imports.py -> 4 passed. Final dashboard short suite after the last code change -> 2396 passed, 67 deselected, 0 failed, 4 dependency warnings in 1160.95 pytest seconds (1174.865 dashboard wall seconds), under the 1200-second budget. git diff --check passed.

**2026-09-22T19:50:19Z**

Pull request: https://github.com/engimatic-systems/cffg/pull/158 (base cff-1o72).
