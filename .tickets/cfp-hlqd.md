---
id: cfp-hlqd
status: closed
deps: [cfp-rh8k]
links: []
created: 2026-11-22T01:00:18Z
type: task
priority: 1
assignee: coopwilliams
parent: cfp-okq2
tags: [scheduling, performance]
---
# Recognize Taylor chains whose transition approximations depend only on upstream evidence

## Background in Plain English — WHY

On the reactor chain, cutting a transition leaves only upstream evidence at its input. Two nonlinear sensors sharing a hidden state violate this condition even when their graph is a tree. Identify the safe case before changing automatic execution. Start both workstreams from the shared handoff above PR #185 (cff-adaptive-runtime at f9df9d6), including the directed reference in 265ff14. Each workstream can branch independently.

## Outcomes / Acceptance Criteria — WHAT

1. A structural analysis of compiled roles, directed dependencies, model declarations and eligibility returns the supported ordered chain independently of model names or insertion order.
2. The analysis accepts a renamed eligible chain and rejects interacting nonlinear sensors, unsupported model families and shared uncertain parameter couplings. Its explicit supported envelope names noise, masks, goals and transition methods.
3. Existing explicit and automatic scheduling behavior remains stable until the executor and its qualification are ready. Record which remaining preconditions need runtime checks on actual inputs.

## Methodological Suggestions — HOW

Start at docs/DERIVATIONS.md D13.5–6, schedule.py, and test_mutually_changing_cavities_match_independent_site_root. Architecture row: Runtime numerical or status invariant. Inspect auto, explicit and prebuilt schedules, structural fingerprints and serialization round trips. Analysis is structural; host/JIT numeric checks, convergence and Colony dispatch belong to execution. Sparse discrete support is outside the Gaussian family; rank-deficient goal information is in scope.

## Future work

The executor consumes the eligibility result; cfp-okq2 owns production qualification and automatic selection.

## Definitions

A cavity contains the information reaching a factor from its neighbors. Taylor linearization uses a function value and its derivative at that cavity mean.

## Notes

**2026-11-22T19:48:09Z**

Change-path index: Runtime numerical or status invariant. This change reads sealed Gaussian rows and slot maps. Serialization and insertion order tested by recompile. Auto/explicit/prebuilt scheduling, fingerprint, Engine/Colony, host/JIT numerical checks and ledger execution are N/A here. Positive noise, proper prior, finite functions and observed values, masks and finite calculations need execution-time checks; a rank-deficient conditioned linear goal is structurally valid. Exact zero support is discrete and outside scope; missing observations and vacuous reverse messages are valid. The analyzer reads integer indices only and scales with compiled rows. No precision-specific numeric thresholds.

**2026-11-22T19:52:42Z**

Implemented the compiled-role recognizer in directed_chain_plan. It walks from the prior by Gaussian gather/scatter slots, identifies the equality leg facing each neighbor, and accounts for all factor rows. The envelope has one square Taylor transition group, one state dimension, one linear conditioned emission per site and fixed parameters with per-row masks. Function capability is supplied by 8093e1f. Checked the renamed, reordered, bundle-recompiled model; 19-transition retained reactor; and structural negatives including a reversed transition, interacting sensors, fixed vs learned ties, disconnected components and mixed families. Validation: uv run --frozen --no-sync pytest -q tests/test_directed_chain_plan.py tests/test_cstr_runtime.py::test_recent_validation_structure_and_covariances_are_not_the_short_example (11 passed). The parent owns auto dispatch, documentation, dashboard and full gate. The numerical executor checks live state, parameters, conditioned values, masks, proper moments, and finite updates.
