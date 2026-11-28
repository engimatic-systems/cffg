---
id: cml-hj19
status: in_progress
deps: []
links: []
created: 2026-11-28T04:28:22Z
type: feature
priority: 2
assignee: coopwilliams
parent: cff-cj5h
---
# Expose opt-in structural model lint and DELTA run warnings

## Background in Plain English — WHY

A valid planning graph may use ordinary KL-control where its author intended an observation-epistemic term. A unary fixed-parameter mean-field declaration also does no work. A run can explicitly disable point-mass hardening. Authors need a way to inspect those choices while building a model.

## Outcomes / Acceptance Criteria — WHAT

1. Meet cff-cj5h criteria 1–20, 81–86 and 91. A pure JAX-free cffg.core.lint(template) returns ordered immutable records with stable codes, named owners and explanations. Expand plates to match compiled names.
2. Identify each unclamped matrix_cat output facing a categorical goal without output P-substitution. Say that the pair has no observation P-substitution epistemic term and that ordinary KL-control is valid. Skip DATA ports and correctly substituted outputs.
3. Report a mean-field declaration only when the factor belief contains one ordinary variable and no random parameter edge. A categorical likelihood with a Dirichlet parameter edge has two random variables and can use mean-field meaningfully.
4. The compiler and Engine.run each accept an opt-in lint flag. Emit structural findings only after successful compilation. Warn for explicitly disabled DELTA hardening with a DELTA-constrained arena. Preserve default warnings, validation errors, arena contents, scores and execution.
5. Verify deterministic 256-pair ordering, scoped plate names, absence of JAX imports, positive and negative cases, default and opted-in compiler and Engine calls. Document public entry points.

## Methodological Suggestions — HOW

Use the Architecture change-path index rows for template structure and runtime status as references. The host structural pass owns the lint records. The compiler forwards those findings. Engine knows the actual hardening setting. Numerical host/JIT, single/vmap and checkpoint formats are unchanged; finite probability arrays need not be opened for lint.

## Future work

The ledger explanation and environment action-selection policy have separate tickets under cff-cj5h.

## Definitions

P-substitution changes the observation objective. KL-control denotes the ordinary preference-based objective. Mean-field splits a joint belief into independent parts. DELTA hardening projects a categorical belief to a single outcome. DATA denotes an evidence port.


## Notes

**2026-11-28T04:33:46Z**

Implementation and focused verification are complete. The structural pass reads expanded declarations in factor-name order. It names each individual goal/likelihood pair and skips unary likelihoods connected to a random Dirichlet parameter when diagnosing redundant mean-field splits. The compiler emits findings after successful assembly. Engine.run independently checks the resolved number of DELTA hardening sweeps when lint=True. All defaults remain quiet. Tests/test_model_lint.py, tests/test_core_imports.py and tests/test_readme.py passed all 12 selected checks, including the no-JAX subprocess, 256 scoped pairs, numeric parity, validation failures and all warning controls. Commands: PYTHONPATH=src uv run --frozen --no-sync pytest -q tests/test_model_lint.py tests/test_core_imports.py tests/test_readme.py; tk dep cycle; git diff --check. The parent coordinates the full short-suite gate before opening a review PR.
