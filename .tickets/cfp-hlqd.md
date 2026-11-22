---
id: cfp-hlqd
status: open
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
