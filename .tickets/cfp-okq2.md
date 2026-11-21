---
id: cfp-okq2
status: open
deps: [cfp-rh8k]
links: []
created: 2026-11-21T21:30:43Z
type: task
priority: 2
assignee: coopwilliams
---
# Recognize directed nonlinear Gaussian chains and solve them through Engine.run

## Background in Plain English — WHY
The 19-transition reactor on the PR 185 branch chooses each local Taylor model using only evidence from the past and the current observation. A directed reference computes the same message root in two passes. The production engine still builds a coupled nonlinear solver for each fresh model. Other graphs, including an acyclic pair of nonlinear sensors that depend on each other, genuinely need coupled inference.

## Outcomes / Acceptance Criteria — WHAT
1. A conservative schedule recognizes an eligible normalized Gaussian chain from its compiled factor and cavity dependencies, independently of factor names. Unsupported cases retain the current route. Include the two interacting nonlinear sensors as a negative control.
2. A supported CPU solve consumes live parameters, data masks and goals, respects budgets and explicit schedule choices, and returns normal messages, readouts, objective metadata and truthful method-specific status through Engine.run. Cover Engine versus Colony and derivatives versus inference explicitly.
3. At the same returned state, compare both directional updates, raw and canonical successors, local status, free energy, posterior and action with the stock engine and with the independent directed reference. Preserve exact-support and failure reporting.
4. Measure new-process time with an empty disk cache, preparation cost, warm plans, changed data and horizon scaling. Publish the whole-process time to first action. The subsecond reference is a comparison point, not an established production guarantee.
5. State the envelope before enabling automatic dispatch. Include missing observations, singular goal information, a vacuous prior, nonfinite physics and nonpositive noise, float32 and float64, larger horizons and dimensions. Fail or fall back explicitly where the proof or numerical accuracy ends.

## Methodological Suggestions — HOW
Start with DERIVATIONS D13.6 and tools/benchmark_cstr_chain.py. Architecture change paths are schedule construction and runtime numerical/status invariants. Analyze the forward prediction and backward information dependencies separately. Keep a frozen-surrogate objective label and distinguish a solved root from a derivative of the adaptive objective.

## Future work
Hybrid mixtures, loopy graphs and uncertain parameter learning require separate mathematical support.

## Definitions
An input cavity is the information arriving from the neighboring factors on the other side of an endpoint. A directed solve computes one ordered pass in each direction. A surrogate is the local Gaussian model used for the nonlinear transition.
