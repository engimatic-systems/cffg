---
id: cfp-okq2
status: closed
deps: [cfp-ru3t]
links: [cfp-rqcd, cfp-9806]
created: 2026-11-21T21:30:43Z
type: epic
priority: 2
assignee: coopwilliams
---
# Recognize directed nonlinear Gaussian chains and solve them through Engine.run

## Background in Plain English — WHY
The 19-transition reactor on the PR 185 branch chooses each local Taylor model using only evidence from the past and the current observation. A directed reference computes the same message root in two passes. The production engine still builds a coupled nonlinear solver for each fresh model. Other graphs, including an acyclic pair of nonlinear sensors that depend on each other, genuinely need coupled inference.

This is the integration and automatic-dispatch ticket. cfp-hlqd supplies the dependency recognizer and cfp-ru3t supplies the executor and message-parity evidence. Reuse their artifacts for criteria 1–3. This ticket owns enabling eligible automatic dispatch through Engine.run, public documentation and performance qualification. The control experiment has no dependency on this rollout.

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

## Notes

**2026-11-22T19:47:30Z**

Implementing on the updated PR 189 base. Architecture: Runtime numerical or status invariant. Auto single CPU Engine.run is in scope. Explicit schedules, forced trust-region, transformations, Colony routing and checkpoint reconstruction checked separately. Testing compiler-derived topology, directional slots and candidate/successor objectives against retained PCGym data.

**2026-11-22T20:53:49Z**

Integrated clean code head 8c4993a: tools/qualify_directed_reactor.py passed 9 retained and 590 held-out decisions from 10 seeds, all using the public directed method. All 59 archived coupled-solver comparisons passed; deliberately stale data was rejected by both independent checks. Warm held-out median 7.230 ms; cold after engine preparation 208.64 ms; process launch to first action 1.402 s with disk cache disabled, preparation 0.819 s. Maximum held-out absolute action difference vs archived coupled solver 2.13e-6 K. Scaling from 6/19/194 transitions: warm median 3.97/6.96/48.41 ms. Among supported fields these are replay/latency results, not closed-loop return comparisons. Logs and raw JSON under ~/agent-share/auto-dispatch-plus-paired-reactor-neon/. The final full gate passed: 2755 passed, 194 opt-in deselected, four warnings. Measured wall time 1203.506s. Moved only the duplicate length-2 float64 hardening parameters (2.49+2.46s) to slow for an estimated default 1198.556s. Retained float64 length 20 and float32 length 2 with both tolerances; verified collection. Reused the passing run per session rule.
