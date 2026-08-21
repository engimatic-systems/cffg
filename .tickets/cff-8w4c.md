---
id: cff-8w4c
status: in_progress
deps: [cff-jq9p, cff-888k]
links: []
created: 2026-08-21T03:20:06Z
type: task
priority: 1
assignee: coopwilliams
parent: cff-l5ag
tags: [rslds, validation, statistics]
---
# Run a shared recurrence-prior development study

## Background in Plain English — WHY

The frozen paper-adapter chains mixed poorly under a very weak recurrence
prior. A stronger prior may prevent extreme gates and component collapse, but
it could also erase real source-dependent switching or make the comparison
unfair if CFFG and the adapter receive different priors.

A development study must compare a small declared set of shared priors after
claim-bearing mixing and collapse evidence exist. The study reports every
candidate. It does not silently choose a production value.

## Outcomes / Acceptance Criteria — WHAT

1. One recurrence-prior object is constructed per candidate and supplies
   byte-identical mapped arrays to CFFG and the paper adapter.
2. The study evaluates the baseline and a small predeclared set of moderate
   recurrence strengths on both dataset families and both initialization tiers.
3. Only calibration or new development seeds are used; registered final seeds
   are rejected before data generation.
4. Every candidate retains claim-bearing mixing, raw parameter mixing,
   component-collapse evidence, task adequacy, convergence status, and oracle
   results for both implementations.
5. The artifact reports every candidate and contains no selected-prior field or
   hidden ranking that can be mistaken for a frozen decision.
6. Classic adequacy cannot be traded away to improve heterogeneous mixing, and
   source-dependent recovery cannot be averaged across failed source rows.
7. Commands, runtime identities, prior-array digests, result schemas, and every
   artifact file are independently reproducible and manifested.
8. Focused compact tests and one staged development arm pass before broader
   study arms run; no completed cff-oyie artifact is changed.

## Methodological Suggestions — HOW

These suggestions are optional. The implementer may choose another sound
method that satisfies the outcomes.

- Start with the existing baseline plus moderate recurrence strengths such as
  0.1, 0.3, and 1.0; record the declared set before fitting.
- Reuse the dual-protocol runner and build a small study orchestrator rather
  than duplicating fitting code.
- Funnel execution through one family, tier, and candidate before expanding to
  every declared coordinate.

## Future work

`cff-l5ag` reviews the complete study and decides whether diagnostics alone,
one shared prior, or a separate sampler remedy can restore eligible reference
inference. A chosen prior must enter a new calibration and immutable freeze;
this ticket does not select or freeze it.

## Definitions

- **Recurrence prior:** The probability distribution placed on switching-gate
  weights before data are observed.
- **Shared prior:** The same mathematical prior, mapped from one source object,
  given to CFFG and the independent adapter.
- **Task adequacy:** Absolute evidence that a fitted model still performs the
  registered NASCAR prediction, switching, and rollout tasks.

## Notes

Architecture handoff: use `docs/ARCHITECTURE.md`'s canonical rSLDS consumer
validation boundary and the sibling experiment's protocol/freeze boundary.
CFFG factor equations, IR, compiler, VM, schedules, kernels, ledger, and engine
status semantics are N/A.

Duplicated surfaces: both families, both initialization tiers, both
implementations, baseline and candidate priors, calibration and future final
protocols, isolated and modern runtimes, checkpoints, manifests, and aggregate
recomputation must stay aligned. Exact-zero support, nearly deterministic
gates, non-finite draws, and one failed coordinate must remain explicit.
