---
id: cff-8w4c
status: in_progress
deps: [cff-jq9p, cff-888k, cff-2wb6]
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

**2026-08-21T06:51:44Z**

Code-only study implementation completed in the sibling experiment and integrated on `rslds-repair-integration`: d00b776 adds the declared four-strength/two-family/three-seed study, 4ae9a9c binds it to the real cff-tlh3 authority, and 7bb020d reuses the six baseline classic CFFG fits as cff-3v7q replicate-0 checkpoints. Candidate strength reaches both CFFG and adapter tiers from one prior object with byte-identity digests. The command rejects the current pilot artifact and permits execution only after cff-tlh3 reports `complete_converged`, all 12 arms converged, no censoring/final seeds, and a derived budget. Code validation passed (106 prior-study tests plus 64 checkpoint/integration tests); no study fit, chain, or dataset ran yet.

Future eligible command:
uv run python -m cffg_nascar_validation.prior_study --protocol artifacts/rslds-dual-calibration/selected-protocol.json --scale-budget artifacts/rslds-selected-scale-outer-development/results.json --output artifacts/rslds-shared-prior-development

**2026-08-21T08:50:59Z**

Funneling corrected in experiment commit ed876ec before execution. The immutable/resumable stages are: `funnel` runs only baseline 0.001/canonical/seed5000 and treats a completed scientific miss as valid evidence; `screen --resume` evaluates all four strengths on both families at seed5000; `expand --resume` runs seeds5001/5002 for every candidate passing the predeclared two-family eligibility, claim-mixing, occupancy, and absolute-adequacy gates. No ranking or winner field exists. If none pass, the study ends `complete_no_eligible_candidates` without expansion. Orchestration/schema/provenance/checkpoint failures stop every stage. Validation: 120 passed, 3 environment-deselected; no fit/chain/dataset ran.

The funnel produces the two baseline classic seed5000 periodicity checkpoints. Baseline seeds5001/5002 are not run through the expensive adapter study if the baseline fails screening; cff-3v7q will obtain those four CFFG-only checkpoints through its separate development-fit path.
