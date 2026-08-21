---
id: cff-jq9p
status: closed
deps: []
links: []
created: 2026-08-21T03:20:06Z
type: task
priority: 1
assignee: coopwilliams
parent: cff-l5ag
tags: [rslds, validation, statistics]
---
# Define claim-bearing adapter mixing diagnostics

## Background in Plain English — WHY

The paper adapter currently decides whether its four chains mixed by comparing
43 raw parameter values. Equivalent rSLDS models can use different regime
labels, latent coordinates, or decision-list coefficients. Raw disagreement can
therefore reject chains that make the same predictions, while a pooled maximum
can hide which claim failed.

The parity claim needs diagnostics on quantities that keep their meaning after
those equivalent representations are aligned. Raw parameter diagnostics should
remain available as secondary debugging evidence.

## Outcomes / Acceptance Criteria — WHAT

1. The adapter reports named split R-hat and effective-sample-size results for
   label- and affine-aligned claim quantities.
2. The claim quantities include predictive fit, aligned latent and regime
   recovery, and aligned transition probabilities used by the registered
   classic and heterogeneous outcomes.
3. The existing raw 43-value parameter projection remains preserved and is
   labeled as secondary evidence rather than eligibility authority.
4. The diagnostic schema, variable order, alignment policy, and content digest
   are deterministic and recorded in every new result.
5. Tests prove invariance under every four-regime label permutation and under
   decision-list coefficient changes that preserve destination probabilities.
6. Tests prove that a changed transition function or one stuck chain fails
   loudly instead of being aligned away.
7. Both dataset families, both initialization tiers, calibration and future
   final execution, raw checkpoints, and aggregate recomputation use the same
   diagnostic authority.
8. Completed cff-oyie artifacts remain immutable, and no registered final seed
   is inspected or run.

## Methodological Suggestions — HOW

These suggestions are optional. The implementer may choose another sound
method that satisfies the outcomes.

- Build the authority in the modern experiment runtime from retained draws.
- Reuse the existing label, affine, and destination-probability alignment
  functions instead of changing the isolated sampler.
- Preserve per-variable values as well as family extrema so a failure names its
  owner.
- Version the claim projection separately from the raw coefficient projection.

## Future work

`cff-888k` adds component-collapse evidence in parallel. `cff-jq9p` and
`cff-888k` together unblock the shared-prior study `cff-8w4c`. `cff-l5ag` owns
the final diagnostic and remedy decision.

## Definitions

- **Claim-bearing diagnostic:** A mixing check on a quantity used to support
  the registered scientific conclusion.
- **Decision list:** The ordered binary gates used to express one categorical
  transition distribution.
- **Effective sample size:** An estimate of how many independent samples a
  correlated chain represents.
- **R-hat:** A comparison of within-chain and between-chain variation; values
  near one indicate agreement.

## Notes

Architecture handoff: use `docs/ARCHITECTURE.md`'s canonical rSLDS consumer
validation boundary. IR, compiler, VM, schedule, kernel, ledger, serialization,
learnable, and engine status changes are N/A.

Duplicated surfaces: classic and heterogeneous families, shared and native
initialization, four chains, calibration and future final protocols, raw and
claim projections, checkpoint resume, and aggregate recomputation must stay
aligned. Exact-zero support, nearly deterministic gates, label-switched chains,
non-finite draws, and one stuck chain must fail loudly. The isolated reference
runtime is float64-only, so float32 is N/A.

**2026-08-21T04:08:51Z**

Implemented and integrated in experiment commit e4db972 (source fb13bff). Claim-bearing paper-adapter mixing now uses six named label/affine/decision-list-invariant quantities with per-variable R-hat/ESS/stuck-chain status, deterministic schema/version/digest, and aggregate/checkpoint validation. The legacy 43 raw parameters remain versioned secondary debugging evidence and never control eligibility. All K=4 label permutations, affine reexpression, decision-list reexpression, genuine transition changes, stuck chains, and non-finite draws are covered. Combined integration validation contributed to 61 focused passes, 13 protocol passes, and 47 surrounding passes; two unavailable-vendored-source tests were deselected and no chain or final seed ran.
