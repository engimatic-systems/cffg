---
id: cpr-oc8z
status: closed
deps: []
links: []
created: 2026-11-22T01:59:00Z
type: bug
priority: 1
assignee: coopwilliams
tags: [premerge, numerics, hybrid]
---
# Preserve normalized float32 mixture responsibilities and a correct certified free energy

## Background in Plain English — WHY

On the PR 186 head (80ae1f2), a two-component mixture can report a converged free energy about 50% above its analytic answer. With two identical unit Gaussians, equal component priors and a unit-variance observation at 10000, the selector marginal is [0.5, 0.5]. The factor statistics report [1, 1] and mark those weights proper. At large absolute log scores float32 can round away the small correction in the log of the sum. PR 174 routes both mixture ledger variants through this normalization.

## Outcomes / Acceptance Criteria — WHAT

1. On the scalar fixture, public Engine inference, mixture statistics, and free energy agree with the identical-component Gaussian oracle at dtype-appropriate tolerance. A valid selector posterior sums to one.
2. With unequal component priors and identical components, a shared likelihood offset preserves the prior odds even when the offset exceeds float32 resolution for the log odds.
3. The prior-mixture and affine-mixture rules agree across selector messages, sufficient statistics used for learning and node energy. Exact-zero support stays zero; impossible or invalid inputs report the documented failure.
4. Cover both precisions, hostile observation offsets and a larger component count. Recheck the normal PR gate after the final numerical edit.

## Methodological Suggestions — HOW

Architecture rows: New factor or rule family and Runtime numerical or status invariant. Inspect src/cffg/mixg.py (_exact_probability, mix_stats, mixlin_stats, mix_energy and mixlin_energy) and mixture_evidence.py. Evaluate centering evidence on live components before adding the incoming selector message, then use a stable normalization on the combined supported scores. Retain independent value oracles. Check host/readout versus JIT/ledger, flooding/scheduled and single/vmap behavior where applicable. Serialization is N/A.

## Future work

This ticket gates landing the cumulative stack. The separate known unobserved linear-Gaussian float32 cancellation issue remains cff-d63w. The fixed float64 reactor reference contains no regime mixture.

## Definitions

A responsibility is a posterior probability assigned to a mixture component. The free-energy ledger adds the local contributions used for convergence and policy scoring. An oracle is an independently calculable answer.


## Notes

**2026-11-22T02:00:15Z**

The same public-API reproducer was run from isolated source archives with the same CPU/JAX environment. Main a4c6d1b and the immediate pre-174 base b49e9d3 both returned 24,999,998. The #174 head 416d1a1 and the stack tip 80ae1f2 both returned 37,499,996. All four reported converged, proper=True and [1, 1]. Thus #174 introduced the ledger regression and inherited the bad factor readout. Reproduce from a checkout with uv run --no-sync python ~/agent-share/cffg-premerge-review/cffg-pr174-mixture-repro.py 80ae1f2. Raw four-revision outputs sit beside the reproducer.

**2026-11-22T02:30:03Z**

Implementation handoff. The New factor or rule family and Runtime numerical or status invariant rows point to the common mixture rules. Check prior and affine selector messages, host factor statistics and predictive weights, JIT node energies and the public convergence ledger. Exercise scheduled and flooding inference and batched members with different support. Cover float32 and float64, equal and unequal priors, large common evidence, exact log zeros, empty support, invalid input and a larger component count. Existing improper and vacuous cavity tests cover the Gaussian admission rules. Serialization and checkpoint decoding are N/A. Cooper owns the final stack gate.

**2026-11-22T02:34:32Z**

Public float32 comparison at observation 10000, using source archived from 80ae1f2 and the repaired worktree in the same uv environment. Auto scheduling and forced flooding agreed on all three cases. For equal prior weights, responsibilities changed from [1, 1] to [0.5, 0.5] and free energy from 37,499,996 to 24,999,998 (Gaussian oracle 25,000,001.2655). For prior weights [0.2, 0.8], responsibilities changed from [0.13533528, 1] to [0.2, 0.8] and free energy from 26,691,690 to 24,999,998 (same oracle). For an affine mixture with unit input, slope and noise, responsibilities changed from [0.13533528, 0, 1] to [0.2, 0, 0.8] and free energy from 17,418,530 to 16,666,669 (oracle 16,666,668.1349). All repaired runs converged, reported proper statistics matching the public selector marginal, and returned the same value from Engine.bfe and the final trace entry.

**2026-11-22T02:35:13Z**

Implemented support-aware centering of selector log weights and component evidence before combining them. Both mixture families now use max-shifted softmax for predictive probabilities, learning statistics and node responsibility weights. The evidence integrals and absolute Gaussian terms retain their existing definitions. All six new regression cases failed before the fix and passed afterward. Regression coverage includes batched 32-component cases in float32 and float64 with an unsupported likelihood winner, ordinary and extreme observations, exact zeros and invalid selectors. Validation after the final edit: uv run --no-sync pytest -q tests/test_mixg.py tests/test_mixture_evidence.py tests/test_hybrid_readouts.py tests/test_switching_kf.py --tb=short passed 69 tests with 2 deselected in 113.27 s. The original public reproducer also passed its numerical comparison. Cooper owns the final full-suite stack gate.
