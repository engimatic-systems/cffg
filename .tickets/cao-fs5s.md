---
id: cao-fs5s
status: closed
deps: [cao-0cla, cao-lty5, cao-0pbk]
links: []
created: 2026-11-24T05:48:30Z
type: feature
priority: 2
assignee: coopwilliams
parent: car-mps6
tags: [axiom]
---
# Reduce interaction rules on accumulated model-generated experiences

## Background in Plain English — WHY

The interaction model can learn several rules for the same situation. AXIOM periodically draws synthetic experiences from its current model and considers combining rules. Its decision uses the predicted reward and motion for each experience, with a penalty for changing the mixture frequencies.

## Outcomes / Acceptance Criteria — WHAT

1. Every 500 completed actions, draw 2,000 synthetic records from the model and retain all previous generated batches. Use posterior mixing concentration only where it exceeds reset concentration, draw Gaussian outputs with the release-projected expected covariance, and draw each category from its posterior mean. Save/restore the random key and accumulated records for exact continuation.
2. Freeze at most 2,000 proposed pairs from the release's sampled live sources, using soft mean categories and all candidate rows for neighbor selection. Match modal own and other identity, reward and switch. Use the original seeded values of never-born dormant slots for unmasked likelihoods, independently of the raw reset prior.
3. On the fixed generated sample set, select the contextual winner over every row without reward, switch or mixing weight. Score its two held-out heads only if its raw mixing concentration exceeds the reset prior. Subtract full Dirichlet mixing KL. Log the proposed pair, scores, active counts and decision. Transfer learned natural statistics through the exact host merger and accept nondecreasing scores for still-live pairs.
4. Confirm the frozen positive/negative golden deltas, exact ties and donor reset, a dormant contextual winner, phase-to-phase sample retention, reference handling of missing/invalid support and fixed-seed restart. Compare cached proposal scoring with a direct full-bank calculation. Exercise 5,000 slots / 2,000 generated records or bound the actual deployment cost explicitly; no formal benchmark until coordinated.

## Methodological Suggestions — HOW

Use the Learnable or checkpoint feature and Runtime numerical or status invariant change paths for this host experiment. The checkpoint retains the reduction history alongside the exact separate bank. Capture at least one sparse-support, both source dtypes, and a larger-bank case. JIT/vmap inside the graph, graph ledger terms, IR, and bundle format changes are N/A. Review the pinned upstream rmm.py proposal and scoring rules and HybridMixture.sample directly. Use cached context winners so a rejected proposal need not reevaluate the entire bank.

## Future work

car-mps6 integrates this operation into the causal model and car-zieb owns a full episode campaign. Generic compiled merge cancellation is tracked in cao-ar2d. The release-specific scheduler uses the exact host component merge.

## Definitions

An rMM is the recurrent mixture model for object interactions. A pseudo-observation is synthetic data sampled from that model. The held-out heads here are the reward and motion-switch categories. KL denotes Kullback–Leibler divergence of the posterior mixture weights from their reset prior.


## Notes

**2026-11-24T06:03:10Z**

Implemented reduce_interactions and reduce_on_tape in src/cffg/experiments/axiom_reduction.py. A separate key and the complete generated history round-trip via ReductionState.state_dict; save ComponentBank.state_dict(exact=True) and the factory key/config alongside it. Soft-neighbor queries use all six head means. Both held-out heads are omitted from context selection. Raw mixing mass above the reset prior supplies the emission mask and the KL. The optional initialized_view supplies virgin means and category values to unmasked queries. Proposed merges use the exact host lifecycle with a decision log for accepted, rejected, inactive and invalid proposals.

Source check: b99f625 hybrid_mixture_model.py sample draws Gaussian noise using sampling_view(Psi)/(nu-d-1) followed by a per-component covariance eigenvalue repair. Likelihoods instead use the minimum normalized eigenvalue across the full effective model. Each proposal refreshes the two changed row minima and compares the global minimum before deciding. Changed repair requires a new full competition including untouched, virgin and retired rows. A rejected proposal discards the candidate cache. The scalar NumPy oracle exposes a merge that changes the winner from untouched slot 2 to untouched slot 3 in f32 and f64, with opposite verdicts for two observed outcomes.

Scaling: at 2,000 records and 5,000 slots, a full context competition is 10 million record/row evaluations, blocked at 512 records by 64 slots. Changed repair in all 2,000 proposals would require another 20 billion evaluations. A 40,000-record accumulated tape multiplies those counts by 20. With unchanged repair each proposal evaluates two changed columns; acceptance refreshes at most 128 columns and merges cached leaders. The separate soft-neighbor search compares at most 4,000 sources to 5,000 destinations using the six head vectors. No dense record-by-slot matrix is retained. Full-size tests exercise a global repair change and the 2,000-entry frozen list with 1,999 stale entries. Saturated pair throughput and campaign timing remain unmeasured.

Parent owns the aggregate dashboard/short suite at the integrated head. JIT graph iteration, graph BFE/GFE, planner scoring, training writes and public IR are outside the host reducer. Host score comparisons use float64; floating-point near-tie identity with the source float32 kernels has not been established.

**2026-11-24T06:04:01Z**

Final focused gate on 8407736 plus this change: uv run --frozen --no-sync pytest -q -m "slow or not slow" tests/test_axiom_reduction.py passed all 20 cases in 30.81 s, including the real K=5,000 full rescore, seven-dimensional K=5,000 sparse pass, frozen accept/reject/tie, independent soft-neighbor fixture, sampler and nonzero low-word exact restart. The parent will run the one combined short-suite gate.
