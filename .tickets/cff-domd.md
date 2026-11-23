---
id: cff-domd
status: closed
deps: []
links: []
created: 2026-09-16T00:24:03Z
type: task
priority: 2
assignee: coopwilliams
parent: cff-1o72
---
# Keep the neural PR short suite within twenty minutes

## Background in Plain English — WHY

The completed neural PR short suite passed all 2,404 tests, but measured wall time was 1,279.975 seconds. Cooper authorized moving lower-priority tests out of the default suite using that run's measured timings, without rerunning solely for those selection changes.

## Outcomes / Acceptance Criteria — WHAT

1. Nine measured cases move to the opt-in slow suite, retaining their code and assertions.
2. The selected cases total 105.936 seconds, giving an estimated remaining wall time of 1,174.039 seconds, below 1,200 seconds.
3. Required gradient, sparse-support, float32 and uncomfortable-scale coverage stays in the short suite. Checkpoint variants retain representative coverage.
4. The measured passing run remains the validation authority; estimates are labeled separately and no replacement passing run is claimed.

## Methodological Suggestions — HOW

Use the Runtime numerical or status invariant change path for test selection. Move repeated checkpoint configuration/cadence/dtype variants and long behavioral roundtrips to slow. Validate collection and diff only. Production kernels, scheduling, serialization and energy semantics are unchanged; their duplicated execution surfaces are N/A for this selection-only change. The user's explicit reuse protocol applies to this final selection change.

## Future work

The final PR for cff-1o72 publishes the measured result and estimate. No further work is identified.

## Definitions

Short suite: pytest's default not-slow selection. Estimated runtime: measured suite wall time minus the measured duration of cases moved out of that selection.



## Notes

**2026-09-16T00:25:03Z**

Measured passing result at code head10e0615: 2,404 passed,58 deselected; pytest1,265.83 seconds, full wall1,279.975 seconds. The following9 cases passed in that run and now carry slow marks:

- test_functional_checkpoint_configuration_mismatch_is_atomic[min_ess]: 10.522 seconds
- test_functional_checkpoint_configuration_mismatch_is_atomic[refresh]: 10.617 seconds
- test_all_families_resume_identical_next_update[accumulate-float32-rslds]: 10.825 seconds
- test_all_families_resume_identical_next_update[accumulate-float64-rslds]: 10.147 seconds
- test_fitted_rslds_exports_complete_state_and_resumes: 11.345 seconds
- test_b_recovery_from_random_walk: 8.975 seconds
- test_optimizer_continuation_and_bundle_are_replayable_and_strict[True]: 15.736 seconds
- test_recognition_replay_batching_and_invalid_proposals[True]: 10.625 seconds
- test_trial_loop_matches_direct_engines_and_checkpoint_replays: 17.144 seconds

Measured case sum105.936 seconds; estimated remaining wall1,174.039 seconds (19m34.039s). This is an estimate, not a fresh run. Retained both nonlinear-gradient methods, float32 gradients/training/recognition, sparse GFE and32-regime checks, all neural scale checks, representative checkpoint configuration rejection, both rSLDS replacement dtypes and both Gaussian accumulation dtypes. The long behavioral/checkpoint paths remain opt-in.

Validation: uv run pytest --collect-only -q selects2,395 tests; compare against the passing dashboard inventory confirms exactly these9 removed and no added tests. git diff --check and tk dep cycle pass. The existing dashboard and runtime history continue to report the measured passing run. No rerun was performed solely for selection changes, as Cooper requested.
