---
id: cff-garu
status: closed
deps: []
links: [cff-l7y4]
created: 2026-08-09T05:13:04Z
type: bug
priority: 2
assignee: coopwilliams
parent: cff-ahps
tags: [rslds, numerical-robustness, scheduling]
---
# Stabilize inner rSLDS solves after repeated parameter replacement

## Background in Plain English

The observed-only fitter from cff-l7y4 alternates hidden-state inference with point-parameter replacement. On the compact NASCAR pilot (T=240, K=4, latent d=2, observed p=10, seed 101), two replacement coordinates followed by final inference converge under the registered damped schedule. After a third replacement, the next unchanged-protocol inner solve exhausts its iteration budget: the Bethe free-energy change becomes small while the absolute message-delta certificate remains large.

This boundary does not invalidate the preregistered cff-l7y4 protocol, whose outer budget is exactly two. It does prevent callers from assuming that additional finite coordinates monotonically improve or preserve solvability. This ticket owns diagnosing and stabilizing that repeated-replacement path without weakening the public converged / budget_exhausted / diverged distinction. cff-l7y4 owns the fitter and comparison; cff-r6bl owns cold setup performance.

## Design

Use the Runtime numerical or status invariant change-path row. Retain the pilot mechanism as a bounded regression: preserve the observations, initializer, immutable priors, parameter-write cadence, schedule, damping, and tolerances while recording per-phase Bethe free energy, message delta, recurrence coordinate status, and parameter scales before and after the third replacement.

Determine whether the failure is a scale-dependent certificate, a recurrence mean-field cycle, or a Gaussian-message instability. Fix the owning layer and keep exact-zero support, float32, and long-horizon behavior explicit. Do not tune the cff-l7y4 preregistration or relabel exhausted inference as converged.

## Acceptance Criteria

1. A compact deterministic regression reproduces the post-third-replacement budget exhaustion using observations and an explicit key, without receiving generating states or parameters.
2. The regression records enough per-phase evidence to distinguish a recurrence-coordinate failure, a non-finite or indefinite Gaussian belief, a finite message cycle, and a convergence certificate whose scale is inappropriate.
3. The corrected path completes at least five parameter replacements and a final hidden-state solve with RunStatus.CONVERGED under a preregistered bounded budget, or returns RunStatus.DIVERGED with a falsifiable mathematical reason when the coordinate map has no supported fixed point.
4. The correction does not weaken the engine rule that convergence requires finite Bethe free energy and the documented message/status certificate.
5. Exact-zero regime support remains literal, improper beliefs remain loud, and both float32 and x64 use dtype-derived guards.
6. The compact regression runs in the short suite; larger NASCAR confirmation remains in the sibling experiment.
7. docs/ARCHITECTURE.md records the owning convergence or scheduling mechanism, and docs/LIMITATIONS.md records any accepted repeated-coordinate boundary.

## Future work

This ticket does not change the observed-only parameter coordinates, add a
joint parameter objective, or optimize canonical cold setup. cff-l7y4 owns the
first fitting and comparison contract, cff-ahps owns any future joint
variational objective, and cff-r6bl owns repeated-slice lowering and setup
cost.

## Definitions

- **Inner solve:** One hidden-state and regime inference run between parameter
  replacements.
- **Message-delta certificate:** The finite message-change condition the engine
  requires in addition to a stable Bethe free energy before reporting
  convergence.
- **Parameter replacement:** Recomputing a one-sequence posterior from fixed
  priors and installing its point parameters into a fresh state.
- **Repeated-replacement path:** More than two consecutive inference and
  parameter-replacement coordinates on one observed sequence.


## Notes

**2026-08-09T05:13:25Z**

Intelligible-ticket review, 2026-08-08. Capability owners are explicit: this ticket owns the repeated-replacement inner-solve mechanism; cff-l7y4 owns fitting and comparison; cff-r6bl owns cold setup. The bounded wiggle scan returned no retained candidates.

**2026-08-09T14:03:08Z**

Implementation handoff, 2026-08-09.

Root cause: the post-third-replacement failure is a real finite recurrence/mixture coordination cycle, not an improper Gaussian belief, non-finite state, failed recurrence coordinate, or scale-only message certificate. On the pinned T=240, K=4, d=2, p=10, seed-101 pilot, the first phase-four attempt at damping 0.9 retained finite FE and proper beliefs but exhausted 2,000 sweeps with a final raw message delta of 1.0462583226.

Correction: fit_rslds now retries only finite budget-exhausted inner solves. Every retry starts from the identical fresh parameter-bound state and doubles the convex update fraction, preserving the fixed point, exact support, schedule, tolerance, and FE/message certificate. The registered sequence 0.9 -> 0.8 -> 0.6 converges on phase four; the fixture completes five parameter replacements and a terminal hidden-state solve. Non-finite, infeasible, and diverged attempts remain loud and do not retry. The returned state is now certified after the final parameter write. RSLDSFitResult.phases records attempt damping/status/recurrence status, FE and delta tails, Gaussian propriety, state finiteness, and parameter scales before and after replacement.

Validation: the exact regression passed in 103.87 seconds. The final default short suite at commit db995e3 passed 1,334 tests with 39 deselected in 509.51 seconds and refreshed ~/agent-share/cffg-tests.html. git diff --check passed. docs/ARCHITECTURE.md, docs/LIMITATIONS.md, and README.md now describe the retry authority, remaining finite-retry boundary, diagnostics, and terminal-state contract.

Next: cff-nkbj may consume the stabilized longer fitting cadence. cff-r6bl remains the separate cold-setup and repeated-slice performance owner. Gotcha: free_energies now includes the terminal solve, and phases marks that final solve with coordinate_replaced=False.

**2026-08-09T14:03:44Z**

Post-handoff wiggle audit, 2026-08-09. All 35 retained candidates have one intended reading in context: canonical=the named cold NASCAR scale; fixed=constant priors or the repaired defect; preserve=semantic invariants; public=the documented status contract; register=the preregistered scientific protocol; implement=the delivered code change; support=literal probability support; complete=the named execution or acceptance set; remaining=the deliberately accepted finite-retry boundary or preserved behavior; sufficient=the enumerated diagnostic evidence in criterion 2; certify=the runtime FE/message/status certificate; check=the named automated or propriety check; converge=RunStatus.CONVERGED; correct=the defect correction; evidence=diagnostic or validation records; exact=literal support or pinned fixture identity; fixed_point=the mathematical coordinate-map fixed point; pass=the named test gate; reproduce=the deterministic failing fixture; robust=the AGENTS.md numerical contract; stable=absence of the diagnosed finite cycle under the retry cadence; test=the named test artifact or run; validate=the named verification action; fit=observed-only parameter estimation; infer=hidden-state inference; performance=cold setup latency; scale=parameter magnitude, certificate scaling, or named horizon; small=the numeric FE change; may=downstream possibility; explicit_owner and named_ticket_handoff=the ticket or component named in each sentence; any=the locally bounded accepted-boundary or future-objective set; at_least=the inclusive five-replacement lower bound; every=each bounded retry; only=the stated observed-only or retry scope and exclusions. Capability ownership remains explicit: cff-garu owns the repeated-coordinate retry and diagnostics, cff-l7y4 owns the original fit/comparison contract, cff-r6bl owns cold setup, cff-nkbj owns multi-seed parity, and cff-ahps owns a future joint objective.

**2026-08-09T14:04:00Z**

Additional candidates introduced by the classification note retain these meanings: deliver=the completed code and documentation artifacts; estimate=observed-only point-parameter estimation; parity=cff-nkbj's future literature comparison; verify=the named verification action; intended=the explicit contextual meaning map; all=the enumerated 35-candidate set in the prior note; read=the contextual interpretation of a candidate.

**2026-08-09T14:04:55Z**

Draft PR: https://github.com/engimatic-systems/cffg/pull/60, targeting agent/cff-l7y4-observed-only-rslds (PR #59).

**2026-08-13T05:07:28Z**

PR review correction (2026-08-12): split the observation-only initialization responsibility out of rslds_fit.py into rslds_init.py. The new module owns observation validation, initialization protocol/result types, PCA/transition clustering/local AR construction, and initialization shape/propriety validation; rslds_fit.py remains the host coordinate-loop and status/result owner. The files are 350 and 692 lines respectively, both below the 1,000-line limit. cffg.models exports are unchanged, and the former direct rslds_fit imports remain compatible. The 10 focused fitter tests passed in 130.67 seconds; final short-suite validation is pending on the committed refactor.

**2026-08-13T05:16:22Z**

Final PR-review validation (2026-08-12): the default not-slow suite at code commit a961905 passed 1,334 tests with 39 deselected and zero failures in 509.97 seconds; the shared test dashboard was refreshed. No code changed after that run.
