---
id: cff-5xdy
status: closed
deps: [cff-nyad, cff-7kfk]
links: [cff-ii9t]
created: 2026-09-02T04:17:42Z
type: chore
priority: 1
assignee: coopwilliams
parent: cff-ws3c
tags: [engine, architecture, lifecycle, refactor]
---
# Extract state, data, and functional-mu3 lifecycle services

## Background in Plain English — WHY

At authoring snapshot e488df3, `src/cffg/engine.py:323-915` is the scope anchor
for initial state, observation binding, functional-mu3 preparation/update,
diagnostics, and epoch scores; it is not a promised future line map. The
implementation starts from the exact `cff-sf29` engine baseline. These
lifecycle operations are mixed with public construction and the run loop even
though they do not decide convergence.

This ticket gives ordinary state/data preparation and functional-mu3
preparation explicit internal owners. `Engine` construction stays public.
Engine and Colony delegate final status to the shared runtime result owner.

## Outcomes / Acceptance Criteria — WHAT

1. `cff-nyad` is closed with one schedule program using shared executors.
2. A state lifecycle owner accepts immutable arena/default parameters and
   returns a complete initialized `State`; a data-binding operation accepts a
   `State` plus named observations and returns a bound state.
3. A separate functional-mu3 lifecycle owner accepts only mu3 plan/state/key
   views and returns prepared stochastic state, next key, epoch statistics, or
   diagnostics. Ordinary factors are not forced through this interface.
4. Public `Engine.__init__`, `Engine.init_state`, `Engine.bind_data`, mu3 public
   methods, and diagnostics remain compatible delegates.
5. PRNG key consumption, cache identity, retained moments, status-aware mu3
   state, observation masks, parameter defaults, and readout shapes retain the
   `cff-sf29` baseline, including the #107 repairs present in its history.
6. The affected rows are **Learnable or checkpoint feature**, **Runtime
   numerical or status invariant**, and the mu3 portion of **New factor or rule
   family**. **New IR field** is N/A because state and plan schemas do not
   change.
7. Flooding/scheduled and direct/one-instance `vmap` states bind through the
   same lifecycle owner. Host/JIT see the same prepared values. Latest-schema
   and supported-legacy checkpoints use the unchanged decoder before these
   operations.
8. Literal zero observations/support masks, vacuous/improper initial state,
   non-finite mu3 moments, float32, and uncomfortable horizon/domain fixtures
   retain their baseline value or loud rejection. Invalid restore remains
   atomic.
9. Existing mu3/checkpoint/agent/family oracle tests remain authoritative.
   Old/new lifecycle characterization protects the move; this ticket creates
   no new scientific recovery experiment.
10. No checkpoint/bundle schema, PRNG policy, learning cadence, factor rule,
    convergence status, objective, diagnostic promise, or performance target
    changes.
11. The ordinary lifecycle and specialized mu3 owner each stay below 1,000
    physical lines. The ticket fits one context because it moves the
    preparation/binding responsibility anchored by the bounded e488df3 region
    and leaves run strategies separate.
12. Focused state/data/mu3/checkpoint tests and the final short suite pass.
13. `docs/ARCHITECTURE.md` names both lifecycle owners and their narrow typed
    inputs in the extraction commit.

## Methodological Suggestions — HOW

These suggestions are optional. An implementer may choose another sound method.

- Extract pure state/data operations first, then move mu3-specific preparation
  without broadening the ordinary lifecycle interface.
- Preserve explicit PRNG key inputs/outputs in the mu3 owner.
- Keep public methods as delegates so checkpoint and agent callers do not move.

## Future work

`cff-ii9t` extracts ordinary iterative run control and public status inputs.

## Definitions

- **Lifecycle:** Initialization, data binding, and prepared stochastic state
  before iterative inference chooses a termination result.
- **Prepared stochastic state:** Mu3 cache/moment/key information ready for an
  epoch without consuming an extra key.
- **Status authority:** The public owner that turns internal termination facts
  into `RunStatus` and `RunResult`.

## Notes

**2026-09-10T03:44:02Z**

Read-only preparation at 47701a6: ordinary initialization/uniform messages are engine.py:291-347; functional initialization currently lives inside init_state, preparation/readiness/update at :348-531, mu3 diagnostics/results/epoch scoring at :600-703, and data binding at :704-769. Hybrid GFE diagnostics (:532-599) are a readout responsibility already covered by cff-3ola, not a reason to broaden the functional-mu3 lifecycle interface. The new prerequisite cff-7kfk first relocates the remaining synchronous transport without changing these lifecycle algorithms.

Avoid an Engine import cycle: State/_replace and any needed pure public result records can move to a leaf owner with Engine re-exports preserving public imports and field schemas. Ordinary initialization should install defaults and arrays, then merge functional initial parameters supplied by the separate mu3 owner. A bounded functional plan view needs the existing Mu3Plan, associated moment/goal addresses and precomputed latent-message indices; its operations need numeric state/parameter views and explicit keys, not a whole Engine or access to unrelated factor families. Preserve scalar status/residual results and exact key splitting. Epoch scoring must call the existing run authority through an explicit typed callback; it may consume that result but must not add a convergence decision or resample during restore. The prepare/update code currently shares base-posterior checks and low-ESS policy that must move verbatim.

Binding preserves discrete None-to-uniform behavior, Gaussian/EF None-to-unbound masks, custom observed-Gaussian validators, existing unknown-port errors, and all-or-nothing State installation. cff-jby3 constraint keys and old prepared-cache reconstruction are now part of the current baseline; do not resurrect pre-repair empty constraint parameters. Current/legacy decoding remains outside these owners. New IR/schema/formulas/PRNG policy are N/A to modification. Use existing mu3, agent/checkpoint, custom-kernel data, constraint reconstruction and runtime tests plus bounded old/new state/key/diagnostic characterization. No implementation begun; nyad remains under its final gate.

**2026-09-10T04:14:59Z**

Additional boundary checks at cff-7kfk source 2aaef36 (read-only): do not turn this extraction into constructor-time numeric caching. init_state/_uniform_messages currently create JAX arrays at call time, so the active JAX precision setting is observed then. A probe constructed one HMM Engine under x64=True, then initialized under False/True: both messages and Gaussian arrays were float32/float64 respectively. Changing one functional Engine.mu3_samples from 16 to 32 made subsequent initial particle arrays contain 16/32 rows. prepare_mu3/update/init_state read the current Engine.mu3_samples and mu3_min_ess_fraction attributes; the separate owner should receive those current values rather than silently capturing initial settings. Gaussian observation validators currently come from the compiled binding snapshot, while EF validators are looked up through kernels.observation_validator during each bind; preserve those lookup times and error behavior.

A leaf runtime-types module can avoid Engine import cycles while preserving public re-exports. RunResult defaults refer to OuterTerminationReason, currently the small enum at outer.py:22-36; if the result records move, move that pure enum to the same leaf and re-export it from outer instead of making the state/types leaf depend on the numerical solver. Do not move solver code or change enum values. Mu3ScoreResult needs the same state/result/status identity. Epoch scoring must receive the current run/prepare callbacks at method call time, preserving explicit key threading and any existing public delegation. The unchanged checkpoint codec is array/metadata based, not Python-class serialization; verify public import identity and all declared field/value contracts, not an unsafe pickle compatibility promise. Hybrid GFE diagnostics remain with the cff-3ola readout responsibility.

**2026-09-10T04:37:04Z**

Starting from updated and remotely verified PR #127 head 67745474ee6bf1ef7a51875f36c7c60854dcb62e after its 1,992-test final gate. The existing handoffs above define scope and checked pairs. Implementation will use a leaf runtime-types module with public Engine/outer re-exports, a state/data lifecycle owner over immutable Arena/default parameters, and a separate functional owner over Mu3Plan views with precomputed latent-message indices. Functional settings are passed from current Engine attributes at each call; ordinary models do not enter the mu3 owner. Public epoch scoring supplies the current run/prepare callbacks, leaving run-result authority outside preparation. Move declarations and operations without changing field schemas, enum values, formulas, error policy, key order or call-time dtype behavior. Hybrid GFE diagnostics remain for cff-3ola. Current/legacy checkpoint decoding stays untouched; the cff-jby3 reconstruction regressions remain part of the baseline.

**2026-09-10T04:40:31Z**

First atomic step: State, RunStatus, InferenceError, RunResult, Mu3ScoreResult, replacement and OuterTerminationReason now live in runtime_types.py, a data-only leaf with no Engine or numerical-solver imports. Engine and outer re-export their existing public types; cffg.__all__ is unchanged (State was not a top-level cffg export before this move). Runtime-status and public-API selections pass all 14 cases in 17.02 seconds, including current checkpoint and configured-precision checks. Architecture records the type owner. Numerical methods, state fields and enum values are unchanged. This is a foundation commit, not completion of the lifecycle ticket; ordinary/functional owner extraction and the final full gate remain.

**2026-09-10T05:10:38Z**

Implementation checkpoint: pure runtime types are committed separately in e8493c4. StateLifecycle now owns initial arrays and DATA binding; Mu3Lifecycle receives immutable plans/latent indices and a messages/parameters-only State view. Engine preserves public delegates and supplies current particle settings plus current prepare/run callbacks. Hybrid-GFE diagnostics remain with the later readout ticket. No serialization, PRNG, solver, objective or learning-cadence changes.

Compatibility evidence: the trusted pre-extraction Engine at 2aaef36 and the extracted implementation produced 2,908 byte-identical comparisons across 17 model fixtures and both call-time precisions. This compares every State field, binding/unbinding, functional budgets 16/32, prepared and updated particles/moments, key progression and diagnostics. A separate 47-test existing-family/checkpoint/status selection passed (3 slow deselected); the new lifecycle file initially passed 19 cases. The final focused selection and full short-suite gate are the remaining validation, not yet claimed.

Ownership sizes: ordinary lifecycle 143 lines, functional lifecycle 380, pure values 146. Engine is 409 physical lines shorter than 2aaef36. New modules/tests pass Ruff and formatting; existing Engine lint findings decrease from 20 to 19 and outer remains 11, with no new findings after explicitly marking compatibility re-exports. Architecture change paths now name both lifecycle owners. Independent review agents remain unavailable; this evidence is local comparison and tests.

**2026-09-10T05:32:27Z**

Completed at code/test source cf5cdb6, with pure value types separately committed in e8493c4. Final focused lifecycle/public-API selection passed 26 tests in 19.70 seconds. The single final default suite passed 2,011 tests, zero failures, 41 slow deselections and four existing Gym warnings: pytest 1,127.50 seconds, measured wall 1,140.978 seconds. It completed 2026-09-10T05:29:54.237366+00:00 at ticket-only head 9009343. Runtime history and shared dashboard are refreshed. The gate was read once after the planned first completion check; no later code/test changes supersede it.

The lifecycle owners, pure type leaf, compatible Engine delegates and updated architecture/change paths are done. PR publication will reuse this gate. Next: cff-3ose repairs existing late-failure diagnostic discrepancies before cff-ii9t unifies the iterative policy. No significant user decision is pending. Keep hybrid-GFE diagnostics with the later readout extraction; do not capture JAX precision or mutable particle settings at Engine construction. Independent review agents remain unavailable and are not claimed.
