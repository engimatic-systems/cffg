---
id: cff-5xdy
status: open
deps: [cff-nyad]
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
