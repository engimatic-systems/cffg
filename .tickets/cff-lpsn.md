---
id: cff-lpsn
status: open
deps: [cff-8a5m]
links: []
created: 2026-09-01T02:20:19Z
type: feature
priority: 1
assignee: coopwilliams
external-ref: gh-83
---
# Clean-port functional mu3 through RollingAgent

## Background in Plain English — WHY

The engine can evaluate the Part II functional `mu3(A)` message with importance
sampling, but current `RollingAgent` does not carry that evaluator through a
complete perception, planning, action, learning, and next-window cycle. Old PR
#83 implemented that capability on discarded ancestry. This ticket rebuilds
the capability after `cff-8a5m` on current `main`, while keeping the existing
analytic learned-observation path as a separately described approximation.

## Outcomes / Acceptance Criteria — WHAT

1. A planning template opts into the registered functional evaluator through
   serialized constraint metadata; `RollingAgent` does not infer it from
   factor names or a learning flag.
2. A stochastic agent owns an explicit JAX key cursor, requires a supplied key
   before its first stochastic plan, and keeps action-selection NumPy randomness
   in a separate stream. Deterministic agents remain key-free.
3. Each planning epoch installs the current proper observation-model posterior
   through `Learnables` and prepares particles from exactly that posterior.
4. One inference attempt holds a fixed particle set. A declared new scoring
   epoch may refresh it, and any changed posterior or prepared input invalidates
   stale particles before inference.
5. The planning result exposes each evaluator's effective sample size, log
   normalizer, stochastic status, and advanced key cursor.
6. The agent returns an action only when ordinary inference and every active
   functional evaluator converge. Sampling-budget exhaustion, inference-budget
   exhaustion, infeasibility, and divergence remain distinct.
7. Filtering remains the owner of realized observations and sufficient
   statistics. Planning messages do not count as observed data, and rollover
   injects the declared filtering posterior into the next planning window.
8. Runtime checkpoint state is sufficient to replay stochastic continuation,
   including the key cursor, prepared particles and weights, diagnostics,
   refresh identity, and posterior identity. The stricter versioned public
   envelope is completed later by `cff-cfhi`.
9. Direct and agent tests cover exact-zero support, improper or vacuous inputs,
   float32, a planning horizon or domain 10–100 times the smallest fixture, key
   replay, cache invalidation, and checkpoint continuation.
10. README, architecture, derivation, and limitation text distinguish the
    finite-sample functional path from the analytic posterior-moment path and
    state the supported lifecycle precisely.
11. The implementation is a semantic port from old PR #83, not a replay of its
    ancestry, and it preserves current-main declared-learning and recurrence
    checkpoint behavior.

## Methodological Suggestions — HOW

These suggestions are optional. An implementer may choose another sound method.

- Adapt old commit `dfae3149` rather than cherry-picking it blindly. Current
  `main` has newer learning/checkpoint owners that must remain authoritative.
- Use the **Learnable or checkpoint feature** change-path row. Compiler/Arena
  detection, agent lifecycle, plan-result state, parameter installation,
  rollover, and current checkpoint transport apply. The serialized IR schema,
  factor rules, flooding/scheduled pair, and `Colony`/`vmap` remain unchanged.
- Keep the agent module readable by moving cohesive `mu3` lifecycle and state
  validation helpers into a private module if needed.

## Future work

`cff-7ozr` adds an independent paper-literal constraint-flip oracle on top of
this layer. `cff-cfhi` supplies the strict public checkpoint envelope and
newcomer facade. `cff-9tvt` hardens restored cached planning results after all
runtime state formats are integrated.

## Definitions

- **Effective sample size:** A diagnostic that measures how many importance
  samples meaningfully contribute after weighting.
- **Planning epoch:** One period during which the stochastic estimator uses a
  fixed prepared particle set.
- **Posterior identity:** A stable description of the parameter posterior from
  which cached particles were prepared.
- **`mu3(A)`:** The nonconjugate functional message from future planning slices
  to an uncertain observation-model parameter in Part II.

## Notes

**2026-09-01T02:23:06Z**

Planning audit, 2026-08-31. Capability owners are serialized evaluator metadata, Engine mu3 preparation/results, Learnables parameter installation, RollingAgent lifecycle and checkpoint transport, and the named documentation/tests. Wiggle scan reviewed in context; retained candidates mean: fixed=one particle set or conjugate family; integrate=agent lifecycle wiring; preserve=current-main compatibility; public=RollingAgent behavior; register=runtime evaluator identity; expose=RunResult fields; implement=old-PR behavior or component wiring; support=probability support or supported lifecycle; complete=the named trial cycle or replay state inventory; remaining=active posterior/state; approximate=named analytic path; material=meaningful state change; sufficient=checkpoint inventory or statistics; converge=RunStatus criterion; estimate=importance estimator; exact=posterior identity or literal zero; oracle=independent quadrature test; sound=optional method quality; stable=posterior identity; test=automated artifacts; validate=cache/state input checking; infer=runtime inference; learn/learnable=declared parameter update workflow; rollover=boundary transfer; small=fixture size; can=engine capability; may=declared refresh or optional method; explicit_owner=filtering; all/every/any/only=the enumerated active evaluators and state transitions; current=main 5c3b06a or active posterior; existing=pre-ticket analytic path; later=named downstream envelope.
