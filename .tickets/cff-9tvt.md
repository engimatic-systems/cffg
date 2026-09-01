---
id: cff-9tvt
status: open
deps: [cff-cfhi]
links: []
created: 2026-09-01T02:20:19Z
type: bug
priority: 1
assignee: coopwilliams
external-ref: gh-86
---
# Harden integrated mu3 cache and reconcile release docs

## Background in Plain English — WHY

The preceding ports combine functional `mu3(A)` planning with strict live-agent
checkpoints. Old PR #86 added one unique correctness repair after that
integration: commit `7cd3d851` rejected cached planning results whose restored
fields were corrupt, internally impossible, or stale for the prepared inputs.
Its merge commits and old release snapshot must not be replayed. This ticket
ports only that unique hardening and then reconciles the release documentation
against the current repository.

## Outcomes / Acceptance Criteria — WHAT

1. Restore rejects cached planning results with corrupt fields, impossible
   status or diagnostic combinations, invalid particles or weights, or a cache
   identity that does not match the prepared inputs. Particle, weight, moment,
   ledger, and finite-state validation applies to converged, budget-exhausted,
   infeasible, and diverged results alike.
   1.1. A genuine importance-weight collapse restores with raw evaluator status
        2, exact zero ESS, and the last coherent particles, weights, moments,
        and ledger preserved by the engine transaction.
   1.2. Zero ESS is accepted only for raw status 2. Analytic base moments are
        accepted only with the dtype-uniform prepared weights, exact-zero
        prepared ledger, and ESS equal to the particle count or the status-2
        zero sentinel. Otherwise moments must match the retained weights and
        their committed ledger; other combined states remain invalid.
2. Cache validation binds every value whose change would make a planning result
   stale, including the installed posterior, prepared stochastic state, bound
   planning data, runtime configuration, and structure identity.
3. Validation happens before live-agent mutation and preserves the atomic
   restore contract from `cff-cfhi`.
4. A valid cache restores and replays exactly; invalid cache state cannot be
   mistaken for a converged plan or reused to emit an action.
5. Tests cover exact-zero and non-finite arrays, malformed shapes and dtypes,
   contradictory statuses and diagnostics, changed prepared inputs, float32,
   and an uncomfortable functional-planning scale.
6. README, architecture, derivations, and limitations describe the integrated
   behavior and bound the release claim to the registered discrete paper core,
   finite-sample functional `mu3(A)`, and the tested constraint-flip subclass.
7. Release documentation incorporates current-main bundle and learning work;
   it does not restore stale text from old PR #86 or claim a universal solver.
8. The final branch contains the semantic work from PRs #82 through #86 in a
   clean linear ancestry from updated `main`, with no old merge commits.

## Methodological Suggestions — HOW

These suggestions are optional. An implementer may choose another sound method.

- Port the behavior of old commit `7cd3d851` only. Reconcile old final commit
  `800af9ee` by meaning and current documentation, not by patch replay.
- Use the **Runtime numerical or status invariant** and **Learnable or
  checkpoint feature** change-path rows. Ordinary and functional planning,
  action gating, cache identity, current/versioned/legacy restore, float32/x64,
  and atomic failure apply. Rule kernels, schedules, `Colony`/`vmap`, and the
  serialized model IR remain unchanged.
- Treat cache objects as untrusted decoded input. Validate cross-field
  invariants as well as individual array shapes and finite values.

## Future work

No additional work from discontinued PR #86 is identified. Broader paper
coverage, exact functional integration, and general constraint-flip
equivalence require separate roadmap tickets.

## Definitions

- **Cache identity:** The complete set of inputs and configuration that make a
  saved planning result reusable.
- **Prepared inputs:** The installed posterior, data bindings, particles,
  weights, and runtime state used to compute a stochastic plan.
- **Stale cache:** A cached result whose identity no longer matches the agent
  state that would consume it.

## Notes

**2026-09-01T02:23:06Z**

Planning audit, 2026-08-31. Capability owners are RollingAgent checkpoint/cache decoding, mu3 prepared-state and RunResult validators, action gating, and the named release documents/tests. Wiggle scan reviewed in context; retained candidates mean: integrate=combined agent/checkpoint layer; preserve=atomicity invariant; register=runtime-supported discrete family; reusable=artifact reuse under equal identity; complete=enumerated cache identity; remaining=explicitly unchanged surfaces; converge=RunStatus criterion; correct=defect repair; equivalent=separately scoped future proof; exact=bitwise continuation or literal zeros; match=identity equality; sound=optional method quality; test=automated artifacts; validate=untrusted input checks; learn/learnable=current-main declared-learning scope or change-path name; scale=functional horizon/domain size; may=optional method; every=the named identity inputs; only=ticket scope exclusion; current=main 5c3b06a, active docs, or decoder.

**2026-09-01T02:23:06Z**

Roadmap stack, 2026-08-31. Legend: prerequisite --> dependent.

[ ] cff-8a5m (ready, PR #82 port)
      |
      +--> [ ] cff-lpsn (PR #83 port)
                  |
                  +--> [ ] cff-7ozr (PR #84 port)
                              |
                              +--> [ ] cff-cfhi (PR #85 port)
                                          |
                                          +--> [ ] cff-9tvt (PR #86 unique hardening and release reconciliation)

Immediate blocker: each layer waits only for the preceding layer. Parallel work: none in this clean linear review stack. Downstream result: the final layer carries the bounded paper-core and newcomer-workflow release claim. The tk dependency records are authoritative.

**2026-09-01T03:54:54Z**

Completion handoff, 2026-08-31.

Done: clean-ported only the unique post-integration cache hardening from old commit 7cd3d851 and reconciled the bounded release claims against the current stack. No old merge commit, stale release snapshot, or discontinued ancestry was replayed. The restore path now rejects unclaimed, missing, malformed, non-finite, contradictory, or stale cached mu3 planning results before mutating the live agent. Cross-field validation covers cache identity, epochs, counters, configuration, statuses, diagnostics, particle and weight normalization, ESS, cached moments, KL consistency, and converged-state finiteness. README, architecture, and limitations now state the bounded registered-core, finite-sample functional-mu3, and fixed-A binary joint-boundary constraint-flip claims.

Validation: focused cache/public tests: 38 passed, 1 deselected; documentation, hygiene, and executable quickstart checks: 5 passed; final dashboard short suite at proposed PR head: 1519 passed, 41 deselected, 0 failed, 4 dependency deprecation warnings in 872.45 seconds. No slow run was needed because the corrupt-cache mechanism is covered by the default suite. Dashboard refreshed at ~/agent-share/cffg-tests.html.

Review stack: cff-8a5m -> cff-lpsn -> cff-7ozr -> cff-cfhi -> cff-9tvt. This branch is the final clean-port layer and should target the cff-cfhi branch until the stack is merged.

**2026-09-01T03:55:27Z**

Completion wiggle audit, 2026-08-31. All retained matches were reviewed in context. Meanings: fixed=the supplied, non-learned A boundary; integrate=combined agent and checkpoint layers; preserve=atomic restore invariant; public=the exported newcomer API and its tests; register=runtime family coverage; reusable=cache reuse under equal identity; support=probability support or explicitly named runtime capability; complete=the enumerated cache identity; done=the completion handoff label; ready=tk dependency readiness; remaining=explicitly unchanged surfaces; close=ticket or superseded-PR lifecycle state; check=automated invariant or artifact check; converge=RunStatus criterion; correct=the named defect repair; equivalent=separately scoped mathematical equivalence; exact=bitwise continuation or literal zeros; match=identity equality; pass=successful automated-test outcome; sound=optional method quality; test=automated artifacts; validate=untrusted-input checks; learn/learnable=current-main declared-learning scope or change-path name; scale=functional horizon or domain size; may=optional method; should=temporary stacked-branch base instruction; every=the locally named identity inputs; only=ticket-scope exclusion; current=main 5c3b06a, the active runtime value, or the named active documents. Capability ownership remains RollingAgent checkpoint/cache decoding, mu3 prepared-state and RunResult validation, action gating, and README/architecture/limitations plus their tests.

**2026-09-01T03:55:33Z**

Completion wiggle-audit supplement, 2026-08-31. The retained all candidate means the full enumerated set of mechanical-scan matches, not an unbounded correctness claim.

**2026-09-01T05:20:00Z**

Second-review repair handoff, 2026-08-31. The first cache validator applied
particle, weight, ESS, moment, and ledger checks only when the serialized
`RunResult` claimed convergence. That status gate admitted negative particles
in a budget-exhausted checkpoint; its restored runtime could return the cached
failure directly or reuse those particles after an input change. The prepared
evaluator domains and finite-state invariant now apply to every cached result
status. A mechanism regression modifies a real budget-exhausted checkpoint,
rebuilds its exact field inventory, and verifies rejection before mutation of
an already-used target. Focused corruption and status validation passed 15
tests in 141.64 seconds. Capability ownership remains the functional-mu3
prepared-state validator and RollingAgent atomic restore path.
The refreshed wiggle scan retains the recorded meanings; `verify` is the
regression assertion and `could` names the two concrete post-restore runtime
paths described in the preceding sentence.

Final second-review validation at repaired stack head `55a9e32`: the dashboard
short suite passed 1,525 tests with 41 deselected, 0 failures, and 4 dependency
deprecation warnings in 885.21 wall-clock seconds (873.87 seconds reported by
pytest). This one cumulative run validates the unchanged #103 through #106
layers together with #107's repaired cache and ticket state; their focused
runs were not duplicated. The dashboard was refreshed at
`~/agent-share/cffg-tests.html`. No slow run was required because both newly
reported mechanisms are covered in the default checkpoint and mu3 tests.

**2026-09-01T06:15:00Z**

Third-review repair handoff, 2026-08-31. Research question: what state does a
real functional-mu3 importance-weight collapse commit? Answer: the rule returns
an invalid candidate update, but the engine transaction commits only raw status
2 and exact zero ESS. It retains the fixed particles and the last coherent
weights, moments, and ledger. The prior validator incorrectly treated those
retained values as a newly committed importance update, derived positive ESS
from the weights, and rejected the genuine failure checkpoint. Sources:
`mu3_importance_update`, `Engine._update_mu3`,
`Engine._mu3_result_diagnostics`, `PlanningRuntime.run`, and the checkpoint
encoder/decoder. Confidence: high; two read-only agent traces and the direct
zero-goal reproduction agreed on the exact stored arrays and status path.

Validation now follows the engine transaction. Raw evaluator status must be
0, 1, or 2. Exact zero ESS is valid only for diverged status 2; otherwise ESS
must equal the retained normalized weights. Converged moments must match the
committed weighted particles. Budget-exhausted and diverged moments may match
either that committed update or the analytic prepared base, the only two
legitimate writers. The retained ledger identity remains mandatory for every
status. A real zero-goal collapse now round-trips, replays without consuming a
key, and advances to the same next refreshed epoch. Unknown status, negative
ESS, and invalid retained moments remain atomic restore failures. The complete
checkpoint/public focus passed 44 tests with 1 deselected in 281.54 seconds.
Capability ownership remains the engine mu3 update transaction, the
functional-mu3 prepared-state validator, and RollingAgent checkpoint restore.
The refreshed wiggle scan retains the recorded meanings; `same` is exact
source/restored stochastic continuation, `reproduce` is the direct zero-goal
mechanism, and `read` describes non-mutating agent investigation.

Final third-review validation at `9ba0987`: the refreshed dashboard short
suite passed 1,529 tests with 41 deselected, 0 failures, and 4 dependency
deprecation warnings in 885.69 wall-clock seconds (874.21 seconds reported by
pytest). The dashboard was refreshed at `~/agent-share/cffg-tests.html`. No
slow run was needed because the engine-produced collapse and its checkpoint
continuation are covered by the default functional-agent suite.

**2026-09-01T07:10:00Z**

Fourth-review repair handoff, 2026-08-31. Research question: which combined
payloads can the engine write before a checkpoint? Answer: exactly two. The
prepared payload has dtype-uniform weights, analytic moments from
`base_alpha`, an exact-zero ledger, and ESS equal to the particle count. A
committed importance update has normalized weights, their weighted particle
moments, a coherent ledger, and weight-derived ESS. Stationary failure retains
one whole payload. Weight collapse also retains one whole payload and replaces
only status with 2 and ESS with the exact-zero failure sentinel. Sources:
`Engine.prepare_mu3`, `Engine._update_mu3`, and `mu3_importance_update`.
Confidence: high; two read-only traces agreed in float32 and x64, and a direct
checkpoint mutation reproduced the acceptance gap.

The validator now accepts those payloads as atomic alternatives instead of
mixing their independently valid fields. Analytic prepared moments therefore
require bitwise dtype-uniform weights, the zero prepared ledger, and ESS equal
to the particle count or zero for status-2 collapse. The regression replaces
uniform weights with a one-hot normalized vector, changes state and metadata
ESS together to 1, leaves analytic moments and the zero ledger unchanged, and
verifies atomic rejection. Genuine collapse replay and all prior single-field
corruption cases remain valid. The complete checkpoint/public focus passed 45
tests with 1 deselected in 279.05 seconds. Capability ownership remains the
engine's two mu3 payload writers and the functional checkpoint validator.
The refreshed wiggle scan retains the recorded meanings; `can` is bounded to
the two enumerated engine payload writers and `all` refers to the prior named
single-field regression cases.
