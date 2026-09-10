---
id: cff-7kfk
status: closed
deps: [cff-nyad]
links: [cff-3ola]
created: 2026-09-10T03:41:20Z
type: chore
priority: 1
assignee: coopwilliams
parent: cff-ws3c
tags: [engine, architecture, execution, refactor]
---
# Move synchronous sweep transport behind the execution owner

## Background in Plain English — WHY

The engine still contains the global synchronous update loop. Local factor
calculations now have execution owners, and schedule construction has its own
owner, but Engine still gathers each factor's inputs, collects its proposed
messages, combines shared-goal contributions and applies damping. Leaving this
code for the final readout ticket would add an unrelated responsibility there.

At source 47701a6, engine.py:2306-2648 contains about 343 lines of these helpers
and the sweep factory. This is the source anchor, not a future line promise.
The schedule program already accepts the factory as an explicit input. This
ticket moves that factory and its transport helpers to an execution module;
it does not change the update algorithm.

## Outcomes / Acceptance Criteria — WHAT

1. cff-nyad is closed and its published head is updated before branching.
2. One synchronous execution owner accepts an immutable Arena and the existing
   full-versus-iterative-phase selection, and returns the same typed sweep.
3. Engine supplies that owner's factory to ScheduleProgram instead of defining
   the global numerical transport loop itself.
4. Local factor calculations and parameter selection remain delegated to the
   existing execution owners; this move creates no second family rule.
5. Every factor reads the pre-sweep arrays. Shared-goal predictions are pooled
   once before damping; cross-domain writes join their original damping phase.
6. Full flooding and the restricted mixed phase preserve their selected groups,
   update ordering, output arrays, dtype, inner residual/status facts, log-zero
   flooring and delta hardening.
7. Constraint updates still consume the current State.params values repaired
   in cff-jby3, not captured compiled defaults.
8. The thin Gaussian parameter selector remains usable by ledger/readout code
   without importing the public Engine into the new execution owner.
9. Affected change paths are Runtime numerical or status invariant and New
   factor or rule family. Full/restricted sweeps, direct/JIT/vmap transport,
   shared goals and cross-domain writes are checked pairs. State schemas,
   checkpoint decoding, public result classification and objective formulas
   are N/A to modification; restored states remain consumers of the same sweep.
10. Exact zeros, vacuous/improper inputs, non-finite updates, float32/float64
    and existing larger-chain fixtures preserve baseline values or failures.
    Existing guards are moved unchanged; no new numerical policy is introduced.
11. Existing family, shared-goal, constraint, schedule, Colony and engine-contract
    assertions remain authoritative. Bounded old/new array characterization
    protects the move; no new scientific or speed claim is added.
12. The new owner is below 1,000 physical lines and does not import Engine.
13. Focused tests, one final default gate and artifact checks pass.
14. ARCHITECTURE and the change-path index name the delivered sweep owner.

## Methodological Suggestions — HOW

These suggestions are optional; another sound method is acceptable.

- Move the existing helpers and factory to execution/synchronous.py, reusing
  execution/transport.py and the existing typed local outputs/selectors.
- Keep the ScheduleProgram factory input unchanged. Redirect private test
  instrumentation to its new owner without weakening assertions.
- Reuse the bounded trusted-source comparison approach from cff-nyad, including
  shared goals, constraints, precision, recurrence, hybrid and functional cases.

## Future work

cff-5xdy follows this move with state/data and functional lifecycle owners.
cff-3ola can then remain a readout/facade extraction instead of also moving the
synchronous transport loop. The convergence-loop compilation target remains
separate in cff-wbhu.

## Definitions

- **Synchronous sweep / flooding:** Compute all proposed factor messages from
  the same old graph state, then install their combined updates.
- **Damping:** Blend a proposed update with the previous value.
- **Restricted mixed phase:** Update iterative factor groups before the
  schedule refreshes ordinary belief-propagation islands.
- **Transport:** Gather local inputs and scatter local outputs, without
  redefining the factor calculation.
- **Delta hardening:** Project a declared point-mass edge onto its selected value.
- **JIT / vmap:** JAX compilation and vectorized execution transformations.
- **State.params:** One graph instance's numerical runtime parameters.

## Notes

**2026-09-10T03:44:02Z**

Planning evidence at 47701a6: the remaining engine.py:2306-2648 block is synchronous transport and damping, not the family-local numerical rules already extracted by cff-ny7f. ScheduleProgram receives _make_step as a factory input, so that boundary need not change. _gauss_group_params also serves Gaussian readouts (:1962,2201) and the ledger (:2661); preserve its selector identity or redirect those callers to the existing selector. Private factory instrumentation appears in shared-goal and schedule tests. Confidence: high, bounded symbol/caller inspection; no independent agent review is claimed.

Legend: prerequisite --> dependent; based on tk dep tree --full cff-5xdy, tk ready and tk blocked.
[x] cff-ny7f local execution
  --> [x] cff-jby3 constraint isolation
        --> [~] cff-nyad schedule construction (final gate running)
              --> [ ] cff-7kfk synchronous transport (immediately blocked by nyad)
                    --> [ ] cff-5xdy lifecycle
                          --> cff-ii9t --> cff-jrbw --> cff-gl89
                                --> cff-3ola --> cff-ws3c runtime epic

cff-5xdy retains its direct nyad dependency as well. No parallel architecture root is introduced; unrelated P2 validators cff-34gi and cff-6n9c remain independently ready. tk dep cycle passes. This small insertion keeps cff-3ola limited to readouts/facade work and requires no new inference choice.

**2026-09-10T04:01:32Z**

Starting from updated, remotely verified PR #126 head 930a8c32f88235dadf6b572538a62169add79e09 after its 1,991-test final gate. The source/caller inventory above remains current. Keep the existing sweep factory signature and every numerical operation; move transport into execution/synchronous.py and redirect Engine plus private test instrumentation. Replace the redundant Gaussian selector wrapper with its existing kernel_runtime.parameters owner at the three ledger/readout callers. No new model, State, schema, objective, public status or PRNG policy is in scope.

**2026-09-10T04:09:07Z**

The move is implemented in execution/synchronous.py (417 physical lines); Engine is 352 lines shorter. Its nine moved function bodies match baseline 47701a6 under AST normalization that excludes only docstrings, the factory name/return annotation and the redundant Gaussian selector-forwarding alias. The three Gaussian ledger/readout callers now use kernel_runtime.parameters directly. Engine injects make_sweep into the unchanged ScheduleProgram contract; private tests follow the moved owner. No formula, parameter ownership, State schema or public status changed.

Focused execution/specialized/shared-goal/constraint/program/Colony/engine-contract checks passed 168 cases in 94.91 seconds. After adding the explicit ownership-boundary test, test_execution.py passed all 13 cases in 6.03 seconds. The boundary check requires one canonical factory outside Engine and no Engine/State/public-status/family-rule imports in the synchronous owner. New/small-file Ruff and format checks pass. Engine retains 20 baseline findings and schedule tests retain three; no new code/message findings appeared. Architecture and the change-path index are updated. The existing schedule/custom-kernel-focused selection and bounded numerical parity are finishing; no final full-suite claim yet.

**2026-09-10T04:11:58Z**

Final schedule/custom-kernel selection passed 40 tests in 27.49 seconds. The bounded numerical comparison at /tmp/cffg-synchronous-parity.py loads trusted Engine source 47701a6 and compares 17 models across automatic/flooding and float32/float64: all 2,622 arrays are bitwise identical, including matched NaNs, with the same functional scatter dtype warning as the prior extraction. Schedule identity, priming, reconstruction and exact units also match. All focused checks are complete; no further code change is planned. Commit the source, then run the final default suite once. Latest measured wall is 1,135.251 seconds and suggested wait is 1,440 seconds; first output check at 20 minutes, then at the 24-minute allowance only if still running.

**2026-09-10T04:32:30Z**

Final default gate passed at source 2aaef36: 1,992 tests, zero failures, 41 opt-in slow deselections and four existing Gym warnings; pytest reported 1,124.65 seconds and measured wall time was 1,138.059 seconds. Completion: 2026-09-10T04:30:56.938172+00:00, recorded against 8a92f49. The dashboard includes the new ownership-boundary case. Subsequent e35aae6/8a92f49 commits change only lifecycle ticket text, not code. The suite was checked once after its 20-minute estimate; no repeated test polling or publication rerun is needed.

All acceptance criteria are met: one synchronous transport owner supplies the existing schedule factory interface; every moved function body and all bounded numerical outputs match; runtime constraint ownership, shared goals, cross-domain damping, hardening and failure facts are preserved; public schemas and policies are unchanged; the owner is below 1,000 lines and the architecture index is updated. Close and publish above PR #126, then update that published base and proceed to ready cff-5xdy using its detailed handoff. No significant user decision is pending.

**2026-09-10T04:34:12Z**

Published as PR #127, https://github.com/engimatic-systems/cffg/pull/127, above #126. The shared architecture page now explains the completed synchronous transport owner, preserved global update ordering and baseline parity, and the next lifecycle boundary. HTML validation passed with 63 unique IDs and 35 resolved internal anchors. The dashboard shows 1,992 passing tests. Source 2aaef36 and gate/closure 9f939fa remain authoritative; publication changes no code. Next: verify/update this published branch and start ready cff-5xdy. The PR stack remains open for review, not merged.
