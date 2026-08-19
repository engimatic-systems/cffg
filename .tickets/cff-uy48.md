---
id: cff-uy48
status: closed
deps: [cff-epdp, cff-in0q, cff-iqqk, cff-u3m3, c1-a2cb, c1-2h8y]
links: [cff-cfqg, cff-bpgm]
created: 2026-08-13T04:26:16Z
type: epic
priority: 1
assignee: coopwilliams
tags: [paper-fidelity, release, usability]
---
# Epic: Prove paper faithfulness and a minimum usable release

## Background in Plain English

The engine-trustworthiness epic `cff-cfqg` covers runtime truth, performance,
central execution seams, and declared learning. It deliberately does not prove
that the public agent workflow realizes the two CFFG papers or that a newcomer
can use the package through one supported path.

This separate gate closes that gap. It requires a source-anchored inventory of
the papers' constraint language and Part II Figure 5 messages, puts the
functional `mu3(A)` message into the real agent trial loop, compares the
two-window agent implementation with one literal constraint-flip reference,
and establishes one public path from model construction through checkpoint
restore. It is a bounded release claim, not a claim that every factor and
constraint combination has a closed-form rule.

## Outcomes / Acceptance Criteria

1. Every ticket in this epic's dependency metadata is closed because its own
   acceptance criteria are satisfied.
2. A source-anchored inventory accounts for every constraint construction
   defined by the two papers and every message in Part II Figure 5.
3. Each inventoried construction is classified as an exact executable path, a
   named approximation, syntax without an executable rule, or unsupported.
4. Every public paper-faithfulness claim names the factor family, constraint
   combination, ledger meaning, and executable evidence that bounds it.
5. `RollingAgent` can execute the nonconjugate functional `mu3(A)` message as
   part of a complete perception, planning, action, learning, and next-trial
   cycle.
6. The agent's `mu3` path has explicit random-key ownership, fixed-particle and
   refresh semantics, stochastic status propagation, and exact checkpoint
   continuation.
7. One compact discrete model compares the paper's literal constraint change
   with the repository's separate planning and filtering windows against an
   independent exact oracle.
8. One documented newcomer path builds a model, compiles it, runs inference,
   checks status, inspects beliefs and the objective, checkpoints an agent,
   restores it, and continues through supported public imports.
9. README, architecture, derivation, and limitation statements distinguish
   the paper's general derivational framework from the finite executable rule
   registry.
10. Closing this epic permits the statement that the named paper core and
    minimum public workflow passed their declared evidence gate. It does not
    permit claims of a universal solver, every paper construction being
    executable, or numerical exactness for a declared approximation.

## Methodological Suggestions

These suggestions are optional. An implementer may choose another sound method
that satisfies the numbered outcomes with equivalent evidence.

- Treat `cff-epdp` as the scope ledger. A missing critical behavior becomes a
  dependency of this epic before the inventory ticket closes; it does not
  disappear into an "unsupported" label after being advertised as supported.
- Review the formal local message and node-energy evidence separately from
  behavioral demonstrations. End behavior cannot prove a local update or
  ledger identity.
- Keep the paper-literal comparison small enough for exact enumeration. Use it
  to test the window transformation, not to repeat the T-maze benchmark.
- Keep `cff-cfqg` parallel. This gate owns paper and newcomer evidence;
  `cff-cfqg` owns the broader operating and extension baseline.

```text
Legend: prerequisite --> dependent

[ ] cff-epdp (ready) --> [ ] cff-in0q --+
                    +--> [ ] cff-iqqk --+--> [ ] cff-uy48
                    +--> [ ] cff-u3m3 --+

Parallel operating gate:
[ ] cff-cfqg -------------------------------+
                                             +--> [ ] cff-ahps
[ ] cff-uy48 -------------------------------+
```

## Future work

- `cff-epdp` owns the constraint and Figure 5 inventory and its direct evidence
  map.
- `cff-in0q` owns the complete functional-`mu3` agent trial lifecycle.
- `cff-iqqk` owns the literal constraint-flip versus two-window oracle.
- `cff-u3m3` owns the stable newcomer-facing path and top-level facade.
- `cff-cfqg` remains the separate engine-trustworthiness gate. Neither epic
  substitutes for the other.
- New factor families, general structured variational forms, and executable
  coverage for combinations classified as unsupported require their own
  derivation and registration tickets.

## Definitions

- **Constraint flip:** Changing a future variable from a planning constraint
  to observed data after the environment returns an observation.
- **Figure 5:** The message table for the discrete goal-observation composite
  in Part II of the CFFG papers.
- **Minimum usable release:** The smallest documented public workflow that a
  newcomer can execute and resume without relying on internal modules.
- **`mu3(A)`:** The nonconjugate functional message from future planning slices
  to an uncertain observation-model parameter in Part II.
- **Paper faithfulness:** Agreement between the declared model, constraints,
  local updates, objective ledger, and trial protocol and their paper sources.
- **Two-window runtime:** `RollingAgent`'s separate compiled planning and
  filtering models, reused across an episode.

## Notes

**2026-08-13T04:37:28Z**

Roadmap refinement, 2026-08-12. Legend: prerequisite --> dependent.

[ ] epdp (ready) --> [ ] in0q --+
                 +--> [ ] iqqk --+--> [ ] uy48
                 +--> [ ] u3m3 --+

Immediate blocker: the three implementation/evidence children wait for epdp. Downstream baseline: cfqg + uy48 --> ahps.

**2026-08-13T04:40:34Z**

Authoring handoff, 2026-08-12: the complete survey paragraph is assigned across cff-epdp, cff-in0q, cff-iqqk, and cff-u3m3. cff-epdp is the ready root. The ticket scan was reviewed after refinement; dependency-cycle and diff-hygiene checks passed. No pytest run was required because this change modifies only tickets and documentation.

**2026-08-18T22:58:29Z**

Integration handoff, 2026-08-18. Applicable change paths: Learnable or checkpoint feature and the RollingAgent trial-time planning/data boundary. This gate merges the three already-reviewed sibling heads without new serialized IR. Required duplicated surfaces: ordinary and functional-mu3 planning, filter rollover, action selection, status propagation, plan-result cache identity, PRNG ownership, current/versioned/legacy checkpoint decoding, grounded static-parameter and plate digests, float32/x64 execution, exact-zero support, literal mixed-history comparison, README/ARCHITECTURE/LIMITATIONS claims, and dashboard registration. Host/JIT rule pairs, flooding/scheduled pairs, and Colony/vmap are unchanged by this integration. The merge must retain cff-in0q functional evaluator state/config, cff-iqqk independent literal-oracle evidence, and cff-u3m3 atomic public checkpoint/facade contracts; any overlap is resolved to the stricter combined invariant.

**2026-08-19T00:09:35Z**

Completion handoff, 2026-08-18. The integration contains cff-in0q at aca4ba3, cff-iqqk at 9e21b9f, cff-u3m3 at 9cc1be7, and final cached-runtime hardening at 7cd3d85. Independent read-only audits found no remaining actionable checkpoint, functional-mu3 lifecycle, constraint-flip, documentation, or merge gaps. The final default not-slow dashboard run against code commit 7cd3d85 passed 1,411 tests, deselected 41, failed 0, and recorded 1,247.018 seconds of runner wall time; it refreshed /Users/ravd/agent-share/cffg-tests.html. The permitted release statement is bounded to the registered fixed-A discrete paper core and the documented minimum public workflow. Functional mu3(A) remains finite-sample importance sampling; the literal/two-window equivalence is limited to the tested fixed-A binary discrete joint-boundary construction; this is not a universal solver claim.
