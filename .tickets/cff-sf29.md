---
id: cff-sf29
status: in_progress
deps: [cff-edoz]
links: [cff-ny7f, cff-34gi, cff-ddsq]
created: 2026-09-02T04:02:01Z
type: task
priority: 1
assignee: coopwilliams
parent: cff-ws3c
tags: [engine, architecture, characterization]
---
# Pin engine contracts before extraction

## Background in Plain English — WHY

The compiler track gives runtime consumers stable typed plans. The next work
will move execution, schedules, objective accounting, and readouts out of a
3,973-line engine module. Before that move, reviewers need direct evidence for
the values and failures each path produces at the named baseline.

This ticket inventories the existing mathematical authorities, adds only the
small old/new observable characterization needed for refactor seams, and maps
ownership at a named baseline commit. It does not move production engine code,
redefine a public result, or create a new scientific validation program.

## Outcomes / Acceptance Criteria — WHAT

1. `cff-edoz` is closed with its compiler characterization and typed-stage
   outcomes satisfied.
2. The characterization record names the exact commit that closes compiler
   track `cff-edoz` and uses it as the engine baseline. e488df3 remains the
   authoring/source-map reference and, where `cff-209l` promises parity, an
   external public/serialized/numerical reference; it is not the engine
   characterization baseline.
3. A compact inventory maps ordinary discrete, Gaussian, exponential-family,
   mixture, recurrence, structured recurrence, precision, hard-constraint,
   fixed-A GFE, transition GFE, hybrid GFE, and functional `mu3` execution to
   their existing focused tests and mathematical authority where one exists.
4. Small characterization fixtures are added only where an extraction seam has
   no existing observable protection. Together with the inventory they pin:
   4.1. outgoing messages and public beliefs;
   4.2. run status, logical iteration count, residual, and termination reason;
   4.3. schedule kind and fingerprint;
   4.4. BFE or GFE total and independently testable local terms;
   4.5. factor and parameter diagnostics;
   4.6. public readouts and their shapes; and
   4.7. prepared stochastic state, key cursor, and cache identity where
        applicable.
5. Existing independent analytic, enumeration, quadrature, literature, or
   source-specific oracle tests remain the mathematical authority where they
   already exist. Old/new observable characterization is permitted for a
   refactor-only seam and does not become a scientific correctness claim. A
   newly discovered evidence gap becomes a separate linked ticket and does not
   expand this epic unless it shows that the proposed code move is unsafe.
6. Failure fixtures cover budget exhaustion, divergence, infeasibility,
   unsupported execution, non-finite state, and an inner-solver failure.
7. A compact ownership table identifies the named baseline regions that gather
   inputs, select parameters, call rules, write messages, build schedules,
   control run strategies, prepare lifecycle state, compute objective terms,
   or construct readouts. Repeated interpretations are named for later removal.
8. Deliberate execution pairs are explicit.
   8.1. Flooding and scheduled execution agree where both are supported.
   8.2. A deterministic Colony fixture contains at least two instances with
        one structure and different parameters/observations, plus one instance
        with a different structure. The shared group uses ordinary iterative
        BP/VMP and actually enters Colony's vectorized update path; a GFE
        per-instance controller or exact one-shot group cannot substitute.
        Per-instance state, objective trace,
        status, residual, and iterations are compared with separate Engine
        runs under the named baseline contract. Pre-existing discrepancies
        are recorded and separately ticketed unless they make extraction unsafe.
   8.2.1. One shared structure retains one cached Engine; different structures
          retain separate groups. Updating/removing one instance does not alter
          another's data or identity. A one-instance group is not this proof.
   8.3. Host result classification agrees with status values produced by JIT.
   8.4. Latest-schema and supported legacy agent checkpoints reconstruct results that
        satisfy the same public engine invariants; checkpoint decoding itself
        remains outside the engine extraction.
9. Hostile fixtures include literal zero support, a vacuous message, an
   improper continuous belief, a non-finite update, float32, and one domain or
   horizon at least ten times its smallest fixture.
10. No production engine behavior, public API, serialized structure, checkpoint
    field, or performance target changes in this ticket.
11. Focused runtime, schedule, rule-family, ledger, readout, agent, and
    checkpoint tests pass. The dashboard-backed short suite passes after the
    final test change.
12. The ticket fits one implementation context because it inventories existing
    evidence and adds only the smallest characterization for uncovered seams;
    it moves no production code and creates no new scientific benchmark.
13. The ownership inventory includes Colony's existing status loop and result
    construction, plus State/public result type imports. It identifies the
    shared result policy to extract in cff-ii9t and a lower-level runtime value
    owner re-exported by the facade, avoiding imports back into Engine.
14. Existing discrete/Gaussian chains record schedule kind, array shapes and
    dtypes, persistent bytes, scan carry, and trace reuse. These extend the
    pre-registration records from cff-wera/cff-zq25; they are structural
    preservation checks, not benchmark or speedup claims.

## Methodological Suggestions — HOW

These suggestions are optional. An implementer may choose another sound method.

- Reuse existing oracle tests without broadening their scientific claims. Add
  only the smallest old/new observable fixture needed by an uncovered
  extraction seam.
- Record values field by field rather than storing one opaque engine snapshot.
- Use the **Runtime numerical or status invariant** row as the primary
  change-path checklist. Also use **New factor or rule family** and **Learnable
  or checkpoint feature** for affected consumers.
- Keep the ownership map in the ticket handoff until implementation changes
  architecture; update `docs/ARCHITECTURE.md` with the delivered owners, not
  the pre-refactor duplication.

## Future work

`cff-ny7f` extracts execution owners after this evidence passes. Any scientific
evidence gap discovered by the inventory gets a separate ticket outside this
architecture epic unless the gap makes the planned refactor unsafe.

## Definitions

- **Independent oracle:** A calculation or source that does not call the
  production path being checked.
- **Local term:** One named factor, parameter, constraint, or edge contribution
  to BFE or GFE.
- **Ownership map:** A list connecting each repeated operation to the code that
  performs it at the named baseline.
- **Termination reason:** The detailed cause associated with a public run
  status.

## Notes

**2026-09-09T23:27:48Z**

Preparation only while cff-ddsq final gate runs at source 707c31b; the required exact compiler-track closing commit will be recorded after that gate and closure. Colony evidence gap confirmed by bounded local inspection: colony.py:94-111 runs GFE groups through per-instance Engine.run; lines113-121 also bypass vectorization for one-instance groups. The real shared path builds filter_vmap(one) at lines148-156 and applies per-instance active masks/results at lines174-276. tests/test_gaussian_kernel_schedules.py:109 proves cached-engine sharing with two distinct parameter/observation instances and analytic Gaussian means/covariances/FE, but not the required iterative BP/VMP comparison of status, residual, iterations, traces, a second structure, and bind/remove isolation. Do not label that Gaussian test a one-shot Colony execution: Colony batches can iterate even when the corresponding standalone Engine uses its exact-unit path. The new characterization should explicitly assert a non-exact ordinary schedule and absence of GFE/mu3 fallback, then observe actual vectorized update entry. Existing test_colony GFE/perception/cache cases remain useful but do not substitute for this complete deliberate pair. Source: colony.py:39-78,94-276; engine.py:329-412; tests/test_colony.py:24-99 and test_gaussian_kernel_schedules.py:109-132. No runtime code changes or probes were made; this is an inventory/handoff, not a reported behavioral defect.

**2026-09-09T23:43:39Z**

Exact engine characterization baseline: 76251b44fc467a976aa056556f8fb371fe5de7f4, the commit that closes compiler track cff-edoz and final assembly cff-ddsq. Its source is 707c31b, validated by 1,813 passing default tests with 41 slow deselections and four existing Gym warnings in 1,016.282 seconds measured wall; runtime record 6364053. Later publication/handoff commits may change tickets/docs only. Start from the published agent/cff-ddsq-arena-assembly branch after its PR is created and update that intended base before branching. This ticket adds characterization/inventory only, with no production engine move or numerical behavior change. P2 cff-34gi records a separate, pre-existing fixed-GFE probability-input validation gap; use valid probability inputs for engine behavior parity and do not redefine that defect as a supported mathematical approximation.

**2026-09-09T23:45:38Z**

Published intended base: PR #121, https://github.com/engimatic-systems/cffg/pull/121, branch agent/cff-ddsq-arena-assembly. Update that branch before starting the engine characterization branch. The exact comparison baseline remains compiler-track closing commit 76251b44fc467a976aa056556f8fb371fe5de7f4, not a later publication-note commit.

**2026-09-09T23:47:40Z**

Starting from updated PR #121 branch at 1f5f18d. The exact comparison baseline remains compiler-track closing commit 76251b44fc467a976aa056556f8fb371fe5de7f4; source 707c31b passed 1,813 default tests. Primary change-path row: Runtime numerical or status invariant. New factor/rule and Learnable/checkpoint rows apply to inventory/readout/reconstruction consumers; New IR field is N/A because no schema or model meaning changes. Deliberate pairs: flooding/scheduled, standalone/actual iterative vmap with two distinct same-structure instances plus another structure, host/JIT status, and current/legacy agent reconstruction. Inventory existing independent authorities before adding tests; add characterization only for uncovered extraction seams, especially per-instance Colony result policy and cache/lifecycle identity. Hostile cases must include zero support, vacuous/improper beliefs, non-finite updates, float32 and a tenfold domain/horizon, primarily through existing focused tests. No production engine changes or new scientific benchmark. New gaps become separate tickets unless they make extraction unsafe. Use the exact baseline for source regions and observable values; update the dashboard for new test files and run one final short suite after the last test change. Additional agents remain unavailable; bounded local inspection and executable comparisons will provide the evidence.
