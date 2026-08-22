---
id: cff-e10u
status: closed
deps: [cff-bmax]
links: []
created: 2026-08-22T20:21:34Z
type: feature
priority: 1
assignee: coopwilliams
parent: cff-ngxx
tags: [rslds, compiler, engine, trustworthiness]
---
# Compile and execute D10 recurrent switching components

## Background in Plain English — WHY

cff-bmax supplies a pure D10 coordinate with one categorical path, one
Gaussian trajectory, and one conservative free-energy objective. Production
still runs D3 mixture collapse and D9 recurrence as separate local rules.
The engine needs an explicit component owner so eligible recurrent switching
graphs use the D10 coordinate once and do not also run or score the replaced
local rules.

The serialized graph remains the source of truth. The compiler may derive a
component plan only when topology proves the D10 assumptions. Unsupported
graphs must retain current D3/D9 behavior and claims.

## Outcomes / Acceptance Criteria — WHAT

1. The compiler derives a named structured hybrid component only for an
   acyclic D9 recurrence chain paired with destination-selected
   `gaussian_mixture_linear` dynamics over a matching Gaussian state chain.
   It records every recurrence, dynamics, categorical, Gaussian, and boundary
   owner needed for dispatch and free-energy accounting.
2. Self-loops, parallel or ambiguous ownership, cardinality or dimension
   disagreement, unsupported equality topology, recurrence/component
   P-substitution, and mixed objective ownership fail explicitly or keep the
   documented D3/D9 path without a D10 claim.
3. Engine state stores one authoritative D10 coordinate for the component.
   Flooding and scheduled execution, host and JIT, and single and vectorized
   runs return the same categorical singleton/pair and Gaussian marginal/lag
   beliefs within dtype-scaled tolerance.
4. The engine uses D10.6 once. Replaced internal D3 dynamics and D9 recurrence
   node terms and separator corrections do not double count. K=2 and K=3
   objective values match the cff-bmax authority.
5. Status distinguishes converged, budget exhausted, and diverged from the
   installed coordinate's finite objective plus categorical and Gaussian
   residuals. A non-finite value is never returned as success.
6. A documented `Engine` readout API returns source-row/destination-column
   pairs, singleton masses, Gaussian means/covariances, and lag moments from
   the installed D10 state. It does not reconstruct a second D3 or D9 belief.
7. Fixed-drive D9 and K=1 Gaussian reductions, exact-zero support, float32,
   non-finite/improper input, K=8, and T>=100 retain the cff-bmax behavior.
   Runtime work is O(T*K^2 + T*d^3), with no categorical path enumeration.
8. README, ARCHITECTURE.md, DERIVATIONS.md, and LIMITATIONS.md describe the
   production eligibility and claim boundary in place. D10 remains ordinary
   BFE; component-owned GFE/P-substitution is rejected unless separately
   derived.

## Methodological Suggestions — HOW

These suggestions are optional. The implementer may choose another sound
method.

- Add derived `StructuredHybridPlan` metadata while leaving serialized factor
  names, roles, and parameter blobs unchanged.
- Land an unfused implementation first. Fuse the categorical or Gaussian
  scans only if profiling justifies it and parity with the unfused authority
  remains exact.
- Treat paired dispatch surfaces as one change path: compiler arena packing,
  synchronous sweep, mixed schedule, batch/scan helpers, objective builder,
  status, and public belief readout.
- Add a topology test that perturbs one valid rSLDS chain in each rejected way
  so eligibility cannot broaden accidentally.

## Future work

cff-6rav makes D10 beliefs authoritative for learning, persistence, and rSLDS
fit helpers. Experiment requalification remains in cff-ngxx and cff-h0hx.
This ticket does not change checkpoint compatibility or run NASCAR data.

## Definitions

- **Derived component plan:** Compiler metadata reconstructed from the
  serialized graph. It is not a new serialized model format.
- **Energy owner:** The one runtime component responsible for a free-energy
  term and its entropy corrections.
- **Unfused implementation:** Correct component updates executed as separate
  operations before combining them into a performance optimization.

Architecture handoff: use the **New derived component coordinate** row in
`docs/ARCHITECTURE.md`. Compiler topology/plan, arena ownership,
synchronous/scheduled dispatch, host/JIT, single/`vmap`, objective replacement,
status, and documented readout all apply. Serialized IR changes are N/A.
Learning installation, checkpoint migration, and experiment consumers are N/A
and belong to cff-6rav or cff-ngxx.

## Notes

**2026-08-22T21:10:05Z**

Implementation start after the pure D10 authority passed the full short suite. Production eligibility is deliberately narrow: the exact acyclic recurrence/equality/destination-dynamics/Gaussian chain with tied x0 and shared emission only. The engine must install and score one authoritative component coordinate, suppress every replaced local energy owner exactly once, and retain D3/D9 fallback for unsupported graphs.

**2026-08-22T22:23:20Z**

**Why**

Fixed-parameter recurrent switching graphs needed one production owner for the D10 categorical-path and Gaussian-trajectory coordinate, rather than separate D3 and D9 updates and ledgers.

**What landed**

- The compiler derives a name-independent D10 plan only for the proven whole-template tied-boundary topology. Equality incidence cycles, state-root reuse, dynamics self-closures, duplicate emission or observation ownership, constraints, extra attachments, and declared learning retain the documented fallback.
- Engine and Colony install and execute one unfused D10 coordinate with source-row/destination-column pairs, Gaussian marginal and lag moments, literal support, separate residuals, and converged/budget-exhausted/diverged status. The D10 objective replaces all owned initial, emission, dynamics, recurrence, equality, and separator ledger terms exactly once.
- Public point-parameter writes mark derived D10 state stale. Step, run, BFE, readout, and Colony require reset or rebinding consistently. Missing DATA rebinding clears stored payloads as well as masks.
- Serialized IR, recurrence learning, and persistence formats remain unchanged.

**How verified**

Independent read-only review cleared exact head 5d0b209 after cycle, duplicate-time, objective freshness, stale-state, Colony divergence, independent K2/K3 extraction, and nonfinite-to-missing regressions. Focused affected selection passed 99 tests with 3 deselected before the final missing-data fix; the exact post-fix recovery selection passed 3 tests. The final not-slow suite at 5d0b209 passed 1505 tests with 39 deselected and zero failures in 689.562 seconds. Dependency-cycle and diff checks pass.

**Future work — cff-6rav**

- Atomically replace every generic D3/D9 learning readout and fit_rslds _attempt_evidence consumer when learning-enabled D10 dispatch is activated.
- Make every Learnables.write, RSLDSLearning.install, fit installation, and direct replacement invalidate or rebuild the D10 coordinate before inference or readout.
- Preserve source identity and C_t = E[x_t x_(t-1)^T] orientation; recurrence uses source-row/destination-column rho and x_(t-1), dynamics uses destination gamma_t and lag second moments, initial uses gamma_0, and emission uses y with q_X.
- Preserve exact-zero exposure as exact prior retention and atomic four-family installation.
- Replace legacy damping and retry evidence with D10-compatible coordinate semantics; nonzero D10 damping remains rejected until derived.
- Version the executable, belief, statistic, and objective identity and reject D8/D9 resumable learning state before enabling learned dispatch.

**Definitions**

D10 plan means compiler-derived runtime metadata; it is not serialized model structure. A stale coordinate is a valid parameter update paired with derived beliefs from an older parameter point, not numerical divergence.
