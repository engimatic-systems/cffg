---
id: cff-e10u
status: open
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
