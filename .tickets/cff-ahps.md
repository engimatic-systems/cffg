---
id: cff-ahps
status: open
deps: [cff-cfqg, cff-uy48]
links: [cff-dqk2, cff-1o72, cff-o7hs, cff-9xmq, cff-bxk4, cff-6s31, cff-0f3g, cff-dlrj]
created: 2026-07-08T16:48:44Z
type: epic
priority: 1
assignee: coopwilliams
---
# Epic: Integrated extensible-kernel baseline

## Background in Plain English

The architecture already states the correct boundary: each factor and
constraint combination needs its own derived update rule, so CFFG uses an
extensible rule registry rather than a universal solver. This ticket previously
used the title "universal factor library" and made nonlinear factors, neural
factors, plate scaling, visualization, and other breadth prerequisites of one
roadmap closure. That completion rule contradicted the finite boundary stated
by the architecture and kept the milestone open whenever another useful family
was imagined.

This epic now closes on a finite integration baseline. Representative
discrete, Gaussian, exponential-family, and hybrid paths must share their
declared execution and objective contracts, reject unsupported combinations
loudly, survive the numerical robustness cases, and remain usable through the
paper/release and engine-trustworthiness gates. New mathematical families stay
as explicit downstream choices.

## Outcomes / Acceptance Criteria

1. `cff-cfqg` is closed with the existing engine's runtime, learning,
   execution seams, estimator, diagnostics, and maintainability gates
   satisfied.
2. `cff-uy48` is closed with the bounded paper-faithfulness and minimum public
   workflow evidence satisfied.
3. One representative discrete path builds, compiles, runs, reports its node
   and edge objective terms, exposes its documented readout, survives JSON and
   bundle round trips, and reports its named resource costs.
4. One representative Gaussian path completes the same integration surfaces
   under its analytic message and objective contract.
5. One representative scalar exponential-family path completes the same
   integration surfaces through the executable `FactorKernel` registry.
6. One representative discrete-Gaussian hybrid path completes its supported
   compiler, message, objective, readout, serialization, learning, and resource
   surfaces.
7. The discrete, Gaussian, and scalar exponential-family examples use the
   shared registered execution contract delivered by `cff-209l`.
8. A specialized hybrid or constraint plan states why it cannot use a
   single-arena kernel contract and has one named compiler, execution, and
   objective owner.
9. Every supported example preserves exact zero support, applies its declared
   vacuous-message rule, rejects or reports improper beliefs truthfully, runs
   in float32 with dtype-derived guards, and survives a case 10–100 times its
   smallest fixture.
10. Compilation rejects an unregistered factor and constraint combination
    before inference and names the missing executable contract.
11. A factor that uses an approximation names the approximation, the regime in
    which it applies, the ledger quantity it reports, and an independent
    validation bound or diagnostic.
12. Sum-product results report BFE or GFE with the family-specific meaning
    documented by the ledger. Max-product results report a decoded objective
    and are not labeled free energy.
13. README, architecture, derivations, and limitations agree on the finite
    executable vocabulary, specialized-plan exceptions, public extension
    surface, and unsupported combinations.
14. Closing this epic means the extensible-kernel baseline is integrated and
    trustworthy. It does not mean all probability distributions, constraints,
    or neural functions are executable.

## Methodological Suggestions

These suggestions are optional. An implementer may choose another sound method
that satisfies the numbered outcomes with equivalent evidence.

- Use the representative fixtures already owned by the HMM, Kalman,
  scalar-EF, and switching/hybrid test tiers. This epic should reconcile their
  integration evidence rather than add another model family.
- Walk the **New factor or rule family** change path across compiler grouping,
  messages, objective terms, observations, readouts, serialization, estimator,
  and public exports.
- Treat a specialized multi-arena plan as an explicit extension point with a
  smaller contract, not as a failure to force every family through one callback
  shape.
- Close from evidence preserved by the two prerequisite epics. Do not satisfy a
  missing integration surface by adding nonlinear or neural scope here.

```text
Legend: prerequisite --> dependent

[ ] cff-cfqg  engine trust and central seams --+
                                                +--> [ ] cff-ahps
[ ] cff-uy48  paper and minimum release -------+

Parked breadth selected later from evidence:
[ ] cff-o7hs --> [ ] cff-1o72     nonlinear --> neural
[ ] cff-6s31                       compact plate lowering
[ ] cff-9xmq                       balanced broadcast topology
[ ] cff-0f3g                       balanced-tree recurrence
```

## Future work

- `cff-o7hs` and `cff-1o72` retain nonlinear and neural factor work. They are
  not completion requirements for this epic. `cff-o7hs` depends on
  `cff-cfqg`, so that breadth cannot start before the central seams improve.
- `cff-6s31` retains compact plate lowering, `cff-9xmq` retains balanced
  broadcast topology, and `cff-bxk4` retains Gaussian visualization.
- `cff-0f3g` and `cff-dlrj` remain separate rSLDS recurrence research and
  implementation choices.
- Promoting any parked branch requires measured user or scientific demand and
  its own derivation, objective meaning, hostile-input contract, and extension
  path.

## Definitions

- **Executable vocabulary:** The finite set of factor and constraint
  combinations that compile to registered or specialized runtime rules.
- **Extensible registry:** A mapping from a declared factor and constraint
  combination to its validated message, objective, observation, and readout
  implementations.
- **Integration baseline:** Evidence that representative existing families
  traverse all supported compiler, runtime, objective, persistence, and public
  surfaces.
- **Specialized plan:** A compiler and runtime contract for a construction that
  spans arenas or has semantics not represented by one ordinary factor kernel.
- **Universal solver:** A system that derives executable updates for arbitrary
  factor and constraint combinations without a registered family-specific
  derivation. CFFG makes no such claim.

## Notes

**2026-08-13T04:37:28Z**

Roadmap refinement, 2026-08-12. Legend: prerequisite --> dependent.

[ ] cfqg engine trust --+
                        +--> [ ] ahps finite integration baseline
[ ] uy48 paper/release -+

Ready roots live in the two gate epics. Nonlinear/neural, compact plates, balanced broadcast, visualization, and recurrence branches are linked optional work, not closure dependencies.
