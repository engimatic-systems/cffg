---
id: cff-9vab
status: open
deps: [cff-4c7t]
links: []
created: 2026-08-10T02:58:17Z
type: task
priority: 2
assignee: coopwilliams
parent: cff-0f3g
tags: [rslds, validation, performance]
---
# Integrate and benchmark balanced-tree rSLDS models

## Background in Plain English

cff-lv65 and cff-4c7t make balanced-tree recurrence executable and learnable.
This ticket finishes the consumer workflow and measures the tradeoffs that
motivated the topology. A shorter path does not automatically make a complete
rSLDS faster or easier to learn, so the repository must report observed
runtime, memory, convergence, and statistical behavior rather than promise a
speedup from tree depth alone.

## Outcomes / Acceptance Criteria

1. The public rSLDS builder accepts an explicit recurrence topology and keeps
   decision list as the documented compatibility default.
2. Canonical constructors expose decision-list and deterministic balanced
   topologies without asking callers to construct internal index tables.
3. Builder validation relates `K`, topology leaves, coefficient owners,
   weights, offsets, priors, and learning state before compilation.
4. JSON and `.cffgz` round trips reproduce the same builder-selected topology,
   arena structure, parameters, observations, messages, statuses, and Bethe
   free energy.
5. Fixed-parameter balanced rSLDS inference matches the pinned pypolyagamma
   balanced-tree transition normalization after an explicit source, leaf, and
   branch mapping.
6. Compact observed-only balanced fitting updates initial regime, dynamics,
   emissions, and recurrence gates without generating truth and returns a loud
   complete status.
7. At least three leaf-label permutations produce the probabilities implied by
   their serialized labels rather than an accidental array order.
8. A registered comparison uses the same data, initialization information,
   parameter budget accounting, stopping rule, and metrics for decision-list
   and balanced CFFG arms.
9. The comparison measures setup phases, inference and learning phases, peak
   resident memory, outer and inner work, convergence status, held-out
   predictive likelihood, aligned latent RMSE, and aligned regime accuracy.
10. Scale points include `K=4`, `K=8`, `K=16`, and `K=32` with a named horizon
    and drive dimension. Results report gate count, maximum and mean path
    depth, persistent bytes, and observed wall time.
11. The report states whether balanced topology improves, matches, or worsens
    each measured outcome. It does not infer whole-model speed solely from
    logarithmic path depth.
12. Exact-zero support, improper inputs, float32 execution, and the `K=32`
    case retain the recurrence numerical contract and loud statuses.
13. Compact deterministic integration tests remain in the default suite. Long
    multi-trial and scaling runs are marked slow and store reproducible
    artifacts outside the package repository when appropriate.
14. Artifacts record code and dependency commits, environments, commands,
    hardware, seeds, configurations, raw measurements, statuses, and manifest
    hashes.
15. README.md documents topology selection and observed tradeoffs.
    docs/ARCHITECTURE.md records builder-to-compiler ownership.
    docs/LIMITATIONS.md records benchmark scope and every observed failure.

## Methodological Suggestions

These suggestions are optional. The implementer may choose another sound
method that satisfies the outcomes.

- Reuse the NASCAR validation project's provenance and phase-timing schema,
  but preregister this topology comparison separately from cff-nkbj.
- Separate cold setup, JAX compilation, inference, parameter learning, and
  reporting so one phase cannot masquerade as another topology effect.
- Run long configurations sequentially unless memory evidence supports bounded
  parallelism.

## Future work

The cff-0f3g epic closes when this ticket and its prerequisites close. Larger
state spaces, nonbinary trees, learned tree structure, real observations, and
destination-softmax recurrence remain future work.

## Definitions

- **Compatibility default:** The behavior used when an existing caller does
  not request the new option.
- **Mean path depth:** The average number of binary gates from the root to a
  categorical leaf.
- **Parameter budget:** The number and shape of freely learned scalar
  parameters in one comparison arm.
- **Peak resident memory:** The largest amount of physical memory held by the
  process during a measured run.

## Notes

**2026-08-10T03:20:00Z**

Wiggle-audit retained candidates and intended meanings: canonical=the named
decision-list and balanced constructors; fixed=fixed-parameter reference
inference; integrate=the builder and compact end-to-end tests; public=the rSLDS
builder; register=the preregistered comparison; reusable=reuse the named
provenance schema; same=equal builder round trip or matched controls;
expose=public constructors; support=literal support or empirical memory
evidence for bounded parallelism; complete=the whole rSLDS workflow/status;
remaining=long slow tests or explicitly deferred families; appropriate=external
artifact storage when long results should not enter the package repo;
close=completion of cff-0f3g; converge=reported runtime status;
evidence=measured memory; exact=literal zero; match=reference agreement or one
of the three reported comparative outcomes; reproduce=round-trip equality or
rerunnable artifacts; sound=another method satisfying outcomes; test and
validate=the named integration, builder, and experiment checks; fit, infer,
and learn=the observed-only parameter workflow and measured phases; learnable,
trial, performance, scale, and speed=the explicitly named learning capability,
multi-trial slow runs, `K` series, and whole-model runtime claim boundary;
may=optional method; at_least and every=the three permutations and all observed
failures; only=the truth-information boundary; existing=pre-option caller
behavior. Capability owner: cff-9vab owns consumer integration and empirical
tradeoff claims, not the underlying derivation or runtime rule.
Intelligible-ticket review: complete.

Classification-note additions: end_to_end=builder-through-observed-only-fit
execution; check=the builder, reference, or artifact comparison;
intended=the contextual assignments in this note; should=the criterion for
external storage based on long artifact size/runtime; explicit_owner and
named_ticket_handoff=cff-9vab/cff-0f3g; all=the enumerated outcomes and
failures; deferred=larger or different topology families.
