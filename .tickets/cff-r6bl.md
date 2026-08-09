---
id: cff-r6bl
status: in_progress
deps: []
links: [cff-yl0a, cff-wbhu, cff-nkbj]
created: 2026-08-01T05:05:47Z
type: feature
priority: 2
assignee: coopwilliams
parent: cff-ahps
tags: [rslds, performance, scheduling]
---
# Reduce canonical rSLDS cold compile and setup cost

## Background in Plain English

The independent NASCAR run for `cff-y4nx` spent `723.14` seconds constructing
and compiling a `T=10,000` rSLDS, compared with `30.58` seconds for converged
inference. Peak resident memory was `2.03` GB. Cold setup is therefore the
dominant cost when a consumer starts this canonical model.

This ticket owns compact lowering for the repeated rSLDS slice made from
`recurrence_link`, `gaussian_mixture_linear`, and `gaussian_linear`. It must
reduce cold cost without changing factor equations, free-energy accounting,
support handling, solver status, or runtime data and parameter behavior.
Ticket `cff-yl0a` owns exact discrete HMM slice fusion, and ticket `cff-wbhu`
owns the linked general device-loop work. Neither is a prerequisite for this
model-specific lowering. The model and benchmark harness from `cff-y4nx` are
already present in this branch, so ticket closure is not an execution
prerequisite.

## Outcomes / Acceptance Criteria

1. A fresh-process benchmark reports separate durations for template
   construction, validation and grounding, CFFG lowering, JAX lowering and
   compilation, inference, readout, and artifact writing.
2. The benchmark reports fresh generic-path and compact-lowered distributions
   at `T=100`, `T=1,000`, and `T=10,000` in the same benchmark run. Both paths
   use the same machine, process-isolation policy, software environment, model
   dimensions, schedule, dtype, observations, parameters, and timing protocol.
3. The benchmark reports timing distributions rather than one unqualified
   duration and names warmup, repetition count, aggregation statistic, and
   process-isolation policy.
4. At `T=10,000`, the lowered-path median from template construction through
   executable readiness is at least five times faster than the fresh generic-
   path median. The historical `723.14` seconds is reported for orientation
   but is not the acceptance denominator.
5. At `T=10,000`, lowered-path median converged-inference latency is no more
   than five percent slower than the fresh generic-path median. The historical
   `30.58` seconds is reported for orientation but is not the acceptance
   denominator.
6. The benchmark records the number and duration of CFFG and JAX compilations
   so a cache hit cannot be reported as a cold compile.
7. The compact lowered path and the generic path return matching messages,
   categorical marginals, Gaussian means and covariances, Bethe free energy,
   support certificate, inner recurrence status, and final `RunStatus`.
   7.1. Support certificates, inner status values, and final status values are
   exactly equal.
   7.2. For each finite message, marginal, mean, or covariance array, parity
   requires
   `max(abs(lowered - generic)) <= 64 * finfo(dtype).eps
   * max(1, max(abs(generic)))`.
   7.3. Bethe free-energy parity requires
   `abs(lowered - generic) <= 64 * N * finfo(dtype).eps
   * max(1, abs(generic))`, where `N` is the number of factor and edge terms
   summed by the compact fixture's energy ledger.
8. Flooding and mixed schedules preserve their documented relationship before
   and after the lowering on the compact parity fixture.
9. With exact-zero categorical support, the generic and lowered paths retain
   the same literal support mask and negative-infinity sentinel positions.
   Their finite messages and Bethe free energies satisfy criterion 7, and
   their support certificates and statuses are exactly equal.
10. An improper Gaussian belief, non-finite free energy, or failed recurrence
    coordinate remains a loud non-converged result with the original failure
    status.
11. Float32 and x64 compile to dtype-specific executables, use dtype-derived
    numerical guards, and satisfy the compact parity criteria independently.
12. Changing observations without changing structure reuses the compiled
    executable and changes the posterior according to an independent compact
    reference.
13. Changing initial-regime, dynamics, recurrence, or emission parameters
    without changing structure reuses the compiled executable and changes the
    posterior according to an independent compact reference.
14. A change to horizon, regime count, latent dimension, observation dimension,
    dtype, or schedule cannot reuse an incompatible compiled executable.
15. Runtime observations and learned parameters remain executable inputs and
    are not captured as constants by lowering or compilation.
16. The compact repeated-slice executor does not contain one separately lowered
    operation body per time step; the benchmark records the staged or scan
    representation used to establish this property.
17. At `T=100`, `T=1,000`, and `T=10,000`, the benchmark sums `nbytes` over the
    actual array leaves retained by `State`, runtime parameters, and the
    compiled compact representation, reports each category separately, and
    demonstrates that no unreported persistent array grows with the horizon.
    The public resource estimator is N/A for this ticket; `cff-j7mm` owns its
    non-discrete correctness and is not a prerequisite for this direct
    measurement.
18. Peak resident memory at `T=10,000` is no more than `2.25` GB on the baseline
    machine and protocol.
19. No `T=10,000` case enters the default test suite; the compact parity and
    hostile-input cases remain short-suite tests, and long timing cases stay in
    an external benchmark.
20. `docs/ARCHITECTURE.md` records the lowering authority, execution surfaces,
    cache-key boundary, and parity contract.
21. `docs/LIMITATIONS.md` records any accepted schedule, model-shape, device, or
    benchmark boundary that remains after the optimization.
22. Closing evidence records repository and experiment commits, commands,
    environment, raw timing samples, aggregation, peak resident memory,
    compilation counts, profiler artifacts, and artifact hashes.

## Methodological Suggestions

These suggestions are optional. The implementer may choose another sound
method that satisfies the outcomes.

- Use the **Runtime numerical or status invariant** change-path row. Treat the
  compiler hybrid and recurrence plans, mixed-stage construction, and
  `Engine` execution and certification as the primary implementation surfaces.
- Inspect flooding, mixed, and exact schedule construction; host and JIT support
  checks; `_make_step`, scheduled batch or scan stages, propagation, Bethe free
  energy, and inner-solver status. Check direct persistent-array accounting and
  benchmark reporting. `RollingAgent` and `Colony` behavior are outside this
  ticket because the rSLDS helper exposes no agent-facing contract.
- Keep serialization, factor equations, message rules, and public parameter
  shapes unchanged. If the chosen method requires one of those changes, split
  that capability into a prerequisite ticket before implementation.
- Use a compact repeated rSLDS slice for executable parity and hostile-input
  tests. Keep `T=100`, `T=1,000`, and `T=10,000` measurements in a fresh-process
  external benchmark.
- Treat single-instance execution as the required surface. Batched `vmap`
  behavior is outside this ticket until a public batched rSLDS contract exists.

## Future work

`cff-yl0a` owns general discrete HMM chain fusion, and `cff-wbhu` owns the
linked device-loop optimization. Their broader execution changes may reuse the
measurements or lowering ideas from this ticket, but neither ticket receives
the rSLDS-specific acceptance work.

`cff-l7y4` owns the observed-only NASCAR fit that benefits from lower setup
cost. `cff-j7mm` independently owns correction of the public resource estimator
for non-discrete arenas; this ticket's direct benchmark accounting neither
implements nor validates that API. `cff-ahps` remains the downstream roadmap
epic.

## Definitions

- **Bethe free energy:** The factor-graph objective reported by the engine and
  used as a convergence signal.
- **Cache key:** The structural and execution attributes that determine whether
  a compiled executable can be reused safely.
- **Cold compile:** Construction and compilation measured in a fresh process
  without a reusable executable from an earlier run.
- **Generic path:** The existing execution route that lowers each factor and
  schedule stage without the rSLDS-specific compact representation.
- **Hostile input:** A sparse-support, improper, float32, or uncomfortable-scale
  case required by the numerical robustness contract.
- **Lowering:** Conversion of the compiled factor graph and schedule into JAX
  operations before executable compilation.
- **Peak resident memory:** The largest amount of physical memory attributed to
  the benchmark process.
- **rSLDS:** Recurrent switching linear dynamical system, whose regime
  transition depends on the continuous latent state.
- **Support certificate:** The runtime result of checking whether exact-zero
  probability constraints remain feasible.
- **Warm inference:** Repeated inference after the executable has already been
  compiled.

## Notes

**2026-08-08T16:13:45Z**

Rewrite handoff, 2026-08-08. cff-r6bl owns the measured cold rSLDS path from model construction through executable readiness, plus semantic parity, cache-key behavior, scale measurements, and memory evidence. cff-yl0a and cff-wbhu remain linked parallel work rather than prerequisites; cff-y4nx supplies the model and baseline.

Legend: prerequisite --> dependent

    [~] cff-twzc (ready) --> [~] cff-y4nx (blocked) --> [ ] cff-r6bl

    Parallel linked work: [ ] cff-yl0a (ready)   [ ] cff-wbhu (ready)
    Roadmap parent: cff-ahps

The tk dependency records are authoritative. Validation used the bounded historical slice, two bounded scans, tk dep tree, tk ready, tk blocked, tk dep cycle, and git diff --check. Retained scan candidates and meanings: canonical=literature_standard; fixed=constant_parameter; generic=model_general existing path; preserve=semantic_invariant; public=documented_contract; reusable=compiled_artifact reuse; same=equivalent_configuration or value_equality; expose=explicit denial of an agent-facing public API; implement=planned_work or internal_component wiring; support=probability_support or runtime support check; complete=ticket-review completion marker; ready=dependency_ready; remaining=preserved_state or deliberately_excluded_work; close=acceptance_completion evidence; certify=runtime_certificate; check=automated_check or support guard; converge=runtime_status_criterion; estimate=resource_estimator; evidence=provenance_artifact; exact=literal_zero_or_support; match=tolerance_parity; parity=behavioral_parity; robust=the AGENTS.md numerical contract; sound=reasonable_judgment; test=test_artifact or automated_execution; validate=component_validation; fit=the delegated observed-only workflow; infer=fixed-parameter state inference; learn=runtime learned-parameter input; performance=the ticket category covering named cold, warm, and memory metrics; scale=domain_size; speed=cold_latency or named arena growth bound; can=theoretical safe cache reuse; may=permission for an alternative method or linked reuse; explicit_owner and named_ticket_handoff=the ticket or component named in each sentence; any=universal accepted-boundary set; only=the observed-only phase boundary; existing=predates_change. Capability ownership was checked separately. Intelligible-ticket review: complete.

**2026-08-08T16:16:48Z**

Intelligible-ticket review: complete.

**2026-08-08T18:12:00Z**

Readiness resolution, 2026-08-08. Acceptance speed and regression ratios now compare fresh generic and lowered distributions from the same protocol; the historical single measurements are orientation only. Numeric parity uses explicit dtype-derived bounds, exact support/status equality, and direct persistent-array byte accounting. The public estimator is delegated to `cff-j7mm`, so `cff-r6bl` neither repairs it nor depends on it. The model, harness, and baseline artifacts are already in the branch, so the obsolete closure dependency on `cff-y4nx` was removed and this ticket may proceed in parallel.

Legend: prerequisite --> dependent

    [ ] cff-r6bl (ready) --> [ ] cff-ahps

    Parallel linked work: [ ] cff-yl0a   [ ] cff-wbhu

The prior retained-candidate classifications remain applicable; additions and changed readings are n_a=the estimator surface delegated to `cff-j7mm`, correct=public-estimator correctness, generic=the freshly measured pre-optimization path, same=the controlled benchmark configuration, exact=literal support/status equality, match and parity=the stated dtype-derived comparisons, and ready=dependency readiness after removing `cff-y4nx`. Capability ownership was checked separately. Intelligible-ticket review: complete.
