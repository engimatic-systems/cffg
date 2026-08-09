---
id: cff-r6bl
status: closed
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

**2026-08-09T14:24:32Z**

Vision research, 2026-08-09. Question: what endgame in the Obsidian Agents/CFFG notes should inform this optimization without becoming a specification? Answer: the notes envision a reusable serialized CFFG/LAIF world-model that repeatedly accepts live observations, learned parameters, goals, and action affordances while preserving quantified beliefs, true BFE, modular inference, and resource visibility. For cff-r6bl, fast structure-derived setup, dynamic runtime inputs, exact status/parity, and explicit timing/memory accounting serve that direction; for cff-nkbj, multi-seed provenance and loud failures build trust in the learned-model boundary. Commercial examples and LLM/exomemory integration remain motivation only and do not broaden either ticket. Sources: /Users/ravd/Documents/01-obsidian-vaults/agents/CFFG/0.0.md and /Users/ravd/Documents/01-obsidian-vaults/agents/CFFG/Commercial application Brain dump.md. Confidence: high on the note synthesis; medium on ticket implications because they are an inference from aspirational material.

**2026-08-09T15:29:52Z**

Closing handoff, 2026-08-09. Root cause: profiling disproved the assumed per-slice JAX lowering defect. The ordinary arena already lowers the rSLDS to seven shape-batched operation bodies independent of horizon. Template validation and compiler grounding instead called whole-template degree and ports_of scans for each variable, producing quadratic host cost. CFFG commit 037148f builds transient degree and port incidence indexes in linear passes; commit ca295f9 records the authority, cache/runtime boundary, evidence, and limitations. No serialization, factor equation, schedule, state shape, message rule, BFE term, status, observation input, or named parameter input changed.

Evidence: sibling experiment code commit fc2287d and artifact commit 1b1b714. Command: uv run cffg-rslds-cold-benchmark --cffg-repository /Users/ravd/projects/cffg --generic-commit 2337161 --optimized-commit 037148f --output artifacts/cold-lowering. Protocol: Apple M2 Max CPU, x64, Python 3.12.13, JAX 0.11.0, one fresh worker per path/horizon/repetition, three repetitions, median aggregation, one uncached status-bearing sweep, auto/mixed schedule, seed 0, K=4, latent d=2, observed dy=10, tolerance 1e-6, damping 0.9. Environment SHA-256 cebb68bfe18cd76ef57eb4babcaac217217563863f5a73e551aa79eecb8db2d6; results SHA-256 5432c42ba6795850898e878ebdfad39ef82f8cc7dbf83ddc6f98c8ed3d4651db; complete manifest SHA-256 515bd5e2dc05741953e225912f62c8d7b0136a3d1d6b9855938b117bcd7f033b.

Measured gates: at T=100, 1,000, and 10,000, generic setup medians were 2.674, 10.412, and 741.361 seconds; indexed medians were 2.602, 3.290, and 24.060 seconds. The T=10,000 setup speedup was 30.813x. Median converged inference was 27.342 versus 27.020 seconds, ratio 0.988. All workers compiled one CFFG arena and three named JAX executables and converged in 233, 243, and 248 iterations by horizon. Every compared array had zero maximum difference; exact support masks, readout names, inner status, final status, and BFE were identical. The optimized T=10,000 peak RSS maximum was 2,092,040,192 bytes, below 2.25 GiB. Direct T=10,000 array leaves were reported separately: State 25,199,360 bytes, runtime parameters 16,479,472, compiled representation 7,119,400, benchmark inputs 801,648, and readout/artifact arrays 42,958,488. No unreported persistent array leaf grows with horizon.

Profiler artifacts: generic d7448a4bd77581f533bc6b9a3e5ac9d68fdf7d618a0e4c79865c5f8fec9f0a4b and indexed f67bf0e57e9f4ff231d28312e69b42910213dae0021266399a793f38692592bd. At T=1,000 under cProfile, old validation consumed 17.372 cumulative seconds and 8,992 Template.degree calls consumed 17.220; neither appears among the indexed top cumulative costs.

Short semantic anchors remain on the unchanged execution path: test_rslds_tier_matches_tiny_exact_oracle_and_rolls_all_parameters covers float32, an independent quadrature oracle, observation binding, and all rollover parameter families; test_flood_and_mixed_schedules_reach_the_same_bound_root covers schedule parity; test_exact_zero_support_is_preserved covers literal support; test_vacuous_and_improper_drive_are_loud_after_transient_flooding covers improper and non-finite failure; parameter, structure-hash, shape-validation, dtype, and doctored-schedule tests cover dynamic values and incompatible structural boundaries. New tests test_validation_builds_one_incidence_index and test_rslds_compiler_reuses_validation_incidence prevent per-variable rescans. Final validation at proposed head ca295f9: uv run python tools/dashboard.py --run-suite passed 1,336 tests with 39 slow tests deselected and 0 failures; pytest 509.06 seconds, dashboard wall 516.624 seconds. Long timing remained external. Accepted boundaries are CPU/x64, the named rSLDS shape and auto schedule, one instance, and static shape-specific executables; cff-yl0a and cff-wbhu retain broader fusion/device-loop work, and cff-j7mm retains the public non-discrete estimator.

**2026-08-09T15:30:41Z**

Post-handoff wiggle audit, 2026-08-09. The current scan retained 49 candidates and 259 occurrences. Earlier classifications remain applicable. Candidate meanings across the ticket are: canonical=literature-standard named rSLDS; fixed=constant supplied parameter or predetermined budget; generic=the pre-index model-general route; integrate=the aspirational external LLM/exomemory integration; n_a=the estimator delegated to cff-j7mm; preserve=semantic invariant or literal support preservation; public=documented API contract; reusable=shape-safe compiled executable reuse; same=controlled configuration or value equality; expose=explicit absence of a public agent boundary; implement=internal wiring or planned work; support=probability support or runtime support check; complete=artifact or acceptance completeness; ready=executable or dependency readiness; remaining=preserved state or delegated work; close=ticket acceptance completion; certify=runtime certificate; check=automated check or support guard; converge=runtime status criterion; correct=public estimator correctness; estimate=resource estimator; evidence=provenance artifact; exact=literal support, status, or value equality; match=dtype-scaled comparison; oracle=independent analytical reference; parity=behavioral equivalence under the named protocol; pass=automated test or registered gate; robust=the AGENTS.md numerical contract; sound=bounded implementer judgment; test=automated execution or artifact; validate=component or artifact validation; fit=observed-only parameter fitting; infer=hidden-state inference; learn=runtime learned-parameter input; rollover=between-trial point-parameter installation; performance=the named setup, inference, and memory measures; scale=horizon or domain size; speed=the named setup ratio; can=demonstrated safe capability; may=optional method or linked reuse permission; should=methodological recommendation; explicit_owner=the component or ticket named in its sentence; named_ticket_handoff=the named owner transfer; all=the locally enumerated workers, paths, arrays, or families; any=the universal accepted-boundary set; at_least=the inclusive 5x threshold; every=each member of the locally listed comparison set; only=the observed-only or ticket-scope boundary; existing=the route or contract that predates this change. Capability ownership was checked separately: core validation owns the port invariant, compile_template owns transient arena grounding indexes, Engine and factor rules remain unchanged, the sibling harness owns performance/parity evidence, cff-yl0a and cff-wbhu own broader execution optimization, and cff-j7mm owns the public estimator. Intelligible-ticket review: complete.
