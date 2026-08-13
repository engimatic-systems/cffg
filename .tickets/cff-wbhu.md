---
id: cff-wbhu
status: open
deps: [cff-j4mu, cff-d56d, cff-44we]
links: [cff-fxmj, cff-5edo, cff-d56d, cff-yl0a, cff-rqqc, cff-weff, cff-lv2w, cff-g7rc]
created: 2026-07-10T03:34:54Z
type: feature
priority: 2
assignee: coopwilliams
parent: cff-ahps
---
# Colony throughput: jitted sweep loop (lax.scan) — dispatch-bound batching measured at 1.5x

## Background in Plain English

`Engine.run` performs iterative inference by calling a compiled message update
from a Python loop. `Colony.run` stacks compatible instances and vectorizes one
message update, but Python still launches that update and copies status data to
the host after each sweep. A recorded CPU experiment therefore found only a
1.5× throughput gain for 32 Colony instances even though the vectorized update
itself worked correctly. The launch and synchronization cost, rather than the
message arithmetic, dominated the batch.

Three completed prerequisites make the loop safe to compile. cff-j4mu defined
truthful converged, budget-exhausted, diverged, and infeasible results. cff-d56d
made the selected schedule authoritative and removed repeated flooding from
acyclic graphs. cff-44we made the batched state arenas scale with actual domain
sizes instead of global padding. Closed mixed-GFE work also supplies compiled
matrix-free trust kernels, but its active-set and status controller still runs
on the host for each instance.

This ticket moves deterministic iterative loop control and per-instance status
carry onto the device. An ordinary run should launch one compiled loop per
soft or hardening phase instead of one kernel per sweep. A same-structure Colony
batch should carry each member's state, trace, status, and stopping decision in
that loop without allowing one failed or early-converged member to affect its
peers. The fixed-support mixed-GFE controller should use the same approach
without changing its root, support, inner-solver, or free-energy certificates.
Public results and schedule semantics remain unchanged; the measurable outcome
is at least an 8× warm throughput gain for the recorded 32-instance workload
with no single-engine regression.

## Outcomes / Acceptance Criteria

### 1. Public execution contracts remain stable

- [ ] 1.1 `Engine.run` retains its current positional arguments.
- [ ] 1.2 `Engine.run` retains its current keyword arguments.
- [ ] 1.3 `Colony.run` retains its current positional arguments.
- [ ] 1.4 `Colony.run` retains its current keyword arguments.
- [ ] 1.5 `RunResult.state` retains its current meaning.
- [ ] 1.6 `RunResult.status` retains its four-value public vocabulary.
- [ ] 1.7 `RunResult.iterations` retains its logical-update meaning.
- [ ] 1.8 `RunResult.message_deltas` retains its performed-update ordering.
- [ ] 1.9 `RunResult.free_energies` retains its defined-ledger-only ordering.
- [ ] 1.10 `RunResult.residual` retains its status-specific meaning.
- [ ] 1.11 Every mixed-GFE diagnostic field retains its current public meaning.
- [ ] 1.12 `RunResult.require_converged()` rejects every nonconverged status.
- [ ] 1.13 `Colony.last_results` retains one result per live instance after a
  run.
- [ ] 1.14 Colony instance identifiers retain their current lifetime behavior.
- [ ] 1.15 Colony structure grouping remains keyed by template structure hash.
- [ ] 1.16 One cached `Engine` remains shared by instances with one structure
  hash.
- [ ] 1.17 Instances with different structure hashes remain separate execution
  groups.
- [ ] 1.18 This ticket adds no serialized template field.
- [ ] 1.19 This ticket adds no serialized runtime-state field.
- [ ] 1.20 This ticket changes no checkpoint schema.

### 2. Ordinary Engine iteration uses compiled loop control

- [ ] 2.1 Deterministic iterative `Engine.run` has one compiled controller-loop
  authority.
- [ ] 2.2 The controller carry contains the current message state.
- [ ] 2.3 The controller carry contains the performed-iteration count.
- [ ] 2.4 The controller carry contains the active termination code.
- [ ] 2.5 The controller carry contains the final residual.
- [ ] 2.6 The controller carry contains the previous finite free energy when the
  selected objective uses BFE or GFE.
- [ ] 2.7 The controller carry records one message delta for each performed
  update.
- [ ] 2.8 The controller carry records one free-energy value for each performed
  update whose ledger is defined.
- [ ] 2.9 A normal flooding run contains no Python loop over sweeps.
- [ ] 2.10 A normal colored-schedule run contains no Python loop over outer
  sweeps.
- [ ] 2.11 A normal non-GFE mixed-schedule run contains no Python loop over outer
  sweeps.
- [ ] 2.12 A normal max-product run contains no Python loop over sweeps.
- [ ] 2.13 The compiled loop invokes the selected `_iter_step` implementation.
- [ ] 2.14 The compiled loop does not reimplement any factor message rule.
- [ ] 2.15 The compiled loop does not reimplement schedule selection.
- [ ] 2.16 A repeated call with the same execution signature reuses its compiled
  controller.
- [ ] 2.17 No compilation occurs once per performed sweep.
- [ ] 2.18 Device submission count is independent of the requested sweep budget
  within one ordinary phase.
- [ ] 2.19 Host conversion of traces occurs after the compiled phase returns.
- [ ] 2.20 Public trace tuples contain no masked buffer entries after host
  conversion.
- [ ] 2.21 The compiled loop stores no history of complete runtime states.
- [ ] 2.22 Single-engine trace-buffer storage grows linearly in the requested
  iteration budget.

### 3. Compiled stopping preserves the runtime truthfulness contract

- [ ] 3.1 A finite sum-product run reports `CONVERGED` only after both its
  free-energy residual and message delta satisfy the effective tolerance.
- [ ] 3.2 A finite max-product run reports `CONVERGED` only after its message
  delta satisfies the effective tolerance.
- [ ] 3.3 A converged max-product run requires a finite decoded objective.
- [ ] 3.4 `tol=None` consumes the requested ordinary sweep budget unless a
  failure status stops the logical run.
- [ ] 3.5 A successful ordinary `tol=None` run reports `BUDGET_EXHAUSTED`.
- [ ] 3.6 A tolerance below float32 resolution is raised to the existing
  dtype-derived effective tolerance.
- [ ] 3.7 A tolerance below float64 resolution is raised to the existing
  dtype-derived effective tolerance.
- [ ] 3.8 A non-finite input state reports `DIVERGED` before one logical update.
- [ ] 3.9 A non-finite message delta reports `DIVERGED` on the update that
  produced it.
- [ ] 3.10 A non-finite state value reports `DIVERGED` on the update that
  produced it.
- [ ] 3.11 A non-finite inner residual reports `DIVERGED` on the update that
  produced it.
- [ ] 3.12 An inner-solver divergence reports `DIVERGED` on the update that
  exposed it.
- [ ] 3.13 Proven empty exact support reports `INFEASIBLE`.
- [ ] 3.14 An infeasible run records positive-infinite free energy.
- [ ] 3.15 Inner budget exhaustion remains distinguishable from outer
  convergence.
- [ ] 3.16 A continuous or exponential-family ledger that never becomes defined
  reports `DIVERGED` after a nonempty attempted run.
- [ ] 3.17 `n_iters=0` preserves the current zero-update result for a finite
  feasible state.
- [ ] 3.18 A poisoned `n_iters=0` input still reports its actual failure status.
- [ ] 3.19 Logical termination freezes the returned state at the terminating
  update.
- [ ] 3.20 Logical termination excludes later masked iterations from every
  public counter.
- [ ] 3.21 A negative `n_iters` fails before controller execution.
- [ ] 3.22 An unknown `outer_solver` value fails before controller execution.
- [ ] 3.23 Nonzero damping remains rejected for the trust-region mixed-GFE path.

### 4. Hardening, priming, and schedule paths retain their boundaries

- [ ] 4.1 A colored or mixed forest-prime function executes at most once before
  its iterative phase.
- [ ] 4.2 Forest priming does not increment `RunResult.iterations`.
- [ ] 4.3 A run without delta constraints retains `harden_last=0` as its
  default.
- [ ] 4.4 A run with delta constraints defaults `harden_last` to
  `n_iters // 2`.
- [ ] 4.5 A soft phase never fires a delta hardening update.
- [ ] 4.6 A hardening phase always fires its declared delta hardening update.
- [ ] 4.7 Convergence in the soft phase does not skip a requested hardening
  phase.
- [ ] 4.8 A requested soft phase uses one compiled loop invocation.
- [ ] 4.9 A requested hardening phase uses one compiled loop invocation.
- [ ] 4.10 The number of device loop invocations remains constant as each phase's
  sweep budget grows.
- [ ] 4.11 An automatic two-pass schedule continues to use `_run_exact`.
- [ ] 4.12 A two-pass schedule continues to execute one directed inward and
  outward pass.
- [ ] 4.13 A two-pass schedule continues to certify its fixed point.
- [ ] 4.14 A two-pass schedule with `tol=None` continues to report `CONVERGED`.
- [ ] 4.15 Explicit flooding continues to select the ordinary compiled loop.
- [ ] 4.16 Damping continues to select the schedule path permitted by the
  current runtime contract.
- [ ] 4.17 `outer_solver="picard"` continues to identify the diagnostic legacy
  outer iteration.
- [ ] 4.18 The compiled controller does not change `Schedule.describe()`.
- [ ] 4.19 The compiled controller does not change schedule fingerprints.
- [ ] 4.20 A negative `harden_last` fails before controller execution.
- [ ] 4.21 A `harden_last` greater than `n_iters` fails before controller
  execution.

### 5. Colony batches one structure without host sweep synchronization

- [ ] 5.1 Colony stacks the states of one ordinary same-structure group before
  iterative execution.
- [ ] 5.2 Colony applies one vectorized iteration body to that stacked state.
- [ ] 5.3 Colony carries one active flag per instance on the device.
- [ ] 5.4 Colony carries one termination code per instance on the device.
- [ ] 5.5 Colony carries one performed-iteration count per instance on the
  device.
- [ ] 5.6 Colony carries one residual per instance on the device.
- [ ] 5.7 Colony carries one previous finite free energy per applicable instance
  on the device.
- [ ] 5.8 Colony records message-delta traces per instance on the device.
- [ ] 5.9 Colony records defined free-energy traces per instance on the device.
- [ ] 5.10 Colony contains no Python loop over ordinary batched sweeps.
- [ ] 5.11 Colony performs no per-sweep `np.asarray` conversion for ordinary
  batched execution.
- [ ] 5.12 Colony performs no per-sweep Python `float` conversion for ordinary
  batched execution.
- [ ] 5.13 Colony performs no per-sweep host `active.any()` synchronization.
- [ ] 5.14 An inactive instance retains its state while active peers continue.
- [ ] 5.15 An inactive instance retains its trace length while active peers
  continue.
- [ ] 5.16 An inactive instance retains its termination status while active
  peers continue.
- [ ] 5.17 A divergent instance does not mark a finite peer as diverged.
- [ ] 5.18 An infeasible instance does not mark a feasible peer as infeasible.
- [ ] 5.19 An early-converged instance does not stop an unconverged peer.
- [ ] 5.20 A budget-exhausted instance does not overwrite a converged peer's
  result.
- [ ] 5.21 Batched host conversion truncates each delta trace to that instance's
  performed count.
- [ ] 5.22 Batched host conversion truncates each energy trace to that instance's
  defined-energy count.
- [ ] 5.23 `Colony.last_results` receives each reconstructed per-instance
  `RunResult`.
- [ ] 5.24 Each live Colony instance receives its own returned final state.
- [ ] 5.25 A one-instance structure group remains numerically equal to direct
  `Engine.run`.
- [ ] 5.26 A multi-instance exact two-pass group executes the selected exact
  schedule rather than repeated `_iter_step` flooding.
- [ ] 5.27 A multi-instance exact two-pass result reports one logical iteration
  per member.
- [ ] 5.28 A multi-instance delta-constrained group hardens throughout all
  requested iterations, matching Colony's current behavior.
- [ ] 5.29 Regrouping after `kill` preserves the surviving instance-to-state
  mapping.
- [ ] 5.30 Different structure groups may stop after different logical counts.
- [ ] 5.31 Colony trace-buffer storage grows linearly in batch width.
- [ ] 5.32 Colony trace-buffer storage grows linearly in the requested iteration
  budget.
- [ ] 5.33 Colony stores no per-iteration history of complete batched states.

### 6. Mixed-GFE controller batching preserves proof-backed certificates

- [ ] 6.1 Deterministic mixed-GFE `Engine.run` moves its fixed-support controller
  iterations into compiled control flow.
- [ ] 6.2 A same-structure deterministic mixed-GFE Colony group batches its
  fixed-support controllers.
- [ ] 6.3 The batched controller vectorizes the existing matrix-free trust
  kernel.
- [ ] 6.4 The batched controller does not introduce a second trust-region
  numerical method.
- [ ] 6.5 Each mixed-GFE instance carries its own chart coordinates.
- [ ] 6.6 Each mixed-GFE instance carries its own frozen residual scale.
- [ ] 6.7 Each mixed-GFE instance carries its own trust radius.
- [ ] 6.8 Each mixed-GFE instance carries its own active support face.
- [ ] 6.9 Each mixed-GFE instance carries its own inner-solver status.
- [ ] 6.10 Each mixed-GFE instance carries its own termination reason.
- [ ] 6.11 Each mixed-GFE instance carries its own accepted-step count.
- [ ] 6.12 Each mixed-GFE instance carries its own rejected-step count.
- [ ] 6.13 Each mixed-GFE instance carries its own outer-map evaluation count.
- [ ] 6.14 Each mixed-GFE instance carries its own trial evaluation count.
- [ ] 6.15 Each mixed-GFE instance carries its own linear-iteration count.
- [ ] 6.16 Each mixed-GFE instance carries its own solver-compilation count.
- [ ] 6.17 `CONVERGED` still requires the complete undamped outer residual to
  satisfy its unchanged threshold.
- [ ] 6.18 `CONVERGED` still requires an unchanged exact-support face through
  certification.
- [ ] 6.19 `CONVERGED` still requires every active inner solver to converge.
- [ ] 6.20 `CONVERGED` still requires a finite stationary free-energy ledger.
- [ ] 6.21 A stationary nonroot remains nonconverged.
- [ ] 6.22 A collapsed trust region remains nonconverged.
- [ ] 6.23 A non-finite trial reports divergence for its own instance.
- [ ] 6.24 An infeasible trial reports infeasibility for its own instance.
- [ ] 6.25 A fixed-support restart preserves the last accepted state.
- [ ] 6.26 A support change remains an explicit boundary between compiled
  fixed-support solves.
- [ ] 6.27 A hardening change remains an explicit boundary between compiled
  fixed-support solves.
- [ ] 6.28 The outer-kernel cache remains keyed by exact support.
- [ ] 6.29 The outer-kernel cache remains keyed by dtype.
- [ ] 6.30 The outer-kernel cache remains keyed by the continuous chart metric.
- [ ] 6.31 The outer-kernel cache remains keyed by hardening phase.
- [ ] 6.32 Dynamic parameters remain runtime inputs to cached kernels.
- [ ] 6.33 Dynamic observations remain runtime inputs to cached kernels.
- [ ] 6.34 The bounded candidate phase still needs an independent full root
  certificate.
- [ ] 6.35 The legacy Picard path remains diagnostic rather than a silent
  fallback.
- [ ] 6.36 A Colony arena with an observation-GFE group selects the compiled
  mixed-GFE controller.
- [ ] 6.37 A Colony arena with a transition-GFE group selects the compiled
  mixed-GFE controller.
- [ ] 6.38 A Colony arena with a hybrid-GFE group selects the compiled mixed-GFE
  controller.
- [ ] 6.39 One compiled mixed-GFE controller subgroup contains only instances
  with equal static outer-chart keys.
- [ ] 6.40 Instances with different exact-support faces enter separate
  fixed-support controller subgroups.
- [ ] 6.41 Mixed-GFE subgrouping preserves each instance identifier's state and
  result ownership.
- [ ] 6.42 One outer-chart key compiles at most one controller kernel per engine
  cache lifetime.
- [ ] 6.43 Device submission count for one fixed-support subgroup is independent
  of that subgroup's trust-iteration budget.

### 7. Sequential and compiled results agree across runtime modes

- [ ] 7.1 A test-only host-loop reference is frozen before production loop
  replacement.
- [ ] 7.2 The host-loop reference is not exported as a second production
  execution authority.
- [ ] 7.3 A fixed-budget float64 categorical run reproduces the reference final
  messages bit-for-bit.
- [ ] 7.4 A fixed-budget float64 categorical run reproduces the reference delta
  trace bit-for-bit.
- [ ] 7.5 A tolerance-stopped float64 categorical run reproduces the reference
  status.
- [ ] 7.6 A tolerance-stopped float64 categorical run reproduces the reference
  performed-iteration count.
- [ ] 7.7 A tolerance-stopped float64 categorical run matches the reference
  free-energy trace at tolerance `1e-12`.
- [ ] 7.8 A max-product run reproduces the reference decoded assignment.
- [ ] 7.9 A max-product run matches the reference objective at a preregistered
  dtype-aware tolerance.
- [ ] 7.10 A Gaussian flooding run matches the reference marginals at a
  preregistered dtype-aware tolerance.
- [ ] 7.11 A Gaussian flooding run matches the reference BFE at the same
  tolerance.
- [ ] 7.12 An exponential-family flooding run matches the reference marginals at
  a preregistered dtype-aware tolerance.
- [ ] 7.13 An exponential-family flooding run matches the reference BFE at the
  same tolerance.
- [ ] 7.14 A recurrence-link run matches the reference marginals at a
  preregistered dtype-aware tolerance.
- [ ] 7.15 A recurrence-link run matches the reference BFE at the same
  tolerance.
- [ ] 7.16 A colored schedule reproduces the reference status.
- [ ] 7.17 A non-GFE mixed schedule reproduces the reference status.
- [ ] 7.18 A delta-hardening run reproduces the reference selected support.
- [ ] 7.19 A benign mixed-GFE run reproduces the reference termination reason.
- [ ] 7.20 A benign mixed-GFE run matches the reference outer residual at a
  preregistered dtype-aware tolerance.
- [ ] 7.21 A benign mixed-GFE run matches the reference BFE at the same
  tolerance.
- [ ] 7.22 Sequential and Colony members return equal statuses for equal initial
  states and run controls.
- [ ] 7.23 Sequential and Colony members return equal iteration counts for equal
  initial states and run controls.
- [ ] 7.24 Sequential and Colony members match final marginals at a
  preregistered dtype-aware tolerance.
- [ ] 7.25 Sequential and Colony members match BFE at the same tolerance.

### 8. Hostile support, dtype, and scale cases remain isolated

- [ ] 8.1 A same-structure Colony batch contains one initially non-finite
  instance.
- [ ] 8.2 That initially non-finite instance reports `DIVERGED` at zero
  iterations.
- [ ] 8.3 A finite peer in that batch remains finite.
- [ ] 8.4 A same-structure Colony batch contains one exact-zero contradiction.
- [ ] 8.5 That contradictory instance reports `INFEASIBLE`.
- [ ] 8.6 A compatible sparse-support peer remains feasible.
- [ ] 8.7 A vacuous categorical instance contributes no relative log
  preference.
- [ ] 8.8 A vacuous Gaussian instance contributes zero information.
- [ ] 8.9 A float32 single-engine run produces finite supported messages.
- [ ] 8.10 A float32 single-engine run produces finite defined BFE.
- [ ] 8.11 A float32 Colony batch produces finite supported messages for every
  feasible member.
- [ ] 8.12 A float32 Colony batch produces finite defined BFE for every member
  whose ledger is defined.
- [ ] 8.13 Float32 early stopping uses the existing dtype-derived tolerance.
- [ ] 8.14 Float32 mixed-GFE certification uses the existing dtype-derived
  residual threshold.
- [ ] 8.15 The controller adds no absolute numeric floor.
- [ ] 8.16 The controller adds no absolute ridge.
- [ ] 8.17 The controller adds no absolute clamp.
- [ ] 8.18 Every numeric guard added by this work derives from the active dtype.
- [ ] 8.19 Every numeric guard added by this work names the invariant it
  protects.
- [ ] 8.20 An uncomfortable ordinary fixture contains 320 same-structure Colony
  instances.
- [ ] 8.21 The uncomfortable ordinary fixture accepts a 600-sweep fixed budget.
- [ ] 8.22 The uncomfortable ordinary fixture completes without a Python loop
  over its 600 sweeps.
- [ ] 8.23 The uncomfortable ordinary fixture returns one status for each of its
  320 instances.
- [ ] 8.24 The uncomfortable ordinary fixture preserves instance-local states.
- [ ] 8.25 A heterogeneous arena fixture includes categorical cardinality 32.
- [ ] 8.26 The heterogeneous arena fixture includes Gaussian dimension 32.
- [ ] 8.27 The heterogeneous arena fixture retains the exact segmented-state
  byte count supplied by cff-44we.
- [ ] 8.28 A 32-member mixed-GFE batch permits different convergence counts
  without recompiling one kernel per member.
- [ ] 8.29 Stochastic `mu3` evaluators remain rejected by Colony until cff-rqqc.
- [ ] 8.30 A deterministic single-engine `mu3` run retains its current status
  semantics if it remains outside the compiled controller.

### 9. Performance evidence measures dispatch removal directly

- [ ] 9.1 A committed benchmark command reproduces the ordinary 32-instance
  Colony workload.
- [ ] 9.2 Before implementation, that command records a warm Colony speedup
  between 1.2× and 1.8× over sequential execution.
- [ ] 9.3 The benchmark uses one deterministic non-GFE structure for all 32
  instances.
- [ ] 9.4 The 32 instances carry distinct declared parameters or observations.
- [ ] 9.5 The benchmark requests 60 fixed-budget ordinary iterative sweeps.
- [ ] 9.6 The benchmark times 32 direct `Engine.run` calls as the sequential
  denominator.
- [ ] 9.7 The benchmark times one `Colony.run` call as the batched numerator.
- [ ] 9.8 The benchmark excludes template compilation from warm inference
  timing.
- [ ] 9.9 The benchmark excludes Colony spawning from warm inference timing.
- [ ] 9.10 The benchmark excludes data binding from warm inference timing.
- [ ] 9.11 The benchmark excludes marginal readout from warm inference timing.
- [ ] 9.12 Every timed device result is blocked until ready.
- [ ] 9.13 The benchmark records the hardware identifier.
- [ ] 9.14 The benchmark records the JAX and jaxlib versions.
- [ ] 9.15 The benchmark records the Python version.
- [ ] 9.16 The benchmark records the active dtype.
- [ ] 9.17 The benchmark records the warmup count.
- [ ] 9.18 The benchmark records the timed repetition count.
- [ ] 9.19 The benchmark reports sequential wall time per 32-instance group.
- [ ] 9.20 The benchmark reports Colony wall time per 32-instance group.
- [ ] 9.21 The benchmark reports the Colony-to-sequential throughput ratio.
- [ ] 9.22 On the recorded Apple M2 Max, 64 GB, CPU-only configuration, the warm
  Colony throughput ratio is at least 8×.
- [ ] 9.23 If that baseline hardware is unavailable, the handoff reports the
  blocker instead of claiming the 8× outcome from different hardware.
- [ ] 9.24 A paired warm benchmark shows compiled single-engine execution is no
  slower than the frozen host-loop reference.
- [ ] 9.25 The single-engine comparison uses equal initial states.
- [ ] 9.26 The single-engine comparison uses equal run controls.
- [ ] 9.27 The benchmark reports controller compilation time separately.
- [ ] 9.28 The benchmark reports first-call device time separately.
- [ ] 9.29 The benchmark reports steady-state device time separately.
- [ ] 9.30 The benchmark reports logical iterations separately from wall time.
- [ ] 9.31 The benchmark verifies ordinary device submissions do not grow with
  the sweep budget.
- [ ] 9.32 A 32-member deterministic mixed-GFE benchmark reports sequential and
  batched warm wall times.
- [ ] 9.33 The mixed-GFE benchmark reports per-member termination reasons.
- [ ] 9.34 The mixed-GFE benchmark reports per-member outer-map evaluation
  counts.
- [ ] 9.35 The handoff makes no GPU throughput claim without a measured GPU
  result.

### 10. Change-path review, tests, and documentation cover every affected surface

- [ ] 10.1 The implementation handoff walks the `Runtime numerical or status
  invariant` row in `docs/ARCHITECTURE.md`.
- [ ] 10.2 The handoff records `Engine.run` as changed with focused evidence.
- [ ] 10.3 The handoff records flooding execution as changed with focused
  evidence.
- [ ] 10.4 The handoff records colored execution as changed with focused
  evidence.
- [ ] 10.5 The handoff records non-GFE mixed execution as changed with focused
  evidence.
- [ ] 10.6 The handoff records mixed-GFE execution as changed with focused
  evidence.
- [ ] 10.7 The handoff records exact two-pass execution as unchanged or changed
  with focused evidence.
- [ ] 10.8 The handoff compares host `_has_zero_support` with JIT
  `_make_support_check` behavior.
- [ ] 10.9 The handoff records `_make_step` as unchanged or changed with focused
  evidence.
- [ ] 10.10 The handoff records `_make_exact_unit` as unchanged or changed with
  focused evidence.
- [ ] 10.11 The handoff records `_make_propagate` as unchanged or changed with
  focused evidence.
- [ ] 10.12 The handoff records `_make_bfe` as unchanged or changed with focused
  evidence.
- [ ] 10.13 The handoff records inner-solver status transport as changed or
  unchanged with focused evidence.
- [ ] 10.14 The handoff records `RollingAgent` as unchanged or changed with
  focused evidence.
- [ ] 10.15 The handoff records `Colony` as changed with focused evidence.
- [ ] 10.16 The handoff records the resource estimator as structurally
  irrelevant unless controller buffers become persistent state.
- [ ] 10.17 The handoff records benchmark consumers as changed with focused
  evidence.
- [ ] 10.18 The handoff records dashboard consumers as unchanged or changed with
  focused evidence.
- [ ] 10.19 Host and JIT stopping checks are compared with focused evidence.
- [ ] 10.20 Flooding and scheduled execution are compared with focused evidence.
- [ ] 10.21 Single-instance and `vmap` execution are compared with focused
  evidence.
- [ ] 10.22 Current and legacy checkpoint decoding are marked structurally
  irrelevant unless implementation stores new persistent state.
- [ ] 10.23 The `New IR field` row is marked structurally irrelevant.
- [ ] 10.24 The `New factor or rule family` row is marked structurally
  irrelevant.
- [ ] 10.25 The `Learnable or checkpoint feature` row is marked structurally
  irrelevant unless implementation stores new persistent state.
- [ ] 10.26 `tests/test_runtime_status.py` covers compiled Engine termination and
  trace semantics.
- [ ] 10.27 `tests/test_schedule.py` covers flooding, colored, mixed, exact, and
  hardening routes through the controller boundary.
- [ ] 10.28 `tests/test_colony.py` covers per-instance stopping, isolation,
  grouping, and throughput behavior.
- [ ] 10.29 `tests/test_outer_gfe_formalism.py` covers compiled fixed-support
  trust-controller certificates.
- [ ] 10.30 Focused factor-family tests cover categorical execution.
- [ ] 10.31 Focused factor-family tests cover Gaussian execution.
- [ ] 10.32 Focused factor-family tests cover exponential-family execution.
- [ ] 10.33 Focused factor-family tests cover recurrence-link execution.
- [ ] 10.34 Focused hostile-input tests cover exact-zero support.
- [ ] 10.35 Focused hostile-input tests cover vacuous messages.
- [ ] 10.36 Focused hostile-input tests cover float32.
- [ ] 10.37 Focused hostile-input tests cover 320-instance execution.
- [ ] 10.38 Performance assertions that depend on wall time are marked slow.
- [ ] 10.39 `docs/ARCHITECTURE.md` describes the compiled ordinary and mixed-GFE
  controller boundaries after implementation.
- [ ] 10.40 `docs/DERIVATIONS.md` replaces its future cff-wbhu controller note
  with the delivered fixed-support carry boundary.
- [ ] 10.41 `README.md` describes compiled per-instance Colony stopping.
- [ ] 10.42 `docs/LIMITATIONS.md` removes the ordinary 1.5× bottleneck only after
  the 8× gate passes.
- [ ] 10.43 `docs/LIMITATIONS.md` removes the mixed-GFE per-instance host-loop
  limitation only after mixed controller batching lands.
- [ ] 10.44 The final code change is followed by one passing default short-suite
  run through the repository dashboard workflow.

## Methodological Suggestions

### Compile control flow, not another inference implementation

Build one pure JAX loop body around the engine's selected iteration function.
Carry state arrays, termination code, active mask, previous free energy,
residual, logical counters, and fixed-size trace buffers. `lax.while_loop` is a
natural fit for true early stopping; a masked `lax.scan` is also acceptable if
it returns exactly the same logical results and does not waste enough work to
miss the performance gates. Convert the returned buffers into the public tuple
form once on the host.

Construct the compiled controller from the same schedule-specific functions
already used by `Engine.run`. Keep validation of Python arguments and grouping
by structure hash on the host. Keep exact two-pass execution on `_run_exact`,
where one directed execution already supplies the desired dispatch behavior.
Do not copy factor equations, the FE ledger, or support rules into the
controller.

### Treat one Colony group as a masked batch of independent runs

Stack compatible states and vectorize the pure iteration body. Carry a Boolean
active lane for each instance. When a lane converges, exhausts, diverges, or
becomes infeasible, freeze its state and trace indices while other lanes
continue. Return fixed-shape device buffers plus counts, then slice each lane's
public traces on the host. Preserve host-side grouping, port binding, instance
IDs, kill/spawn behavior, and final marginal reads.

Use a test-only copy of the pre-change host controller as the semantic and
performance reference. Keeping it outside the public package avoids two
production authorities. Compare complete statuses, counts, traces, states,
objectives, and BFE rather than checking only selected marginals.

### Reuse the proof-backed mixed-GFE controller state

`solve_outer`, `_make_trust_kernel`, and `_make_root_diagnostics` already define
the accepted numerical method. Move the fixed-support controller state into a
compiled carry and vectorize the existing trust kernel. Keep support changes,
sequential delta-hardening phases, and chart reconstruction as explicit restart
boundaries if their shapes change. A host restart is acceptable; host control
once per trust iteration is not.

Within one structure group, partition mixed-GFE instances by the complete
static outer-chart cache key before vectorizing them. Instances with different
support faces may take separate compiled subgroups; do not pad absent support
coordinates back into the chart merely to keep one batch. Regroup only at an
explicit support or hardening boundary, and preserve instance IDs across that
regrouping.

Maintain lane-local radii, acceptance decisions, diagnostics, and termination
reasons. A fast candidate remains only a candidate until the full undamped
root, support, inner-status, feasibility, and stationary-FE checks pass. Do not
turn the legacy Picard path into an automatic fallback.

### Benchmark asynchronous JAX honestly

Commit the benchmark before replacing the loop and verify that it reproduces
the recorded dispatch-bound range. Warm each compiled path, block the final
device value before stopping each timer, and time inference separately from
compile, spawn, bind, and readout. Compare one 32-member Colony group with 32
sequential calls over equal states and controls. Report raw timings alongside
the ratio so an 8× claim cannot hide a single-engine regression.

### Evidence anchors and dependency state

- `src/cffg/engine.py::Engine.run` contains the Python `phase` sweep loop and
  defines ordinary trace and status semantics.
- `src/cffg/engine.py::_run_outer_trust_region` contains the host fixed-support
  mixed-GFE phase and restart loops.
- `src/cffg/colony.py::Colony.run` contains the per-sweep vectorized dispatch,
  host `active` array, host trace lists, and per-instance result reconstruction.
- `src/cffg/outer.py::solve_outer` supplies the proof-backed mixed-GFE
  controller that must be compiled rather than replaced.
- `tests/test_runtime_status.py` pins the four statuses, trace lengths,
  residuals, sparse support, and float32 tolerance behavior.
- `tests/test_schedule.py` pins exact, flooding, colored, mixed, hardening, and
  fixed-point behavior.
- `tests/test_colony.py` pins structure grouping, sequential agreement,
  kill/spawn behavior, and host port routing.
- `tests/test_outer_gfe_formalism.py` pins single/`vmap` trust-kernel agreement
  and the complete mixed-GFE root certificate.
- `notes/adversarial-review-2026-07-10/02-performance.txt` records the
  32-instance 1.5× CPU result and identifies Python dispatch as one bottleneck.
- `docs/ARCHITECTURE.md` assigns both the general convergence carry and the
  mixed-GFE active-set/status carry to cff-wbhu.
- `docs/LIMITATIONS.md` records both ordinary and mixed-GFE Colony host-control
  limitations.
- cff-j4mu supplies truthful runtime statuses.
- cff-d56d supplies schedule-aware execution.
- cff-44we supplies shape-segmented batch arenas.
- cff-weff and cff-lv2w supply the proof-backed and optimized mixed-GFE
  numerical controller.
- cff-wbhu owns compiled deterministic Engine and Colony loop control,
  per-instance stopping, and the 8× ordinary throughput gate.
- cff-rqqc owns stochastic `mu3` PRNG and particle-state batching.
- cff-yl0a owns discrete HMM `ScanStage` body fusion.
- cff-g7rc owns remaining fixed certification and cold-compile costs for tiny
  mixed-GFE solves.

Legend: `-->` means prerequisite to dependent; `~~~>` means linked interface
coordination without a dependency edge.

```text
[x] cff-j4mu --+
[x] cff-d56d --+--> [ ] cff-wbhu (ready)
[x] cff-44we --+             |
                             +--> [ ] cff-5edo
                             |
                             +--> [ ] cff-9xmq
                             |           |
                             |           +--> [ ] cff-mjku
                             |                       |
                             |                       +--> [ ] cff-pyy9
                             |
                             +--> [ ] cff-ahps (roadmap epic)

[x] cff-weff --+
[x] cff-lv2w --+~~~> [ ] cff-wbhu (mixed-GFE controller foundation)

[ ] cff-wbhu ~~~> [ ] cff-rqqc (stochastic Colony state)
[ ] cff-wbhu ~~~> [ ] cff-yl0a (exact HMM scan-body fusion)
[ ] cff-wbhu ~~~> [ ] cff-g7rc (tiny mixed-GFE fixed overhead)
```

## Future work

cff-5edo owns depth-aware long-rollout budgets and warm-started replanning after
compiled loop control lands. cff-9xmq owns balanced Broadcast topology and its
49-branch propagation and throughput evidence. cff-rqqc owns per-instance PRNG
keys, particles, and stochastic `mu3` outcomes in Colony; this ticket retains
Colony's rejection of those evaluators. cff-yl0a owns fusion of the operations
inside an exact discrete HMM scan step. cff-g7rc owns fixed certification and
cold-compile overhead for tiny mixed-GFE solves. This ticket does not claim GPU
scalability, cross-structure batching, faster public readouts, a new inference
schedule, or new factor mathematics.

## Definitions

- **Engine**: the compiled runtime for one grounded CFFG structure and one
  `State`.
- **Colony**: the runtime that groups independent model instances by structure
  hash and executes compatible groups together.
- **State**: the runtime pytree containing messages, parameters, observations,
  masks, and other inference arrays for one instance.
- **Structure hash**: the deterministic identifier used to group instances with
  identical serialized model structure.
- **Sweep**: one iterative application of the selected message-update schedule.
- **Dispatch**: one host request that launches compiled work on a JAX device.
- **Host synchronization**: a device-to-host read that waits for pending device
  work before Python can inspect a value.
- **Compiled controller**: JAX control flow that carries inference state,
  stopping decisions, and diagnostics across iterations on the device.
- **Loop carry**: the values passed from one compiled loop iteration to the
  next.
- **Active lane**: one batched instance that has not yet reached a termination
  outcome.
- **Controller subgroup**: same-structure instances that also share one static
  outer-chart key and can therefore execute one compiled mixed-GFE loop.
- **Masking**: retaining a stopped lane's values while active lanes continue to
  update.
- **Logical iteration**: an update counted in a public result, excluding masked
  device work after that instance stopped.
- **Flooding schedule**: synchronous update of every eligible factor in one
  sweep.
- **Colored schedule**: an iterative schedule that updates independent loopy-BP
  factor classes in deterministic color order.
- **Mixed schedule**: an iterative schedule that combines non-BP structures
  with exact or colored BP subgraphs.
- **Two-pass schedule**: one exact inward and outward traversal of an acyclic BP
  graph followed by fixed-point certification.
- **Priming**: one exact equilibration of forest islands before iterative mixed
  or colored execution.
- **Hardening phase**: a phase in which delta constraints commit selected states
  to point masses.
- **Run status**: one of `CONVERGED`, `BUDGET_EXHAUSTED`, `DIVERGED`, or
  `INFEASIBLE`.
- **Residual**: the numeric stopping measurement associated with the returned
  status.
- **Message delta**: the largest absolute change among the runtime message
  arenas in one logical iteration.
- **BFE**: Bethe free energy computed from node-local terms and edge-entropy
  corrections.
- **Defined ledger**: an inference state for which the active free-energy
  expression has a finite evaluable value.
- **Exact support**: the categorical states whose log probabilities are not the
  literal negative-infinity sentinel.
- **Vacuous message**: a message that contributes no relative categorical
  preference or Gaussian information.
- **Mixed GFE**: a graph whose generalized-free-energy structures participate
  in a coupled outer fixed-point solve.
- **Fixed-support solve**: one outer solve whose categorical support face and
  array shapes remain constant.
- **Support face**: the subset of categorical states retained as coordinates in
  one outer chart.
- **Outer chart**: the packed, gauge-reduced coordinate representation of one
  fixed-support mixed-GFE problem.
- **Trust radius**: the current bound on a mixed-GFE trial step.
- **Matrix-free trust kernel**: the compiled JVP/VJP-based inner calculation
  used by the proof-backed trust-region controller.
- **Termination reason**: the machine-readable mixed-GFE explanation for why an
  outer solve stopped.
- **Free-energy certificate**: the finite stationary BFE/GFE check required for
  a successful mixed-GFE result.
- **PRNG**: pseudo-random number generator; stochastic evaluators use explicit
  per-instance keys rather than hidden global randomness.
- **`mu3`**: the stochastic functional message used for an uncertain likelihood
  matrix in the LAIF GFE composite.
- **Warm throughput**: completed inference work per unit time after required
  compilation and warmup.
- **Sequential denominator**: the time for 32 direct single-engine runs used in
  the Colony speedup ratio.
- **Test-only host-loop reference**: a frozen pre-change controller retained in
  test or benchmark code solely to compare results and timing.
- **Hostile-scale fixture**: the 320-instance, 600-sweep case that exercises ten
  times the recorded Colony width and sweep budget.

## Notes

### 2026-08-07 — Historical evidence and rewrite audit

The bounded historical slice returned five candidates across five occurrences
and classified zero occurrences as problems. Historical meanings were
`preserve` as a semantic invariant, `same` as equivalent benchmark controls,
`parity` as numeric agreement within a tolerance, `test` as a test artifact,
and `only` as exclusion of alternatives outside the stated condition.

The final scan returned 65 candidates across 545 occurrences. Manual contextual
review found zero ambiguous occurrences. Current-scan meanings are recorded
below; each retained candidate has one intended meaning or an explicitly
enumerated set of intended meanings in this ticket.

- `canonical`: the repository's designated runtime, schedule, or documentation
  authority is meant.
- `fixed`: supplied parameters, a constant support face, or a frozen benchmark
  artifact is meant according to context.
- `preserve`: the named API, numeric behavior, state mapping, or historical
  evidence retains its stated property.
- `public`: the documented `Engine`, `Colony`, `RunResult`, or readout contract
  is meant.
- `reusable`: a compiled controller or cached kernel is used for repeated calls
  with its declared execution key.
- `same`: value equality, one structure group, or equivalent benchmark controls
  are meant according to context.
- `without_recompile`: the named repeated execution reuses a compiled artifact
  instead of creating one compilation per member or sweep.
- `available`: a selected schedule or compiled kernel is ready for the named
  execution path.
- `expose`: a public result makes the named status or diagnostic reachable.
- `implement`: planned code work or the repository component supplying a
  behavior is meant according to context.
- `support`: categorical probability support or a runtime path's declared
  capability is meant according to context.
- `work`: runtime computation, completed prerequisite implementation, or
  planned ticket activity is meant according to context.
- `complete`: the entire named certificate, result, configuration, or ticket
  outcome is meant.
- `deliver`: this ticket supplies the named compiled-loop behavior and evidence.
- `land`: a named repository change is merged.
- `ready`: cff-wbhu has no unfinished declared prerequisite.
- `remaining`: a runtime cost or downstream task continues after this ticket.
- `approximate`: an alternative numeric method is explicitly excluded.
- `close`: the named mixed-GFE prerequisite tickets reached administrative
  completion.
- `sufficient`: the returned logical result satisfies every declared semantic
  condition despite possible masked device work.
- `certify`: the current runtime or mixed-GFE checks establish the named
  termination claim.
- `check`: an executable assertion, validation step, or change-path review is
  meant.
- `converge`: the applicable `RunStatus.CONVERGED` criterion is satisfied.
- `correct`: the vectorized message body produces the declared mathematical
  result.
- `equivalent`: benchmark controls or observable runtime outcomes are equal in
  the respects named by the sentence.
- `estimate`: the resource estimator computes a declared byte or operation
  quantity.
- `evidence`: a named test, benchmark, code location, document, or dependency
  fact supports the surrounding claim.
- `exact`: a literal support state, a two-pass schedule, or a precise count or
  identity is meant according to context.
- `fixed_point`: a state satisfying the current runtime's falsifiable
  stationarity certificate is meant.
- `match`: numeric equality within the criterion's declared tolerance is meant.
- `oracle`: the independently frozen host loop used as a comparison reference
  is meant.
- `parity`: equal status or numeric results under the named tolerance is meant.
- `pass`: one traversal direction or successful automated execution is meant
  according to context.
- `replicate`: repeated plate or instance structure is meant.
- `reproduce`: an executable fixture attains the stated historical measurement
  or reference output.
- `stable`: the public execution interface and field meanings remain compatible
  through the controller change.
- `test`: a named test artifact or automated executable assertion is meant.
- `validate`: argument, support, structure, or audit validation is meant
  according to context.
- `verify`: an executable benchmark check establishes the named dispatch or
  historical-baseline property.
- `fit`: the suggested JAX loop primitive is suitable for the stated early-stop
  behavior.
- `infer`: fixed-parameter state or policy beliefs are computed.
- `learn`: parameter estimation is outside fixed-parameter parity fixtures.
- `learnable`: the named architecture change-path category is meant.
- `trial`: one mixed-GFE candidate-step evaluation is meant.
- `performance`: measured dispatch count, wall time, compilation, or throughput
  is meant.
- `scale`: instance width, sweep budget, domain dimension, or dtype-relative
  tolerance is meant according to context.
- `small`: one bounded graph or lightweight runtime case is meant.
- `speed`: the measured Colony-to-sequential throughput ratio is meant.
- `can`: the implementation or evidence permits the stated behavior.
- `expected`: a result follows from the selected schedule or declared runtime
  condition.
- `intended`: the documented ticket scope or contextual audit meaning is meant.
- `may`: an alternative implementation is permitted if it meets the criteria.
- `should`: an optional methodological recommendation is meant.
- `will`: the ticket commits the named controller behavior as future work.
- `explicit_owner`: `owns` assigns work to cff-wbhu or the named linked and
  downstream tickets.
- `named_ticket_handoff`: a named ticket supplies the output stated beside it.
- `all`: every member of the locally named configuration or diagnostic set is
  meant.
- `any`: every input or nonconverged status inside the named boundary is meant.
- `at_least`: the stated 8× throughput value is an inclusive lower bound.
- `every`: the statement applies separately to each member of the named result,
  instance, field, status, solver, or guard collection.
- `only`: a hard semantic exclusion, temporal gate, or ticket-scope boundary is
  meant according to context.
- `current`: the repository implementation or contract inspected on 2026-08-07
  is meant.
- `existing`: the runtime, schedule, status, test, or numeric contract already
  present in the repository is meant.
- `later`: work assigned to a named downstream ticket follows cff-wbhu.
- `read`: a device-to-host value transfer or final marginal lookup is meant
  according to context.

Intelligible-ticket review: complete.
