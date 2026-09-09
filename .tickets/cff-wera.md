---
id: cff-wera
status: closed
deps: [cff-ri8m, cff-l6i5]
links: []
created: 2026-09-09T03:25:11Z
type: chore
priority: 1
assignee: coopwilliams
parent: cff-209l
tags: [architecture, kernels, discrete]
---
# Register and migrate ordinary discrete factor execution

## Background in Plain English — WHY

Ordinary discrete factors still require compiler and engine type-name branches.
This ticket delivers the discrete half of cff-209l through registration,
execution, scoring, and reconstruction. It defines the discrete binding while
using it for shipped factors, so reviewers can evaluate the interface against
working message rules. The Gaussian half follows in cff-zq25 because both
migrations edit the compiler and engine.

## Outcomes / Acceptance Criteria — WHAT

1. cff-ri8m and cff-l6i5 are closed. The pre-change commit, public call forms,
   layout, messages, BFE, and decoded assignment scores are recorded using
   existing HMM and T-maze fixtures; e488df3 remains the external parity anchor.
2. The discrete binding satisfies cff-209l's discrete and shared declarations:
   roles/cardinalities, parameters, BP/VMP inputs, semiring compatibility,
   energy, assignment scoring, observations/readouts when declared, hostile
   inputs, and serialized implementation identity.
3. The ordinary discrete built-ins listed in cff-209l use that binding through
   compiler grouping/packing, flooding, batch and scan schedules, BFE,
   assignment scoring, and public readouts. Specialized composites and
   structural equality keep their declared specialized paths.
4. One external discrete factor proves sum-product, max-product, and VMP
   against finite enumeration. Its custom named parameters exercise assignment
   scoring without assuming a dense potential at the group-name key. Two
   uncertain neighbors demonstrate the declared difference between BP cavities
   and VMP beliefs.
5. JSON and CFFGZ reconstruct the external implementation identity/version;
   missing or incompatible implementations fail as cff-209l requires. Neither
   the custom type name nor its parameter keys require central dispatch edits.
6. Exact log-zero support, infeasibility, non-finite outputs, float32, and
   cardinality at least 128 satisfy cff-209l's numerical contract. Gaussian
   vacuity is N/A to this discrete binding. Persistent bytes retain flat storage
   and equal the State array-leaf totals and existing estimator totals.
7. Host/JIT, flooding/scheduled, and a batch of at least two distinct-input
   instances agree where the mode permits each path. Checkpoint decoding is
   unchanged; its existing reconstruction tests remain applicable.
8. One existing discrete chain retains its schedule, array shapes/dtypes,
   persistent bytes, and registerized scan carry. Repeated calls to one Engine
   with unchanged shapes/dtypes/static controls and changed parameter values
   reuse the traced numerical executable. This proves structure preservation,
   not a speedup.
9. EF registration and the unmigrated Gaussian paths retain their recorded
   behavior. No storage redesign, estimator cost model, new inference method,
   or serialized schema enters this ticket.
10. README and the architecture change-path index describe the delivered
    discrete binding; LIMITATIONS still identifies the pending Gaussian work.
11. Focused kernel/discrete/VMP/schedule/serialization/status/estimator tests
    and one dashboard-backed short suite pass after the final code change.

## Methodological Suggestions — HOW

These suggestions are optional; another sound method may satisfy the outcomes.

- Use arena-specific typed bindings and adapt existing rules before moving them.
- Keep numerical arguments dynamic and kernel/mode/shape selection static in JAX.
- Audit **New factor or rule family** and **Runtime numerical or status
  invariant**; audit the manifest in **New IR field** and named parameter
  consumers in **Learnable or checkpoint feature**. No checkpoint schema changes.
- Keep each commit independently testable; cff-209l remains the shared contract.

## Future work

cff-zq25 delivers the Gaussian binding and integration proof. cff-209l closes
after both migrations. cff-j7mm owns XLA cost/temporary-memory analysis;
cff-wbhu owns compiled convergence loops and throughput.

## Definitions

- **BP:** Belief propagation using incoming messages that exclude the receiving factor.
- **VMP:** Variational message passing using the declared factorized beliefs.
- **BFE:** Bethe free energy, including node terms and edge-entropy corrections.
- **Semiring:** The sum-product or max-product reduction used by a message rule.
- **Cavity:** Incoming information with the receiving factor's contribution removed.
- **EF:** Exponential-family message representation already registered in CFFG.
- **JIT:** JAX just-in-time compilation; **State** is the runtime array container.
- **Registerized scan:** A chain loop carrying local message blocks and writing
  its outputs to the flat arena after the loop.
- **CFFGZ:** The bundle containing a serialized template and referenced arrays.
- **XLA:** The compiler JAX uses to create numerical executables.

## Notes

**2026-09-09T04:02:25Z**

Implementation handoff: base 30a7425 (review PR109). Change-path rows: New factor or rule family; Runtime numerical or status invariant; Learnable or checkpoint feature for parameter addresses/readouts; New IR field only manifest verification, no schema change. Affected pairs: host/JIT support and execution, flooding/batch/registerized scan/flat scan, direct/vmap with distinct inputs, BP/VMP belief construction, message/node-energy/readout/assignment parameter views. Checkpoint latest/legacy decoding: no schema or state-leaf changes, existing regression tests apply. Hostile inputs: preserve exact NEG and report empty support infeasible; nonfinite callback outputs diverge; float32 rules use active-dtype algebra; custom cardinality128 coexists with small roles in flat storage. Equality, Dirichlet parameter edges and GFE composites remain specialized; Gaussian execution unchanged until cff-zq25. Research: runtime audit identified dense-potential assumptions in both support checks, node_beliefs and objective; declared potential access will own these. Source: engine.py consumers independently audited by performance_review; confidence high. Modular binding and serialized shape metadata must also cover parameter reconstruction and writes; source compiler.py and params.py audit by modularity_review; confidence high.

**2026-09-09T04:14:19Z**

Implementation ready for full gate: typed EF/discrete bindings, unchanged legacy EF constructor, compiled named parameter paths, serialized shape providers and direct-write metadata, BP/VMP-specific inputs, potential access for support/joint beliefs/assignment scores, all flooding/batch/registerized/flat-scan local calls, and abstract message/energy/potential/readout shape checks during compilation. External bilinear fixture supplies enumeration oracles, two uncertain neighbors, total VMP ledger, sparse-support/float32/cardinality128, distinct-input vmap, JSON/CFFGZ/file/init/override reconstruction, named writes, nonfinite statuses, chain transport and trace-reuse proofs. Focused selections passed: 80 tests (kernels/VMP/schedule/params/status/estimate) and latest 43 tests (external/EF/params). Independent review fixed deepcopy recursion and malformed pack/axis metadata; no dispatch bypass found. Strict pre-change comparison at 30a7425 passes all 10 HMM/Kalman/T-maze fixtures, including existing member-address fields, messages/readouts/energies, hashes, params, schedules and scan carries. Newly introduced address declarations are explicitly excluded from old-field comparison only. HMM BFE22.09818901990299, bytes9544; Kalman BFE5.412683875149131, bytes2016; T-maze GFE3.5227151149325486, bytes8064. HMM scan carries one (1,3) block; Gaussian baseline remains unchanged. Reproducer/artifacts: /tmp/cffg-registration-baseline/compare_registration.py and baseline/. Await dashboard-backed short suite, then close and stack Gaussian cff-zq25. No wiggle audit was run.

**2026-09-09T04:30:07Z**

Completed at code commit a9bbf79. Required dashboard short suite passed: 1546 tests, 41 slow tests deselected, 0 failures, 900.41 seconds; four existing Gym deprecation warnings. Dashboard refreshed. Extra direct/JIT float32/float64 EF callback comparison: all64 arrays bit-identical for gamma/beta/poisson_obs/bernoulli_obs including missing masks; reproducer /tmp/cffg-registration-baseline/compare_ef_callbacks.py. External discrete Colony check: two different parameter instances share one cached Engine, both converge and match enumeration. git diff from e488df3 to base30a7425 is empty across src/tests/dependencies, so old-code parity also covers that external anchor. Independent registration review findings (deepcopy recursion and malformed pack/axis/permutation declarations) are fixed and tested. git diff --check and tk dep cycle pass. Next: cff-zq25 Gaussian migration; stop after it and cff-209l. No short-suite rerun is needed for PR metadata or dashboard refresh alone.
