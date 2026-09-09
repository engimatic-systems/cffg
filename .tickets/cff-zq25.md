---
id: cff-zq25
status: closed
deps: [cff-wera]
links: []
created: 2026-09-09T03:25:11Z
type: chore
priority: 1
assignee: coopwilliams
parent: cff-209l
tags: [architecture, kernels, gaussian]
---
# Register and migrate ordinary Gaussian factor execution

## Background in Plain English — WHY

After cff-wera delivers the discrete registration path, ordinary Gaussian
factors still need central compiler and runtime branches. This ticket delivers
their binding, including observed linear factors, and finishes the cff-209l
foundation. It uses the information-form Gaussian representation already
shipped by the engine.

## Outcomes / Acceptance Criteria — WHAT

1. cff-wera is closed. The pre-migration Gaussian/Kalman outputs and arrays are
   recorded at its completion head and compared with e488df3 on the public,
   serialized, and numerical surfaces cff-209l promises to retain.
2. The Gaussian binding satisfies cff-209l's Gaussian and shared declarations:
   role dimensions, information-form inputs, parameters, message and node-energy
   callbacks, conditioning/validation, readouts, hostile inputs, and identity.
3. The ordinary Gaussian built-ins listed in cff-209l use the binding through
   grouping/packing, flooding, batch and scan schedules, BFE, observations, and
   readouts. Observed gaussian_linear remains one serialized factor identity.
4. An external Gaussian factor proves messages, BFE, readouts, and conditioned
   messages/energy against an analytic Gaussian oracle; invalid observations
   fail through its validator. It needs no central type-name branch.
5. JSON and CFFGZ reconstruct its identity and version and reject missing or
   incompatible implementations. Existing checkpoints retain their decoding
   behavior and reconstructed public results.
6. Exact vacuous messages, improper required beliefs, non-finite outputs,
   float32, and dimension at least 100 satisfy cff-209l's numerical contract.
   Sparse categorical support is N/A to this binding; inherited discrete tests
   retain their log-zero behavior. Persistent Gaussian storage remains flat
   and its byte counts equal the actual State leaves and estimator totals.
7. Host/JIT, flooding/scheduled, and a batch of at least two instances with
   different observations or parameters agree where each path is available.
8. One existing Gaussian chain retains its schedule, shapes/dtypes, persistent
   bytes, and registerized scan carry. Changed observation/parameter values at
   unchanged shapes/dtypes/static controls reuse one Engine's traced numerical
   executable. No latency improvement is claimed.
9. The discrete and EF extension proofs still pass. The final integration
   satisfies cff-209l's shared acceptance criteria without another dispatch
   implementation, storage redesign, estimator cost model, or new inference rule.
10. README, ARCHITECTURE's change-path index, and LIMITATIONS describe the
    three delivered arena bindings and the specialized-plan exclusions.
11. Focused Gaussian/kernel/schedule/observation/serialization/status/estimator
    tests and one dashboard-backed short suite pass after the final code change.

## Methodological Suggestions — HOW

These suggestions are optional; another sound method may satisfy the outcomes.

- Adapt gauss.py rules to the binding without changing their formulas.
- Use **New factor or rule family** and **Runtime numerical or status
  invariant**; audit manifest and parameter consumers through **New IR field**
  and **Learnable or checkpoint feature**. Checkpoint schema changes are N/A.
- Keep the observation and parameter values as runtime arrays, separate from
  the static role dimensions and kernel identity captured by JAX.

## Future work

cff-209l closes the foundation and cff-ecw0 records the compiler baseline.
cff-o7hs adds nonlinear Gaussian methods after the architecture epic. Neither
new nonlinear mathematics nor stochastic Colony execution is delivered here.

## Definitions

- **Information form:** A Gaussian message stored as natural vector eta and
  precision matrix Lambda.
- **Vacuous:** The neutral Gaussian message with eta and Lambda both zero.
- **Improper belief:** A message product that cannot be normalized as required.
- **BFE:** Bethe free energy with its node and edge-entropy contributions.
- **EF:** Exponential-family message representation.
- **JIT:** JAX just-in-time compilation; **State** holds runtime arrays.
- **Registerized scan:** A chain loop carrying local message blocks before one
  final write to the arena.
- **CFFGZ:** The template-and-array bundle format.
- **Colony:** The runtime that groups and runs multiple model instances.

## Notes

**2026-09-09T04:30:29Z**

Implementation handoff: base1877533, discrete PR110, cff-wera closed with1546-test gate. Base Gaussian snapshots are bit-identical to captured30a7425/e488df3 source anchor (prior/observed/missing/32-step Kalman, named parameter addresses, public statistics, slots and scan carries). Change-path rows: New factor or rule family; Runtime numerical or status invariant; Learnable or checkpoint feature for registered statistics/parameter writes; New IR field only manifest verification, no schema change. Pairs: host/JIT; flooding/batch/registerized/flat scans; direct/distinct-input vmap; observed/unobserved/missing; generic readout/sufficient-statistics APIs. Latest/legacy checkpoints structurally unchanged and existing tests apply. Research question: which observed Gaussian semantics must remain? Answer: retain every role and write the forward predictive to the observed role; missing observed-linear node energy is negative input-belief entropy, not zero, cancelling the interior-edge correction. Source: gauss.linear_observed_energy, engine._gauss_outs, test_kalman; formalism_review confidence high. The older gauss.py module comment saying zero energy will be corrected to match executable behavior. Learning statistics preserve host operation order and keys, including joint cross-covariances, proper flag and mask-zero skipped occurrences; no readout vectorization. External fixture: proper joint Gaussian over custom roles field/sensor/offset, unequal dims3/2/1, conditioning on middle sensor role; separately packed canonical parameters and independent covariance-conditioning/Schur/KL oracle. Missing sensor marginalizes its block rather than dropping the joint factor. Hostile plan: vacuous inputs are neutral; required invalid precision/nonfinite outputs are loud; active-dtype guards; dimension100 beside small factors with flat byte equality. Stop after this ticket and cff-209l; no compiler/engine extraction.

**2026-09-09T04:48:42Z**

Implementation complete; final integration gate pending. GaussianBinding and GaussianConditioning now drive built-in prior/linear grouping, parameter packing, all message schedules, DATA validation, node free energy, public readouts and learning statistics. The three-role external joint Gaussian proves custom packing and middle-role conditioning against covariance-conditioning and KL oracles, including missing data, float32, dimension 100, flat storage, both scan implementations, trace reuse, and two distinct Colony instances. JSON/CFFGZ and array/file/initializer reconstruction pass. README, ARCHITECTURE change-path index and LIMITATIONS describe the delivered scope; no compiler/engine extraction or new inference mathematics.

Review question: can two registrations silently share storage? Answer: derived Gaussian group-name collisions and multiple bare parameter keys now fail before writes; built-in conditioned aliases are reserved. Source: kernels.register_kernel, compiler Gaussian grouping, tests/test_gaussian_kernels.py; modularity_review reproduced distinct means being overwritten before the guard. Confidence high. Question: can host dtype conversion reject valid input? Answer: covariance symmetry is checked with the source dtype, with Cholesky/inversion performed in float64; array/file regressions include float16, float32 one-ULP asymmetry and float64. Source: compiler._resolve_arrays, params._validate_covariance and regression tests; confidence high. This aligns initial registered covariance validation with named writes, rejecting truly asymmetric/nonfinite/indefinite inputs rather than using the older absolute symmetry tolerance. No rule formula changed.

Validation so far: ten frozen HMM/Kalman/T-maze cases retain all compared preexisting arrays, addresses, hashes, public outputs, objective values, schedule and scan-carry structure; all 64 EF callback arrays are bit-identical. Pristine baseline is 30a7425, whose src/tests/dependency contents equal e488df3. Readout/statistics operation order stays unchanged. Shared fixtures and scripts remain in /tmp/cffg-registration-baseline; the final evidence archive will also be placed beside the shared HTML. Wiggle audit was not run. Stop after this ticket and cff-209l.

**2026-09-09T05:05:28Z**

Final delivery: code commit a635a73 passes the dashboard-backed not-slow suite: 1,572 passed, 41 slow tests deselected, 0 failures, four existing Gym deprecation warnings, 971.29 seconds pytest time. Runtime history and the shared dashboard were refreshed. Focused integration: 99 passed; follow-up parameter/learning tests: 30 passed; final declaration/dtype guard cases: 9 passed. Ten frozen model comparisons and all 64 EF callback arrays pass at this exact code head. Ruff checks for new binding/runtime/fixture modules, git diff --check and tk dep cycle pass. No dependencies or checkpoint schemas changed.

The delivery branch is agent/cff-zq25-gaussian-kernels, stacked on discrete PR110. The shared HTML contains delivery status and a downloadable, reproduction-checked evidence archive. Completed: Gaussian execution registration, observed/missing roles, host statistics, custom extension proof, documentation and final gate. Next: close cff-209l and stop. Compiler baseline/extraction, learning-family expansion and performance work remain unstarted. Gotcha: nonlinear/joint raw-to-runtime packs without direct-write keys reconstruct through compilation; registration does not imply a conjugate learning rule. No wiggle audit was run.
