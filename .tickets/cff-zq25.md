---
id: cff-zq25
status: open
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
