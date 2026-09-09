---
id: cff-wera
status: open
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
