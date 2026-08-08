---
id: cff-ahps
status: open
deps: [cff-1o72, cff-3hgi, cff-44we, cff-5edo, cff-6s31, cff-9xmq, cff-bxk4, cff-d56d, cff-dfpi, cff-l6i5, cff-lquu, cff-o7hs, cff-ri8m, cff-wbhu, cff-izjd]
links: []
created: 2026-07-08T16:48:44Z
type: epic
priority: 1
assignee: coopwilliams
---
# Roadmap: universal factor library — continuous, hybrid, constraints, neural

## Background in Plain English

CFFG began with discrete Categorical and Dirichlet factors. Completed tickets have added Gaussian and hybrid models, conjugate exponential-family messages, hard constraints, executable factor registration, and discrete-regime epistemic scoring. This epic finishes the roadmap when the remaining prerequisites add nonlinear and neural factors, scalable plate execution, long-rollout stability, runtime throughput, and continuous-domain diagrams. The result is one serialized modeling system whose compiler, inference runtime, objective ledger, and public tools agree about every factor family they expose.

The direct prerequisites already completed are cff-3hgi, cff-44we, cff-d56d, cff-dfpi, cff-izjd, cff-l6i5, cff-lquu, and cff-ri8m. The unfinished direct prerequisites are cff-1o72, cff-5edo, cff-6s31, cff-9xmq, cff-bxk4, cff-o7hs, and cff-wbhu. The dependency metadata, rather than the order of this prose, determines when this epic is ready to close.

## Outcomes / Acceptance Criteria

1. Every ticket listed in this epic's `deps` metadata is closed because its own acceptance criteria are satisfied, not because its remaining work was moved into this epic.
2. The supported public modeling workflow satisfies the documented combinations of discrete, Gaussian, hybrid, hard-constraint, and neural factors.
   2.1. The public API accepts a serialized template that combines those factor families.
   2.2. The compiler reconstructs the required arenas from that template and its referenced data.
   2.3. The runtime executes each combination permitted by the documented semiring and constraint rules.
3. The objective reported for each permitted execution has reviewable mathematical meaning.
   3.1. In sum-product mode, factors with analytic node-energy and entropy terms contribute the true Bethe or Generalised free-energy terms, including edge-entropy overcounting corrections and P-substitution where applicable. Automated checks compare analytic reference values within a stated tolerance justified by the reference calculation's conditioning and the active dtype.
   3.2. Every factor that uses an approximation names the mathematical method, the regime where it applies, the ledger quantity it reports, and the validation tolerance or diagnostic bound. `docs/LIMITATIONS.md` records the resulting capability boundary.
   3.3. In max-product mode, the runtime reports the decoded objective value and does not label that value as free energy.
4. Each message-rule family introduced by a prerequisite supplies automated evidence for the numerical robustness contract.
   4.1. Literal zero probability remains excluded from support. An empty feasible-support intersection returns an infeasible result instead of adding probability mass through a numeric floor.
   4.2. A vacuous incoming message follows the family's stated neutral-message rule. An improper belief either follows a proved family-specific rule or produces an explicit infeasible or diverged status and a non-finite ledger value; it never produces a finite converged result by accident.
   4.3. Float32 execution uses guards derived from the active dtype and meets a stated tolerance against an independent analytic or numerical reference.
   4.4. A case with a domain, plate, rollout, batch, or factor count 10–100 times the family's basic fixture completes within its documented resource envelope or returns a machine-readable non-converged status.
5. Integration evidence covers compiler dispatch, synchronous and scheduled inference, objective evaluation, public readouts, serialization, and resource estimation for every affected family. The evidence includes host/JIT and single-instance/`vmap` agreement wherever both execution paths apply.
6. `README.md`, `docs/ARCHITECTURE.md`, `docs/DERIVATIONS.md`, and `docs/LIMITATIONS.md` agree with the delivered public API, mathematical contracts, named approximations, and remaining limits.
7. No completed work remains described only in a ticket or a `docs/todo-notes/` file.

## Methodological Suggestions

These are suggested ways to reach the outcomes. An implementer may choose a different method if it satisfies the numbered outcomes and supplies equivalent artifact and test coverage.

- Use the **New factor or rule family** row in `docs/ARCHITECTURE.md` as the primary change path for nonlinear and neural factors. Walk the `FactorKernel`, `FactorType`, rule module, compiler grouping, slot addressing, parameter packing, arenas, learning membership, synchronous step, scheduled stages, objective ledger, schedule classification, readouts, serialization manifest, estimator, and public exports. Mark an item N/A only with a concrete reason.
- Also use the **Runtime numerical or status invariant** row for convergence, divergence, infeasibility, long rollouts, and throughput changes. Use the **New IR field** and **Learnable or checkpoint feature** rows when a prerequisite changes serialized structure or persistent neural parameters.
- Check the deliberate execution pairs: host/JIT, flooding/scheduled, single-instance/`vmap`, and latest-schema/legacy checkpoint decoding. Record which pairs each prerequisite changes and which are N/A.
- Keep messages in log space unless a family derivation requires another representation and documents that exception. Normalize messages aggressively, derive numeric guards from `jnp.finfo(dtype)`, and give stochastic rules explicit JAX PRNG keys.
- Put message and objective derivations in rule docstrings and `docs/DERIVATIONS.md`. Per Cooper's 2026-07-08 delegation decision, keep free-energy ledger semantics, constraint semantics, bead semantics, and the true-BFE invariant under top-level review. Well-specified arena plumbing, builders, focused test scaffolding, reference-number generation, and bounded literature questions may be delegated and then reviewed at top level.
- Run the contributing ticket's focused rule, compiler, schedule, ledger, serialization, hostile-input, and public-workflow tests during implementation. After the final code change, run the repository's dashboard-backed short suite once as required by `AGENTS.md`; run marked slow tests only when the changed behavior needs them.
- Recheck the roadmap with `tk dep tree --full cff-ahps`, `tk ready`, `tk blocked`, and `tk dep cycle`. The dependency records remain authoritative. A compact view of the open paths at the 2026-08-07 rewrite is:

  ```text
  Legend: prerequisite --> dependent

  [ ] cff-209l (ready) --> cff-o7hs --> cff-1o72 --+
                                                       |
  [ ] cff-wbhu (ready) -+--> cff-5edo -----------------+
                        +--> cff-9xmq -----------------+--> cff-ahps
                        +------------------------------+
  [ ] cff-9bex (ready) --> cff-bxk4 -------------------+
  [ ] cff-j7mm (ready) --> cff-6s31 -------------------+
  ```

## Future work

- cff-o7hs owns deterministic nonlinear maps between Gaussian edges, Jacobian linearization, sigma-point moment matching, named approximate ledger terms, and compiler-to-ledger execution of the `MOMENT_MATCHING` constraint. cff-209l is its immediate open prerequisite.
- cff-1o72 follows cff-o7hs and owns deterministic neural-network factors, free-energy-gradient training, solver-gradient evidence, and amortized message initialization.
- cff-wbhu owns the JIT-compiled sweep loop and its single-engine and batched-runtime throughput and status contracts.
- cff-5edo follows cff-wbhu and owns depth-aware long-rollout convergence and warm-started replanning without changing action choices.
- cff-9xmq follows cff-wbhu and owns balanced broadcast expansion, logarithmic message-passing diameter, and marginal agreement with chain expansion at `1e-10`.
- cff-6s31 follows cff-j7mm and owns plate index-table grounding, namespace separation and tied/per-copy blob resolution at nesting depth two or greater, arena and free-energy agreement with materialized expansion, and analytic resource estimates.
- cff-bxk4 follows cff-9bex and owns Gaussian-domain diagram rendering, golden SVG artifacts, and layout-stability tests.
- Arbitrary Gaussian-only or nonlinear GFE epistemics remain outside this epic. cff-dfpi and cff-izjd deliberately selected and delivered the bounded discrete-regime epistemic route. Expanding that scope requires a separate ticket and a new derivation.

## Definitions

- **Arena:** Flat, shape-grouped JAX arrays and index plans derived from a serialized template for compiled execution.
- **Bead:** A node constraint that selects a variational treatment such as moment matching or P-substitution.
- **Bethe free energy (BFE):** The sum of factor-local energy and entropy terms corrected for entropy counted more than once along graph edges.
- **Categorical (Cat) and Dirichlet (Dir):** The original discrete probability and conjugate-prior families in CFFG.
- **CFFG:** Constrained Forney-style Factor Graph.
- **Factor vocabulary:** The set of factor and variable families that one template uses.
- **Generalised free energy (GFE):** The active-inference objective used by nodes with P-substitution constraints.
- **Host/JIT:** Execution controlled by Python on the host compared with execution compiled by JAX's just-in-time compiler.
- **Improper belief:** A message product that does not define a normalizable probability distribution.
- **MOMENT_MATCHING:** The constraint used for expectation-propagation-style Gaussian site approximations.
- **P-substitution:** Replacement of a selected model factor by a posterior belief inside the GFE construction.
- **Semiring:** The pair of aggregation operations that gives message passing sum-product or max-product meaning.
- **Support:** The states to which a probability distribution assigns nonzero probability.
- **Vacuous message:** A neutral message that contributes no information when multiplied with other incoming messages.
- **`vmap`:** JAX's transformation for applying one function across a batch of inputs.

## Notes

**2026-08-08T04:11:28Z**

Retained wiggle-audit candidates and intended meanings after the bounded current scan: integrate = internal compiler/runtime/ledger wiring; n_a = a structurally irrelevant change-path surface, with a concrete reason required; public = documented consumer contract or exported symbol as named by each sentence; register = runtime family coverage through executable dispatch; expose = a callable documented surface; implement = the construction period described by the method; support = either the documented objective/API scope or literal probability support, as stated locally; complete = ticket-scope completion or execution termination; deliver = a repository capability or evidence artifact; ready = dependency-ready according to tk on 2026-08-07; remaining = unfinished prerequisites or explicitly excluded capability limits; approximate = a named mathematical approximation, never unspecified similarity; close = administrative ticket closure after acceptance; check = an automated comparison or deliberate surface review; converge = the runtime CONVERGED/non-converged status contract; correct = the entropy-overcounting correction in the BFE definition, not a free-standing quality claim; equivalent = equal acceptance artifact and test coverage; estimate = the named resource estimator or its analytic output; evidence = named automated tests, references, artifacts, or diagnostics; match = the mathematical term moment matching; pass = the phrase message passing; robust = the AGENTS.md numerical robustness contract; stable = long-rollout behavior or the named layout-stability tests; test = automated test execution or a named test artifact; validate = numeric validation against the reference named in the same criterion; infer = fixed-parameter state or regime inference; learn = compiler membership for parameter-learning machinery; learnable = the named persistent-parameter change-path category; scale = increased domain, plate, rollout, batch, or factor count, or measured throughput load; may = explicit permission for a method or delegation choice; explicit_owner = this epic owns numbered integration outcomes while named prerequisite tickets own Future work; named_ticket_handoff = a concrete Future-work owner; every = each member of a locally named set; only = a deliberate scope boundary or a stated test-selection condition. Intelligible-ticket review: complete.

Intelligible-ticket review: complete.
