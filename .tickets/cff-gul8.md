---
id: cff-gul8
status: open
deps: [cff-cj5h]
links: [cff-cj5h, cff-fxmj, cff-hp3y, cff-4oq8]
created: 2026-07-10T03:31:28Z
type: feature
priority: 3
assignee: coopwilliams
---
# Epistemic sensing readout: expected information gain over candidate DATA ports

## Background in Plain English

Applications sometimes have several sensors but can observe just one of them. They need to ask which missing observation would reduce uncertainty about a named query variable the most. At repository commit `0bf6cfc`, `Engine.marginals` reports one-edge beliefs and `Engine.node_beliefs` reports ordinary factor beliefs, but the engine does not report this value for candidate DATA ports.

The original ticket recorded a consumer-side NumPy experiment in which an expected-information-gain glimpse policy improved classification accuracy by 27 percentage points over random sensor choice under the same sensing budget. No versioned fixture or raw result for that experiment is present at commit `0bf6cfc`, so the measurement is motivation rather than acceptance evidence. The engine readout will replace the duplicated NumPy calculation for the discrete naive-Bayes graph shape and will gain its own independent numerical checks.

This ticket owns `Engine.eig(state, query_edge, ports)`. For every requested missing discrete DATA port, the method reports the mutual information between the named query edge and the port's edge under the supplied state. Version 1 owns exact discrete sum-product components whose factor graph is acyclic and whose query-to-port path uses ordinary belief-propagation factors. This boundary includes equality factors used to fan a naive-Bayes class variable out to several likelihood factors. This ticket does not assign sensing costs or choose a port; its caller combines the returned information values with the application's decision rule.

Ticket cff-cj5h is the prerequisite recorded in the dependency graph. It owns the separate diagnostic that explains the mutual-information contribution already present inside a GFE objective. This ticket owns predictive information about opening a DATA port. The two APIs must use distinct names and documentation so a caller cannot mistake an objective decomposition for a sensor-value readout.

## Outcomes / Acceptance Criteria

1. `Engine.eig(state, query_edge, ports)` is a documented `Engine` readout supported for downstream callers.

2. The readout returns one mapping entry for each requested candidate port.

3. Each returned mapping key equals the corresponding candidate port name.

4. Iterating over the returned mapping yields candidate port names in the caller's input order.

5. Each returned value uses natural-log units.

6. Each returned value is the mutual information `I(query_edge; port_edge)` under the supplied state.

7. The readout does not mutate the supplied `State`.

8. The supported graph boundary is enforced.

   8.1. The query name identifies a compiled discrete edge that is not a parameter carrier.

   8.2. Each candidate name identifies a compiled discrete DATA port.

   8.3. Each candidate DATA port carries the missing-observation message installed by `Engine.init_state()` or `Engine.bind_data(..., {port: None})`.

   8.4. The query edge differs from every candidate port edge.

   8.5. The connected component containing the query edge and each candidate port edge is acyclic.

   8.6. The connected component containing the query edge and each candidate port edge uses sum-product inference.

   8.7. Every factor on a supported query-to-port path is an ordinary discrete belief-propagation factor reported by `Engine.node_beliefs`.

   8.8. Equality factors on a supported query-to-port path retain their exact shared-variable meaning.

9. Invalid requests fail at the public boundary.

   9.1. An unknown query edge raises `KeyError` and names that edge.

   9.2. An unknown candidate port raises `KeyError` and names that port.

   9.3. A candidate that is not a DATA port raises `ValueError` and names that candidate.

   9.4. A candidate DATA port bound to an observed category raises `ValueError` and names that port.

   9.5. A non-discrete query edge raises `ValueError` and names that edge.

   9.6. A non-discrete candidate port raises `ValueError` and names that port.

   9.7. A query and candidate in different connected components raise `ValueError` and name both endpoints.

   9.8. A request whose component contains a cycle raises `ValueError` and states that version 1 requires an acyclic component.

   9.9. A request whose component uses max-product inference raises `ValueError` and states that version 1 requires sum-product inference.

   9.10. A request whose query-to-port path crosses a VMP factor raises `ValueError` and names that factor.

   9.11. A request whose query-to-port path crosses a GFE composite raises `ValueError` and names that composite.

   9.12. A request whose query-to-port path crosses an unsupported constraint factor raises `ValueError` and names that factor.

10. Independent numerical oracles establish the mathematical result.

   10.1. On a hand-enumerated binary tree, each reported value agrees with mutual information computed from the enumerated query-port joint distribution within `128 * finfo(dtype).eps` times the number of cells in that joint distribution.

   10.2. On a discrete tree where the query and candidate edges are separated by an equality factor, the reported value agrees with direct enumeration under the tolerance in criterion 10.1.

   10.3. The naive-Bayes oracle fixture has at least three candidate DATA ports.

   10.4. On the naive-Bayes fixture, each reported value agrees with a standalone NumPy calculation from the class posterior and the corresponding likelihood table under the tolerance in criterion 10.1.

   10.5. The candidate ordering produced by the engine equals the ordering produced by the standalone NumPy calculation on the fixture in criterion 10.3.

   10.6. The standalone NumPy oracle does not call `Engine.eig`.

   10.7. The standalone NumPy oracle does not call the engine's path-contraction helper.

11. Sparse support has defined behavior.

   11.1. A joint cell with probability zero contributes exactly zero to the mutual-information sum.

   11.2. A supported deterministic query-port relation returns the entropy of the query marginal under the tolerance in criterion 10.1.

   11.3. A candidate independent of the query returns zero under the tolerance in criterion 10.1.

   11.4. The readout introduces no positive probability floor for a literal probability zero.

   11.5. An intermediate separator state with zero posterior probability does not create `NaN` through a zero-divided-by-zero calculation.

   11.6. A query-port component with no joint support raises `ValueError` and names the unsupported pair.

12. Vacuous and improper inputs have defined behavior.

   12.1. Vacuous incoming messages act as multiplicative identities when the model factors still define a proper query-port joint distribution.

   12.2. A proper query-port joint distribution produced from vacuous incoming messages agrees with direct enumeration under the tolerance in criterion 10.1.

   12.3. A non-finite message on the query-to-port path raises `ValueError` and names the affected path.

   12.4. An all-log-zero incoming message on the query-to-port path raises `ValueError` and names the affected path.

13. Float32 behavior follows the numerical robustness contract.

   13.1. A supported float32 fixture returns finite information values.

   13.2. The float32 result agrees with the independent oracle under the tolerance in criterion 10.1 evaluated with float32 epsilon.

   13.3. The float32 calculation preserves literal probability zeros.

   13.4. Any tolerance, roundoff guard, or nonnegativity correction is derived from `finfo(dtype)` and the contracted domain size.

14. An uncomfortable-scale fixture uses a query cardinality of 64.

15. The uncomfortable-scale fixture includes at least 64 candidate DATA ports.

16. Every information value from the uncomfortable-scale fixture is finite.

17. The uncomfortable-scale results agree with independently computed naive-Bayes values under the tolerance in criterion 10.1.

18. `README.md` states that version 1 supports missing discrete DATA ports in acyclic sum-product components.

19. `README.md` states the rejected graph and domain boundaries.

   19.1. `README.md` states that version 1 rejects a loopy component.

   19.2. `README.md` states that version 1 rejects max-product inference.

   19.3. `README.md` states that version 1 rejects a path through a VMP factor.

   19.4. `README.md` states that version 1 rejects a path through a GFE composite.

   19.5. `README.md` states that version 1 rejects a continuous query or candidate.

   19.6. `README.md` states that version 1 rejects a path through an unsupported constraint factor.

20. `README.md` distinguishes predictive port information from cff-cj5h's GFE objective diagnostic.

21. Focused automated tests cover the delivered boundary.

   21.1. A focused test calls the documented readout and checks its returned mapping.

   21.2. A focused test exercises every supported-boundary criterion under item 8.

   21.3. A focused test exercises every rejection criterion under item 9.

   21.4. Focused tests exercise both independent oracle families under item 10.

   21.5. A focused test exercises literal zero support.

   21.6. A focused test exercises vacuous incoming messages.

   21.7. A focused test exercises an improper incoming message.

   21.8. A focused test exercises float32 calculation.

   21.9. A focused test exercises the uncomfortable-scale fixture.

## Methodological Suggestions

These suggestions are optional. An implementer may use another sound method that satisfies every numbered outcome.

Use the compiled arena to resolve the query edge and DATA-port edges. Find the unique factor-graph path between each pair. One likely calculation contracts the ordinary discrete factor beliefs along that path and marginalizes every off-path axis. Where adjacent factor beliefs share a separator edge, use a masked contraction that never evaluates zero divided by zero. Normalize the resulting two-edge joint before evaluating `sum p(q, y) * (log p(q, y) - log p(q) - log p(y))` over supported cells.

Reuse the graph's compiled topology rather than reconstructing model meaning from factor or edge names. Cache path plans by `(query_edge, candidate_port)` if profiling shows repeated topology work matters. Keep the numerical calculation in log space until the final normalized joint when that makes literal zero support safer. Derive any roundoff correction from the message dtype and number of contracted cells, and explain the invariant protected by that correction.

The change-path index in `docs/ARCHITECTURE.md` has no exact row for a new host-side engine readout. Treat `src/cffg/engine.py:Engine.eig` as the public authority. Existing `Arena` edge, group, and DATA-port metadata are derived inputs. If the implementation adds compiled path metadata, keep `src/cffg/compiler.py:Arena`, the readout, the estimator, and serialization reconstruction in agreement. No new serialized intermediate-representation field, factor family, message rule, free-energy term, learnable, or checkpoint field is intended.

Review the duplicated execution surfaces as follows:

- Host/JIT: `Engine.eig` is a host readout from a JAX-produced `State`; no second JIT readout API is required. Compare states produced by the JIT runtime with the host-side independent oracles.
- Flooding/scheduled: both execution routes feed the same readout. Compare converged flooding and exact scheduled states on a supported tree under the criterion 10.1 tolerance.
- Single-instance/`vmap`: version 1 reads one `State`. A Colony or batched EIG API is not part of this ticket.
- Current/legacy checkpoint decoding: not applicable because the readout changes neither `State` storage nor checkpoint decoding.

Extend focused engine tests or add one narrowly named EIG test file. Use direct joint enumeration for the small graph oracle and a standalone class-posterior/likelihood calculation for the naive-Bayes oracle. Update `README.md` with the public call and result units. Update `docs/ARCHITECTURE.md` with the supported readout boundary. Update `docs/LIMITATIONS.md` with the rejected graph and domain families. After the final code change, run the repository's required short suite once through `uv run python tools/dashboard.py --run-suite`.

## Future work

No downstream ticket receives work from this ticket. Ticket cff-cj5h remains its prerequisite and owns the GFE objective diagnostic, not predictive DATA-port EIG. Batched or Colony readouts, sensing-cost optimization, automatic port choice, loopy approximations, VMP approximations, GFE-composite paths, unsupported constraint paths, and continuous-domain information measures have no ticket owner in this handoff and require separate tickets before implementation.

## Definitions

- **Acyclic component:** A connected part of the factor graph with no cycle, so every two nodes have one path between them.
- **Belief propagation (BP):** Message passing that forms a factor belief from the factor and its incoming messages. On a tree with sum-product rules, the resulting beliefs are exact marginals of the represented distribution.
- **Candidate DATA port:** A named sensor boundary supplied to `Engine.eig` for evaluation.
- **Connected component:** A maximal set of graph nodes and edges joined by paths.
- **DATA port:** A boundary edge where runtime observation evidence enters the graph.
- **Dtype:** A numeric storage type such as float32 or float64.
- **Expected information gain (EIG):** Before observing a candidate, the mathematical expectation of how much that observation will reduce uncertainty about the query. In this ticket it equals query-candidate mutual information.
- **Free-energy ledger:** The engine's decomposition of Bethe or generalized free energy into reviewable terms.
- **Generalized free energy (GFE):** The active-inference objective that combines model evidence, preferences, and epistemic terms under P-substitution constraints.
- **GFE composite:** The compiled group of factors and constraints that implements one GFE update.
- **Glimpse policy:** A decision rule that selects which sensor or input region to observe next.
- **Improper message:** A message that cannot participate in a finite normalized belief, including an all-log-zero categorical message.
- **JAX:** The array and compiled numerical computing library used by the engine runtime.
- **Literal probability zero:** A category excluded from support and represented in log space by the runtime's zero-probability value.
- **Max-product inference:** Message passing that selects maximum-scoring assignments instead of summing probability mass.
- **Missing observation:** A DATA-port message installed by `Engine.init_state()` or by binding `None`; it adds no observation evidence.
- **Mutual information (MI):** `I(Q;Y) = sum q(Q,Y) log(q(Q,Y) / (q(Q)q(Y)))`, measured here in nats.
- **Naive Bayes:** A model in which candidate observations are conditionally independent given one class variable.
- **Natural-log units:** Information measured with the natural logarithm; one such unit is a nat.
- **Node belief:** The normalized joint belief over the variables incident on one ordinary factor.
- **Ordinary factor:** A compiled factor handled by the registered BP rule rather than by a VMP update, GFE composite, or constraint-specific solver.
- **Parameter carrier:** A discrete-shaped edge that stores a parameter table and does not represent a categorical random variable.
- **Predictive port information:** Mutual information between a query variable and a missing observation before that observation is bound.
- **Query edge:** The named discrete edge whose uncertainty the caller wants a candidate observation to reduce.
- **Separator edge:** An internal variable shared by adjacent factors on the unique path between the query edge and a candidate port edge.
- **Sum-product inference:** Message passing that sums probability mass to compute marginal beliefs.
- **Support:** The assignments with nonzero probability.
- **Tree-local:** Restricted to a query and candidate joined through an acyclic factor-graph component.
- **Vacuous message:** The multiplicative identity message, which contributes no evidence.
- **Variational message passing (VMP):** Message passing under a declared mean-field factorization rather than the exact tree joint distribution.
- **Version 1:** The capability boundary delivered by this ticket.

## Notes

**2026-08-07**

Evidence and source preservation:

- The 27-percentage-point accuracy result is preserved from the ticket text introduced by commit `a766917`; no versioned fixture or raw output for it was found at repository commit `0bf6cfc`.
- `src/cffg/engine.py:Engine.marginals` and `Engine.node_beliefs` establish the available readout inputs at repository commit `0bf6cfc`.
- `docs/ARCHITECTURE.md` states that ordinary discrete BP node beliefs on trees are exact and that GFE composites are not included in `Engine.node_beliefs`.
- Closed ticket cff-4oq8 supplies the repository's independent normalized-joint MI oracle pattern and hostile-input precedent. It measured cue MI of 0.3681 nats and distractor MI of zero in its separate behavioral task.
- Closed tickets cff-fxmj and cff-hp3y explain why an engine readout must not reuse a GFE objective term as though it were arbitrary predictive mutual information.

Retained wiggle candidates after the 2026-08-07 bounded scan (candidate -> intended meaning): preserve -> literal probability-zero value identity or semantic preservation of the historical ticket claim; public -> the documented `Engine.eig` contract available to downstream callers; register -> runtime BP-family dispatch; reusable -> internal topology reuse or the prohibited reuse of a GFE objective term; same -> equal sensing-budget configuration or one shared readout implementation; available -> a callable readout input present at the pinned repository commit; implement -> planned ticket work or the defined operation of a GFE composite; support -> the explicitly accepted graph boundary or nonzero probability assignments; complete -> ticket-review completion in the exact marker; deliver -> the version 1 ticket promise; remaining -> cff-cj5h continuing as the recorded prerequisite; approximate -> named loopy or VMP approximation methods outside version 1; close -> administrative completion of the named historical tickets; check -> an automated numerical check; converge -> the `RunStatus.CONVERGED` state used for route comparison; estimate -> the repository resource estimator; evidence -> empirical acceptance material or observation data entering a DATA port, as selected by the sentence; exact -> exact tree inference, exact shared-variable semantics, literal-zero arithmetic, or the absence of an applicable change-path row; oracle -> an independently implemented numerical reference; pass -> the act of sending messages in the definition of a named message-passing method; robust -> compliance with the repository numerical-robustness contract; sound -> an alternative method that satisfies every acceptance outcome; test -> an automated executable check; infer -> the named sum-product, max-product, or active-inference method; learnable -> the repository's declared learnable-parameter feature category; scale -> domain size in the 64-state and 64-port fixture; small -> the binary direct-enumeration fixture; can -> the application is able to observe one sensor; expected -> mathematical expectation; intended -> the declared absence of specified repository changes; may -> permission to choose an alternative method; should -> the map records that no normative ticket sentence retains this candidate; will -> the ticket's promised future readout or the mathematical future observation in the EIG definition; explicit_owner -> this ticket owns predictive port EIG, cff-cj5h owns the GFE diagnostic, and unassigned extensions have no ticket owner; named_ticket_handoff -> cff-cj5h is the concrete prerequisite owner of the GFE diagnostic; all -> every log entry in an all-log-zero message; any -> every tolerance or correction, or an unrestricted optional implementation correction; at_least -> an inclusive minimum candidate count; every -> each member of the locally named candidate, factor, result, or outcome set; current -> the named current/legacy checkpoint review pair; existing -> arena metadata present before this ticket's implementation; read -> machine access to one `State`.

Intelligible-ticket review: complete.
