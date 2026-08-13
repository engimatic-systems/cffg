---
id: cff-lv65
status: open
deps: [cff-602p]
links: []
created: 2026-08-10T02:58:17Z
type: feature
priority: 2
assignee: coopwilliams
parent: cff-0f3g
tags: [rslds, hybrid, inference]
---
# Execute balanced-tree recurrence messages and free energy

## Background in Plain English

cff-4sor defines the tree mathematics, and cff-602p makes topology part of the
serialized model. This ticket makes a balanced tree executable for fixed
coefficients. The change must reach every place where the current specialized
hybrid recurrence plan runs or contributes to the objective.

The balanced rule remains a Pólya–Gamma/Jaakkola–Jordan variational update.
Only its mapping from binary gates to categorical destinations changes. It
must preserve the true declared variational objective and must not borrow the
decision-list prefix formula where the topology does not justify it.

## Outcomes / Acceptance Criteria

1. `recurrence_link` compiles every valid balanced topology produced by the
   cff-602p constructor with the coefficient shapes defined by cff-4sor.
2. The compiler lowers topology to static gate, child, path, and orientation
   tables reconstructed from the serialized template.
3. Synchronous and mixed-schedule execution send the derived messages to the
   destination regime, source regime, and Gaussian drive.
4. The fixed-parameter node contribution implements the cff-4sor bounded
   expected log transition, node entropies, and ordinary Bethe edge-counting
   contract.
5. Runtime status reports converged, budget-exhausted, diverged, or infeasible
   through the existing engine certificate. A non-finite message, moment, or
   free-energy value cannot return converged.
6. P-substitution, GFE use, and max-product remain rejected with specific
   diagnostics unless a separate derivation has landed.
7. Small balanced-tree messages and node energy match the independent cff-4sor
   oracle for multiple source beliefs, Gaussian moments, and label
   permutations.
8. Fixed-point beliefs and Bethe free energy match direct quadrature or exact
   enumeration on compact models within stated dtype tolerances.
9. The same balanced categorical probabilities match the pinned
   pypolyagamma balanced-tree convention after an explicit leaf and branch
   mapping.
10. Decision-list templates retain their pre-ticket messages, beliefs, status,
    and free energy within the existing regression tolerance.
11. Exact-zero source or destination support remains exact. Empty support is
    infeasible, and materially invalid Gaussian precision or gate moments are
    divergence rather than regularized beliefs.
12. Float32 uses only dtype- and scale-derived guards and agrees with the
    independent oracle within a registered float32 tolerance.
13. A balanced `K=32`, drive-dimension `16` fixture completes without a
    source-by-destination-by-gate tensor and records bounded persistent plan
    storage.
14. Host/JIT, flooding/mixed schedule, and sequential/`vmap` execution agree
    on balanced fixtures within stated dtype tolerances.
15. Compiler grouping, slot addressing, parameter packing, schedule
    classification, BFE dispatch, readouts, serialization manifests, resource
    estimation, visualization, and public exports are updated or marked not
    applicable with a concrete reason.
16. README.md, docs/ARCHITECTURE.md, docs/DERIVATIONS.md, and
    docs/LIMITATIONS.md agree on topology semantics, the bound direction,
    supported modes, numerical behavior, and remaining learning gap.
17. This ticket does not add shared-gate posterior ownership or balanced-tree
    parameter learning.

## Methodological Suggestions

These suggestions are optional. The implementer may choose another sound
method that satisfies the outcomes.

- Use the **New factor or rule family** and **Runtime numerical or status
  invariant** change-path rows even if the serialized factor type name remains
  `recurrence_link`.
- Build leaf scores with a tree dynamic program and accumulate drive terms
  from gate exposure so production execution does not allocate a cubic path
  indicator tensor.
- Keep decision-list and balanced evaluation behind one tested topology-table
  contract rather than two unrelated engine branches.

## Future work

cff-4c7t consumes balanced fixed-parameter inference and adds explicit
parameter ownership, statistic pooling, and learning. cff-9vab owns public
rSLDS integration and measured scale comparisons.

## Definitions

- **Fixed-parameter inference:** Belief estimation with supplied factor
  coefficients that are not updated during the run.
- **Gate exposure:** Posterior probability that a destination path reaches a
  particular internal binary gate.
- **Mixed schedule:** The engine schedule that alternates discrete, Gaussian,
  and local variational coordinates for hybrid factors.
- **Topology table:** A static array derived from serialized tree structure and
  used by compiled JAX rules.

## Notes

**2026-08-10T03:20:00Z**

Wiggle-audit retained candidates and intended meanings: fixed=fixed supplied
coefficients or fixed-point beliefs; integrate=the cff-9vab future workflow;
preserve=the declared variational objective; public=the named export/readout
surfaces; register=the numeric tolerance fixed by the test; same=the mapped
pypolyagamma probabilities; implement=the production node term; support=literal
categorical support or documented objective scope; complete=termination of the
`K=32` fixture; land=a separately derived prerequisite; remaining=preserved
rejections or learning deferred to cff-4c7t; material=invalidity beyond a
dtype-scaled tolerance; certify and converge=the engine's complete status;
estimate=resource accounting or fixed-belief estimation; exact=finite
enumeration or literal support; match=bounded numeric agreement with named
authorities; oracle=the cff-4sor independent reference; sound=another method
satisfying outcomes; test=the shared topology-table regression; accumulate and
exposure=gate-statistic algebra; infer and learn=fixed-parameter inference and
explicitly deferred coefficient learning; scale and small=the named `K=32`
versus compact fixtures; expected=mathematical expected log transition;
may=optional method; explicit_owner and named_ticket_handoff=cff-4c7t/cff-9vab;
every and only=the enumerated execution surfaces and sole changed topology
mapping; current and existing=the checked-in recurrence plan/status contract.
Capability owner: cff-lv65 owns fixed-weight production inference and BFE only.
Intelligible-ticket review: complete.

Classification-note additions: check=the named oracle, quadrature, or execution
pair; intended=the contextual assignments in this note; deferred=parameter
learning and public integration owned by cff-4c7t/cff-9vab.
