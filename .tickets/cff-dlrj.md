---
id: cff-dlrj
status: open
deps: []
links: [cff-0f3g, cff-nkbj]
created: 2026-08-10T02:58:08Z
type: task
priority: 3
assignee: coopwilliams
parent: cff-ahps
tags: [rslds, hybrid, derivation]
---
# Derive a destination-softmax recurrence VMP and free-energy contract

## Background in Plain English

The later Linderman `ssm` package uses one normalized softmax over destination
logits instead of a sequence or tree of binary logistic gates. For more than
two destinations, an affine softmax cannot be converted to affine
stick-breaking gates without introducing a state-dependent log-sum-exp term.
The current recurrence rule, Pólya–Gamma statistics, and node energy therefore
cannot be reused unchanged.

This ticket decides the mathematical contract before anyone adds a production
factor. It asks whether CFFG can support a source-specific destination-softmax
recurrence with a reviewable VMP approximation and free-energy quantity. It is
linked to the literature and tree roadmaps but blocks neither.

## Outcomes / Acceptance Criteria

1. The target conditional distribution is written explicitly as normalized
   destination logits conditioned on source regime and Gaussian drive, with an
   exact mapping to the pinned `ssm` variant where their model families overlap.
2. The derivation identifies the softmax gauge freedom and chooses a reference
   destination, zero-sum constraint, or another explicit identifiable
   parameterization without using a numeric ridge.
3. The work proves that affine destination softmax and affine sequential or
   tree stick-breaking are non-equivalent for `K > 2`, except for named
   restricted cases.
4. The variational treatment states whether Gaussian expectations are exact,
   bounded, quadrature-based, or sampled and justifies that choice from a
   node-local constrained objective.
5. Messages to destination regime, source regime, Gaussian drive, and any
   learnable coefficient variable are derived as stationary conditions of the
   declared local Lagrangian.
6. The node-local free-energy term, entropy ownership, edge corrections, and
   any approximation or bound direction are explicit.
7. P-substitution and GFE behavior are either derived separately or declared
   unsupported. A fixed-point equation alone is not presented as a convergence
   guarantee.
8. Learning semantics identify the sufficient statistics or nonconjugate
   optimizer required for source-specific and shared coefficients.
9. A pure executable oracle checks probability normalization, gauge
   invariance, messages or gradients, and node energy on small `K=2`, `K=3`,
   and `K=4` cases without calling a future production factor.
10. Exact-zero categorical support remains exact. Empty support, improper or
    vacuous required Gaussian inputs, and non-finite objective values have
    explicit loud outcomes.
11. Float32 uses dtype-scaled guards, and a `K=32`, drive-dimension `16` case
    exercises the proposed computation without a destination-cubed tensor.
12. The conclusion recommends one of three bounded outcomes: proceed with
    named implementation tickets, perform named prerequisite research, or
    stop because the proposed objective cannot meet CFFG's free-energy
    contract.
13. This ticket adds no production `FactorType`, compiler plan, runtime
    dispatch, public builder option, or parity claim.

## Methodological Suggestions

These suggestions are optional. The implementer may choose another sound
method that satisfies the outcomes.

- Start from the **New factor or rule family** change-path row and mark every
  production surface N/A for this derivation-only ticket while naming what a
  later implementation must cover.
- Compare multinomial variational bounds, controlled quadrature, and explicit
  sampling by the objective they report, not only by predictive accuracy.
- Use automatic differentiation only as a check on an independently stated
  objective, not as a substitute for defining the free-energy term.

## Future work

If the derivation supports implementation, create separate fixed-inference,
learning, and integration tickets under cff-ahps. cff-nkbj continues with the
decision-list paper-equation reference, and cff-0f3g continues with binary-tree
topologies regardless of this ticket's result.

## Definitions

- **Destination softmax:** A normalized categorical distribution formed by
  exponentiating one logit per destination and dividing by their sum.
- **Gauge freedom:** A parameter transformation that leaves every probability
  unchanged and must be removed explicitly for identifiable optimization.
- **Log-sum-exp:** The stable function `log(sum(exp(logits)))` that couples all
  softmax destinations through one normalizer.
- **VMP:** Variational message passing, in which local messages follow from
  stationary conditions of a factorized variational objective.

## Notes

**2026-08-10T03:20:00Z**

Wiggle-audit retained candidates and intended meanings: fixed=future
fixed-parameter inference; integrate=future consumer integration; n_a=every
production surface deliberately excluded from this derivation; public=the
excluded builder/runtime surfaces; reusable=the current tree rules that cannot
be reused unchanged; implement=one of the bounded research conclusions or
future tickets; support=the theoretical CFFG objective scope or literal
categorical support; remaining=the zero-support residual wording;
approximate=the explicitly named variational method/bound; sufficient=the
derived learning statistics; check=the executable oracle or AD comparison;
converge and fixed_point=the explicit non-guarantee for stationary equations;
equivalent=the proved model-family non-equivalence; exact=the pinned mapping,
analytic expectation, or literal zero; oracle=the pure derivative/probability
reference; parity=an explicitly excluded production claim; pass=message passing
terminology; sound=another method satisfying outcomes; stable=the numerical
log-sum-exp function; infer, learn, and learnable=the named future message and
coefficient semantics; scale and small=the `K=32` versus finite oracle cases;
can=the theoretical possibility being decided; may=optional method; all, any,
every, and only=the coupled destinations, named messages/approximations,
production surfaces, and derivation-only boundary; current and later=the
checked-in recurrence family and historical/future ssm ordering. Capability
owner: cff-dlrj owns only the mathematical feasibility decision and executable
oracle. Intelligible-ticket review: complete.

Classification-note additions: complete=the enumerated derivation contract;
intended=the contextual assignments in this note; explicit_owner and
named_ticket_handoff=cff-dlrj owns derivation while future implementation
tickets would own production surfaces.
