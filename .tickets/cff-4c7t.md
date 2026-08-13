---
id: cff-4c7t
status: open
deps: [cff-lv65, cff-4apw, cff-y4nx]
links: []
created: 2026-08-10T02:58:17Z
type: feature
priority: 2
assignee: coopwilliams
parent: cff-0f3g
tags: [rslds, hybrid, learning]
---
# Learn explicitly shared and source-specific tree gates

## Background in Plain English

Full source-specific recurrence coefficients let each regime adapt
independently. Tied coefficients pool evidence and use fewer parameters when
the model says several gate uses share one effect. Copying equal arrays is not
true tying because separate posteriors can diverge during continual learning.

This ticket represents parameter identity explicitly and makes Pólya–Gamma
learning obey it for decision-list and balanced topologies. Sharing belongs to
the generative model, not to a variational form constraint. The ticket combines
ownership and learning because the ownership map determines exactly which
sufficient statistics must be pooled.

## Outcomes / Acceptance Criteria

1. Serialized recurrence structure maps every source-and-gate use to one
   explicit coefficient owner.
2. The default ownership map gives every source-and-gate pair its own owner and
   preserves the existing full source-specific model.
3. A source-tied constructor maps the same internal gate across every source
   regime to one owner. A general valid ownership map may express partial
   sharing without copying posterior state.
4. Validation rejects missing owners, unused owners, out-of-range owner
   indices, incompatible coefficient shapes, and ownership that disagrees with
   the topology.
5. JSON, structure hashing, bundles, plates, learnable declarations,
   checkpoints, and model edits preserve owner identity and distinguish it
   from coefficient values.
6. The compiler stores one persistent coefficient and one posterior per owner,
   while each factor use gathers the owner selected by the serialized map.
7. Fixed-parameter messages and free energy are identical whether equal
   coefficients are represented by several source-specific owners or one tied
   owner.
8. Pólya–Gamma linear and quadratic statistics from every use of one owner are
   summed exactly once before its posterior update.
9. Source-specific owners update independently, and evidence for one source
   cannot change another source's posterior.
10. Source-tied learning matches a direct pooled-statistic reconstruction and
    the recurrence-only collapse oracle from cff-64ri on a common decision-list
    fixture.
11. `RSLDSLearning`, the observed-only `fit_rslds` coordinate, parameter
    installation, rollover, and public recurrence readouts obey the same owner
    map for both decision-list and balanced topologies.
12. A current source-specific checkpoint restores without changing parameter
    identity or values. Any schema migration for older checkpoints is explicit
    and fails loudly when identity cannot be reconstructed.
13. Exact-zero gate exposure adds exact zero statistics. Improper beliefs,
    invalid covariance, non-finite statistics, or a non-positive-definite
    posterior fail atomically without a ridge or partial update.
14. Float32 statistic construction uses the existing dtype-scaled PG limits,
    and posterior accumulation states its deliberate accumulation dtype.
15. A `K=32`, drive-dimension `16` case covers source-specific and source-tied
    owners without storing duplicate posterior matrices for tied uses.
16. Host/JIT statistic construction and sequential/`vmap` execution agree;
    current/latest checkpoint and tied/source-specific pairs are tested or
    marked not applicable with a concrete reason.
17. README.md documents owner-based sharing and continual learning.
    docs/ARCHITECTURE.md records generative sharing semantics.
    docs/DERIVATIONS.md records pooled natural statistics.
    docs/LIMITATIONS.md states the point-rollover approximation and supported
    ownership forms.

## Methodological Suggestions

These suggestions are optional. The implementer may choose another sound
method that satisfies the outcomes.

- Use the **New IR field** and **Learnable or checkpoint feature** change-path
  rows. Review validation, JSON/hash, plate scoping, compiler resolution,
  runtime readouts and writes, agent cadence, checkpoints, bundles, and public
  construction.
- Represent ownership as a canonical integer index table and reduce
  sufficient statistics by owner before calling the posterior update.
- Preserve one full source-specific path as the compatibility oracle before
  compacting tied storage.

## Future work

cff-9vab uses this capability for end-to-end balanced-tree rSLDS fitting and
measured topology comparisons. A generic parameter-sharing abstraction for
unrelated factor families remains outside this ticket unless a separate owner
is created.

## Definitions

- **Coefficient owner:** The unique stored parameter and posterior used by one
  or more recurrence gate positions.
- **Continual learning:** Updating retained parameter beliefs as new sequences
  arrive without forgetting that earlier evidence existed.
- **Partial sharing:** A model in which some, but not all, gate positions use
  the same coefficient owner.
- **Point rollover:** Installing a posterior point summary as the factor
  coefficient for a later inference run.
- **Source-tied:** One gate coefficient shared across every source regime.

## Notes

**2026-08-10T03:20:00Z**

Wiggle-audit retained candidates and intended meanings: canonical=the unique
owner-index representation; end_to_end=cff-9vab's builder-through-fit workflow;
fixed=fixed-coefficient message parity; generic=an explicitly excluded
cross-family abstraction; preserve=source-specific behavior or owner identity;
public=the recurrence readout and construction API; same=one owner identity or
the same gate across sources; support=the enumerated ownership forms;
remaining=unrelated factor sharing excluded; approximate=the named point
rollover; sufficient=owner-reduced natural statistics; evidence=posterior data
accumulated across sequences; exact=owner membership, single update counting,
or literal zero; match=bounded agreement with the pooled reconstruction;
oracle=the named collapse or compatibility authority; sound=another method
satisfying outcomes; test and validate=the checkpoint pairs and structural
owner checks; accumulate and exposure=posterior-statistic operations;
fit, infer, and learn=the named `fit_rslds`, fixed-run, and continual-update
workflows; learnable and rollover=the existing project contracts extended here;
scale=the named dtype or `K=32` case; can=posterior divergence when parameters
are merely copied; may=optional partial sharing/method; explicit_owner=one
coefficient owner in the model; all, any, every, and only=the locally named
sharing sets, migrations, uses, and recurrence-only oracle; current, existing,
and later=the checkpoint/rule baseline or ordered future inference run.
Capability owner: cff-4c7t owns recurrence gate identity, pooled statistics,
and their fitting/continual-learning consumers. Intelligible-ticket review:
complete.

Classification-note additions: complete=the source-to-owner map and consumer
set enumerated by the criteria; check=the structural, checkpoint, or statistic
comparison; parity=fixed-message equality under tied versus copied equal
coefficients; integrate=cff-9vab's consumer workflow; intended=the contextual
assignments in this note;
named_ticket_handoff=cff-9vab as the integration/benchmark owner.
