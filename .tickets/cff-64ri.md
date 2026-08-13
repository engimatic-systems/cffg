---
id: cff-64ri
status: open
deps: [cff-akmp]
links: []
created: 2026-08-10T02:58:08Z
type: feature
priority: 1
assignee: coopwilliams
parent: cff-nkbj
tags: [rslds, reference, learning]
---
# Implement the source-specific rSLDS paper-equation Gibbs adapter

## Background in Plain English

The Linderman rSLDS paper defines a separate affine recurrence
`R[z_t] @ x_t + r[z_t]` for each source regime. No unmodified executable in
the audited `recurrent-slds` history learns those source-specific slopes. The
pinned learner shares each continuous slope across source regimes, and its
active NASCAR path removes even the source-specific offset.

This ticket adds an independent reference implementation of the paper's full
recurrence to the locked legacy runtime from cff-akmp. The adapter is derived
from the paper and supplement, not from CFFG code. Its purpose is to give
cff-nkbj a fair same-model comparison while preserving CFFG's more expressive
source-specific capability.

Because the adapter reuses pinned legacy sampling machinery, separate
versioning alone does not make every calculation independent. The ticket must
record which paper equation and legacy primitive owns each calculation and
check every adapter-added surface with a small direct oracle that shares no
adapter or legacy state-update helper.

## Outcomes / Acceptance Criteria

1. The reference adapter is versioned separately from the pinned upstream
   checkout and records both its base commit and adapter commit.
   1.1. A provenance ledger maps every adapter probability, conditional, and
   information-potential surface to exact paper or supplement equation IDs,
   reused legacy primitives, adapter-owned code, and direct oracle.
2. The adapter implements one affine coefficient vector for every source
   regime and decision-list gate, equivalent to
   `psi[i,l](x) = R[i,l] @ x + r[i,l]`.
3. Transition evaluation, recurrence-parameter resampling, continuous-state
   information potentials, and Pólya–Gamma auxiliary resampling all select the
   same source-specific coefficient row.
4. The adapter uses the explicit decision-list topology established by
   cff-akmp and never relies on the dependency's default tree.
5. The adapter imports no CFFG inference, recurrence, learning, or free-energy
   implementation.
6. Compact NumPy or direct-linear-algebra oracles verify every adapter-added
   probability, recurrence conditional, continuous-state information
   potential, and Pólya–Gamma auxiliary surface.
   6.1. The oracles import neither CFFG nor the adapter or legacy helper that
   computes the value under test.
   6.2. Each oracle names the exact published equation it evaluates and checks
   every source/gate selection axis that the adapter adds.
7. Small exact enumeration verifies destination probabilities, normalization,
   path orientation, and leaf labels for at least `K=2`, `K=3`, and `K=4`.
8. A source-isolation test changes one source's coefficients and shows that
   only transitions conditioned on that source change.
9. A collapse test ties every source coefficient and reproduces the forced
   decision-list recurrence-only upstream transition and conditional values
   within a stated dtype tolerance.
10. A compact observed-only Gibbs fit receives no generating states, regimes,
    or parameters and returns every sampled-chain status, including failures.
11. Exact zero transition support remains zero when imposed by the reference
    topology or observations; non-finite designs, coefficients, auxiliaries,
    or conditional moments fail loudly.
12. The reference dtype boundary is explicit. If the legacy sampler requires
    float64, the ticket does not imply float32 support; probability evaluation
    at `K=32` still supplies the uncomfortable-cardinality normalization case.
13. Documentation calls the result an independent paper-equation
    Pólya–Gamma Gibbs reference adapter. It does not call it the unmodified
    authors' implementation.
14. The artifact records the provenance ledger, paper and supplement equation
    IDs, upstream and adapter commits, environment hash, exact commands, seeds,
    configurations, test outputs, and content hashes.

## Methodological Suggestions

These suggestions are optional. The implementer may choose another sound
method that satisfies the outcomes.

- Use the interaction design `one_hot(z_t) ⊗ [x_t; 1]` so each existing
  Pólya–Gamma regression row receives source-specific sufficient statistics.
- Keep probability evaluation and source-row gathering in one adapter-owned
  implementation so state potentials and auxiliary updates cannot disagree.
- Treat the recurrence-only collapse as the executable bridge to the pinned
  upstream learner, not as proof that the full adapter is upstream code.

## Future work

cff-0l31 consumes this adapter to freeze cross-implementation mappings and the
statistical protocol. cff-oyie runs the final seeds. Upstreaming the adapter, a
full source-specific variational reference, and a tied-slope CFFG mode remain
optional future work.

## Definitions

- **Gibbs sampler:** A method that repeatedly samples each unknown from its
  conditional distribution given the current values of the other unknowns.
- **Independent direct oracle:** A small calculation that shares no production
  helper with the adapter surface it checks.
- **Paper-equation reference adapter:** An independent implementation of the
  published model equations built on pinned reference code but not shipped by
  the original authors.
- **Pólya–Gamma auxiliary:** A latent variable that makes a logistic
  likelihood conditionally Gaussian in its coefficients.
- **Source regime:** The discrete regime `z_t` that conditions the transition
  to `z_(t+1)`.
- **Source-specific slope:** A continuous-state coefficient selected by the
  source regime instead of shared across every source.

## Notes

**2026-08-10T03:20:00Z**

Wiggle-audit retained candidates and intended meanings: preserve=retain CFFG's
already delivered expressive family; same=one explicitly selected source row
or full-model comparison; implement=the separately versioned external adapter;
support=literal transition support or the explicitly excluded float32
capability; remaining=deferred optional work; sufficient=Pólya–Gamma natural
statistics; equivalent=the displayed algebraic parameter mapping;
exact=finite enumeration, literal zero, or named commands and commits;
reproduce=collapse to the forced upstream recurrence-only case;
sound=another method satisfying the outcomes; test and verify=the listed
analytic, enumeration, isolation, and collapse checks; fit and learn=the
observed-only Gibbs parameter sampler; infer=the prohibited CFFG implementation
reuse; small=the named compact finite oracle; may=optional method; all,
at_least, every, and only=the locally enumerated update surfaces, cardinalities,
source rows, statuses, and source-isolation boundary; current=the current Gibbs
conditional; existing=the pinned dependency's regression primitive. Capability
owner: cff-64ri owns all semantic changes to the independent source-specific
reference and no CFFG rule. Intelligible-ticket review: complete.

Classification-note additions: reusable=the CFFG implementation that the
adapter must not reuse; complete=the enumerated adapter update surfaces;
deliver=the prior CFFG source-specific capability; check and oracle=the named
analytic/collapse authorities; intended=the contextual assignments in this
note; explicit_owner and named_ticket_handoff=the cff-64ri/cff-0l31 boundary;
deferred=the optional future reference variants.

**2026-08-13T05:19:12Z**

PR merge-gate correction (2026-08-12): separate versioning is no longer treated as sufficient independence. The adapter must map every added surface to exact paper equations, reused legacy primitives, adapter code, and a direct oracle that shares neither the adapter nor legacy state-update helper.
