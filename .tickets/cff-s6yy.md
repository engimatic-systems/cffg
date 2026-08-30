---
id: cff-s6yy
status: closed
deps: [cff-rl7r]
links: []
created: 2026-08-30T16:00:47Z
type: feature
priority: 1
assignee: coopwilliams
parent: cff-0qfy
tags: [consolidation, recurrence, inference]
---
# Port structured recurrence production semantics

## Background in Plain English — WHY

Five selected commits from PR #92 replace the recurrence rule's independent
source/destination approximation with a structured consecutive-state pair
belief. They add the derivation, rule authority, engine integration, learning
readout, and checkpoint version boundary. PR #92 also contains experiment
preregistration, audit setup, and old ticket transitions that must not enter the
new stack.

The applicable architecture change-path rows are **New factor or rule family**,
**Runtime numerical or status invariant**, and **Learnable or checkpoint
feature**. The serialized factor family remains `recurrence_link`; D9 and the
structured recurrence module own its pair rule, the engine owns execution and
the BFE ledger, and recurrence checkpoints own the semantic version boundary.

## Outcomes / Acceptance Criteria — WHAT

1. The retained behavior from commits `7c8ad8fb`, `e70e1b48`, `605136ad`,
   `98070d8e`, and `e38f29d8` is present in that order relative to cff-rl7r.
2. `docs/DERIVATIONS.md` contains the structured recurrence derivation and its
   plain-English companion with executable test anchors.
3. The rule exposes bounded pair potentials, log-space categorical messages,
   a normalized source/destination pair belief, a pair-derived Gaussian
   message, and the matching node-energy term.
4. Compiler, synchronous, scheduled, mixed, and BFE paths use the structured
   rule. Fixed-drive K=2 and K=3 chains match enumeration and `-log Z` within
   dtype-scaled tolerances.
5. Public recurrence and Gaussian readouts and Pólya–Gamma learning consume the
   retained pair belief. No production path silently reconstructs a singleton
   outer product.
6. Literal log-zero support remains exact. Empty support, improper Gaussian
   moments, non-finite potentials or beliefs, invalid shapes, and materially
   indefinite statistics fail loudly. Float32 remains finite, and the retained
   K=8, T=100 case avoids cubic transition storage.
7. Converged, budget-exhausted, and diverged remain distinct in host and JIT
   checks. P-substitution on recurrence remains a compile-time rejection.
8. Structured recurrence checkpoints identify D9 semantics, round-trip current
   values and ownership, reject incompatible D8 learned state by default, and
   validate version, factor, shape, dtype, and finiteness before mutation.
9. Duplicated surfaces agree: host and JIT support/status checks; flooding and
   scheduled messages, beliefs, and BFE; single-instance and `vmap` execution;
   current structured checkpoints and explicit legacy rejection.
10. `README.md`, `docs/ARCHITECTURE.md`, `docs/LIMITATIONS.md`, and
    `docs/DERIVATIONS.md` are reconciled with `main` commit `d805135` without restoring
    superseded claims.
11. Old ticket files and transitions, preregistration commits, audit notes,
    review scaffolding, and old generated runtime metadata from PR #92 are
    absent. Pure oracle helpers under `src/cffg/experiments` are retained only
    when production tests import them.

## Methodological Suggestions — HOW

These suggestions are optional. The implementer may choose another sound
method.

- Port the five selected commits as separate atomic commits so derivation,
  rule, integration, learning, and persistence remain independently reviewable.
- Apply each commit without committing, discard excluded paths, and resolve
  canonical documents in place.
- Run the focused structured-recurrence rule, engine, learning, and checkpoint
  tests after their corresponding commits.

## Future work

cff-4jk1 depends on this ticket and will port the installed-parameter outer
convergence evidence. New scientific runs and recovery experiments remain
outside this consolidation.

## Definitions

- **BFE:** Bethe free energy, the factor-local energy and edge-entropy objective
  used for convergence and scoring.
- **D8 / D9:** The older singleton-product recurrence derivation and the newer
  structured pair-belief derivation in `docs/DERIVATIONS.md`.
- **Pair belief:** The joint probability of consecutive source and destination
  regimes at one recurrence factor.
- **P-substitution:** The LAIF constraint that replaces a selected variational
  marginal with the generative conditional in a GFE node.

## Notes

**2026-08-30T16:28:15Z**

Ported 7c8ad8fb, e70e1b48, 605136ad, 98070d8e, and e38f29d8 as five new atomic commits on cff-rl7r. Added the prerequisite pure recurrence_smoothing oracle and test because the selected commits import and modify them. Kept only production/oracle code, tests, D9 derivation, canonical docs, and EPIC_FILES registrations. Excluded tools/test-suite-runtimes.json, old ticket changes, preregistration, audit notes, and final-parity experiment prose. Focused validation by commit: 16 passed; 37 passed; 64 passed; 82 passed and 1 deselected; final checkpoint plus rSLDS fit 24 passed after correcting the reordered snapshot dependency.
