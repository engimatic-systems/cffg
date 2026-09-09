---
id: cff-n183
status: open
deps: []
links: [cff-kfo7]
created: 2026-09-09T18:17:11Z
type: bug
priority: 1
assignee: coopwilliams
tags: [correctness, gfe, architecture]
---
# Resolve learned-goal ownership across GFE shape groups

## Background in Plain English — WHY

The learned-goal contract already pools predicted observations when several
GFE composites reference the same Dirichlet prior blob. Tests exercise this
within one shape group. The compiler currently builds those ties separately
inside each GfePlan bucket, so sharing breaks when state dimensions or other
grouping flags differ.

Reproduction at 8cd8919: start with test_goal_hyperprior._model(slices=2);
give the second state cardinality 3 and its own correctly shaped A1/d1 blobs,
but retain common Cc. The two groups are gfeC:3x2 and gfeC:3x3, each with a
singleton goal tie. One step yields different proper goal posteriors for Cc;
goal_prior_updates then raises an inconsistent-posteriors error. Runtime count
pooling and goal_dirichlet_energy both loop over group-local ties, so there is
no global shared-parameter owner.

This is pre-existing behavior, not introduced by topology extraction. It
conflicts with cff-kfo7's requirement for one shared-parameter owner and its
promise not to change supported combinations. Cooper has been asked whether
to implement correct cross-group sharing first (recommended) or reject the
affected models explicitly until full support is implemented. No fix is
authorized or implemented by this planning ticket alone.

## Outcomes / Acceptance Criteria — WHAT

1. Record Cooper's scope decision before changing inference behavior:
   1.1. Full support: one logical prior owner pools contributions across all
        compatible groups referencing the shared prior blob, updates every
        associated parameter edge consistently, and charges its parameter
        free-energy contribution once.
   1.2. Temporary rejection: compilation identifies the shared blob and
        conflicting groups before returning an Arena; no apparently valid
        inference state is produced. Full support becomes linked follow-up work.
2. Preserve the existing within-group sharing contract and distinguish a shared
   latent goal from merely equal numeric prior values. Document the serialized
   identity that selects sharing; do not infer it by comparing arrays.
3. For full support, an independent mixed-state-cardinality example checks the
   pooled posterior alpha0 + sum(predicted observations), the single shared
   parameter term plus structural terms, all goal readouts, and trial rollover.
   A one-shape counterpart retains the existing analytic oracle.
4. Check affected architecture rows: New factor/rule, New IR field,
   Learnable/checkpoint and Runtime numerical/status. Cover flooding/scheduled,
   host/JIT, direct/vmap and current/supported-legacy restore consumers as
   applicable; explicitly mark any non-applicable pair before implementation.
5. Hostile inputs are acceptance cases, not only planning prose: exact-zero
   likelihood support stays exact; nonpositive/improper Dirichlet priors fail
   loudly; float32 uses existing dtype-aware guards and status checks; at least
   32 contributing composites include multiple shape groups. If rejection is
   chosen, these fixtures instead prove early rejection plus unchanged behavior
   for still-supported within-group sharing.
6. No Python host callback or repeated whole-arena rebuild is introduced inside
   numerical inference. Report any new per-step reduction/storage costs.
7. The selected policy is explicit in ARCHITECTURE and LIMITATIONS as needed.
   Formal changes update DERIVATIONS with a plain-English companion and test
   anchors. Focused oracle/integration tests and one final short suite pass.
8. Update cff-kfo7's compatibility baseline and handoff to name this behavior
   decision; extraction must not silently decide it again.

## Methodological Suggestions — HOW

These suggestions are optional. Model global goal ownership independently of
GfePlan shape buckets. A group's local observation/state shape does not determine
whether the goal parameter is shared. Separate structural node terms from the
shared parameter term when checking the objective.

## Future work

cff-kfo7 extracts specialized discovery after this ownership policy is resolved.
If temporary rejection is selected, implement full cross-group sharing in a
linked follow-up rather than representing it as completed support.

## Definitions

- **GFE:** Generalised free energy, the objective used by P-substituted nodes.
- **Shape group:** Factors batched together because their array dimensions and
  execution flags match.
- **Dirichlet prior:** Positive concentration parameters describing uncertainty
  over a categorical probability vector.
- **Shared owner:** The one place responsible for a logical parameter's pooled
  update and parameter contribution to the objective.

## Notes

Research question: can the existing learned-goal sharing contract span shape
groups? Answer: no; the compiler creates independent ties, runtime updates and
the ledger use those local ties, and rollover detects inconsistent results.
Sources at 8cd8919: compiler.py:608-645; engine.py:1900-1925,2821-2842,
3807-3833; tests/test_goal_hyperprior.py:103-166,245-252. Confidence: high,
bounded formalism-agent inspection plus a primary-agent compile/step/readout
reproducer. No engine code changed during this investigation.
