---
id: cff-n183
status: in_progress
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
promise not to change supported combinations. Cooper approved fixing sharing
before continuing extraction. Implement full cross-group sharing; do not
temporarily reject these otherwise supported models.

## Outcomes / Acceptance Criteria — WHAT

1. One logical prior owner pools contributions across all compatible groups
   referencing the shared prior blob, updates every associated parameter edge
   consistently, and charges its parameter free-energy contribution once.
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
   32 contributing composites include multiple shape groups. An outcome absent
   in one group's support may still receive another group's positive count.
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
No further learned-goal capability is delegated by this fix.

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

**2026-09-09T19:34:06Z**

Cooper approved full sharing on 2026-09-09: “let’s not delay fixing it.” Base agent/cff-8t4d-topology updated from origin at e4a6640; work branch agent/cff-n183-shared-goals. Before implementation: applicable architecture rows New factor/rule (GFE ownership), Runtime numerical/status (sweep, objective, trust chart), Learnable/checkpoint (rollover and restore), New IR field only for derived compiled metadata; no serialized field changes. Flooding/mixed share _make_step; exact BP-only execution is N/A to GFE. Host/JIT and direct/vmap need parity. Trust-region trial charts must preserve tied parameter copies, not merely completed sweeps. JSON/bundle recompile reconstructs ownership; current and supported-legacy agent envelopes already restore per-blob goal priors through with_goal_priors, with unchanged schema. Sparse exact-zero counts, improper/nonpositive priors, f32 and 32 mixed-group contributors are acceptance cases. Keep unsupported learned-A plus learned-goal/mu3 combinations rejected.

**2026-09-09T19:34:06Z**

Research: correct update ordering? Formalism review confirms all local predictions use pre-sweep goal counts, then sum each new prediction once globally and replace every carrier before damping. BFE recomputes its own current-state predictions and retains the off-coordinate (alpha-alpha0-sum predictions) expected-log term; do not replace it with stored counts or a fixed-point-only beta ratio. Source: engine _make_step/_make_bfe and DERIVATIONS D6c; confidence high. Integration review: global immutable GoalParameterPlan must own grouping, pooling and parameter energy independently of GfePlan buckets; local metadata stays a compatibility view. OuterChart currently gives aliases independent coordinates; constrain them to a single vector copied to all aliases during trial reconstruction. Include aliases/global plans in cache/schedule signatures. Sources: outer.py:224, engine.py:1901,2767,3804, schedule.py:168, _agent_checkpoint.py:88,548,610; confidence high. No separate GFE scan or learned-c mu3 implementation exists.
