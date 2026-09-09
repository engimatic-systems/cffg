---
id: cff-gvch
status: in_progress
deps: []
links: [cff-kfo7]
created: 2026-09-09T18:17:11Z
type: bug
priority: 1
assignee: coopwilliams
tags: [correctness, compiler, validation]
---
# Reject conflicting P-substitution declarations before dictionary collapse

## Background in Plain English — WHY

P-substitution changes which variable a factor predicts and which objective it
contributes. The compiler at 8cd8919 builds a dictionary keyed by factor name
before checking this meaning. When a template declares two different
P-substitutions for one factor, the last declaration silently replaces the first.

Reproduction: in tests/test_mu3.py's _model(), insert a P_SUBSTITUTION
constraint for obs0 with substituted_role="in" before its existing "out"
declaration. Core validation and compilation succeed: the unsupported earlier
declaration disappears. Reversing declarations must not choose the model's
meaning. Specialized discovery in cff-kfo7 needs reliable original declarations,
but rejecting this currently accepted input is a correctness change, not a
mechanical extraction.

## Outcomes / Acceptance Criteria — WHAT

1. Inspect all P-substitution declarations before any dictionary keyed by
   target discards duplicates. Conflicting declarations fail with the affected
   factor and conflicting roles or evaluator details identified.
   1.1. Inspect repeated keys within each raw P-substitution detail tuple too;
        differing values must not disappear in detail_dict or serialization.
2. Preserve identical repeated declarations and identical repeated detail
   key/value pairs, including metadata-key order differences. Do not remove
   repeated declarations from the template or change their serialization/hash.
   Compare all metadata, including unknown nested JSON values; missing keys
   remain distinct from explicitly supplied values. Declaration order never
   resolves a conflict.
3. Preserve supported single-declaration behavior and its existing diagnostics.
   Tests cover both orders of conflicting roles, conflicting evaluator details,
   identical repetitions under the stated policy, and serialized reconstruction.
4. Core constraint validation owns ModelError for conflicts. Preserve existing
   per-declaration target/role diagnostics, then append one conflict report per
   target in sorted target order. Reports identify differing keys and values
   deterministically. Serialization rejects these same conflicts with ValueError
   before its lossy detail conversion, without adding full-model validation to
   serialization. Update cff-kfo7's compatibility baseline after the fix.
5. Host/JIT, flooding/scheduled and direct/vmap runtime execution are N/A:
   conflict rejection is host-side declaration validation before execution.
   JSON/bundle reconstruction reaches the same validator. Check current and
   supported-legacy restore paths and state whether they reconstruct templates.
6. Sparse/zero-support and improper numerical messages are N/A to this
   declaration-only check. Float32 cannot change the decision. A template with
   at least 32 declarations checks deterministic reporting without repeated
   whole-template scans per factor.
7. Focused validation/compiler/serialization regressions and one final short
   suite pass. No numerical update or objective implementation changes.

## Methodological Suggestions — HOW

These suggestions are optional. Use one P-substitution-specific inspection
helper shared by core validation and serialization. Collect declarations per
target before converting raw details to mappings, and compare canonical JSON
values only where repetitions need comparison. Keep declaration validation
separate from later ownership claims so discarded metadata cannot become a
hidden second interpretation of a factor. Do not change other constraint
families or the JSON parser's general duplicate-object-key policy.

## Future work

cff-kfo7 moves the validated declaration meaning into explicit specialized
discovery. This ticket does not broaden supported substituted roles or add an
evaluator family.

## Definitions

- **P-substitution:** A declared change to a factor's variational objective in
  which its predictive distribution substitutes for a selected marginal.
- **Evaluator details:** Metadata selecting the supported method used to
  evaluate that objective and its messages.
- **Dictionary collapse:** Retaining one value per key while silently replacing
  an earlier value with the same key.

## Notes

Research question: can conflicting claims disappear before specialized
discovery? Answer: yes, the current target-keyed lookup retains only the last
P-substitution. Sources at 8cd8919: compiler.py:161-165;
core/validate.py:452 onward; tests/test_mu3.py model fixture. Confidence: high,
bounded formalism-agent inspection and compile reproducer. No fix implemented.

**2026-09-09T19:50:49Z**

Preparation during cff-n183 final gate; no validation implementation started. Question: where should conflicts be detected? Answer: core.validate._check_constraints before compiler psub dictionary collapse, preserving existing per-declaration target/role diagnostics and appending deterministic per-target conflict diagnostics. Preserve identical repeats, including metadata-key order changes, without rewriting template constraints or hashes. Compare complete metadata, including unknown/nested JSON values; missing keys are distinct from explicit values. Sources: core/validate.py:452, compiler.py:165, core/ir.py:233, core/serialize.py:108,111,314 at cdbad68; confidence high from bounded modularity-agent review. Additional loss point in the same P-substitution declarations: repeated raw detail keys are collapsed by both detail_dict and serialization dict conversion. Before implementation, explicitly settle raw-key conflicts (recommended: reject differing repeats, retain identical repeats) and protect that serializer conversion without invoking full-template validation. Canonical serializer sorting can also TypeError on conflicting nested mappings; do not depend on sorting for validation. JSON and flat current/legacy bundles normally defer whole-model validation until compile; plate fragments already validate; agent checkpoint restores do not reconstruct templates and need no format change. Test locations: test_ir constraints, test_serialize, existing current/legacy-flat bundle paths, and test_mu3 model for the swallowed unsupported in-role declaration. Check 32 declarations and deterministic reversed-order conflicts. Keep scope P-substitution only, not a general constraint rewrite.

**2026-09-09T20:05:42Z**

Starting from updated agent/cff-n183-shared-goals at a9e28ce (PR #116); branch agent/cff-gvch-psub-validation. Applicable architecture rows: New IR field and New factor/rule for declaration interpretation and serialized source truth. No new IR field or schema. Runtime numerical/status and learning/checkpoint implementation N/A. Host validation runs before numerical JIT, flood/mixed/direct/vmap all share compile; no separate runtime change. Core validation retains existing target/role ordering and adds deterministic conflict diagnostics afterward. Serializer invokes only the P-substitution conflict guard, not full-template validation. Identical repeats remain valid and unmodified; conflicting raw detail-key repeats are included because the same dict conversion otherwise erases them. Scalar/string metadata and nested JSON metadata retain meaning; a missing key differs from an explicit value. General JSON duplicate-object-key policy and other constraint kinds are outside scope. JSON/bundles recompile the same declarations; current/legacy agent restores do not reconstruct templates. Sparse/improper numerical messages N/A, float32 structurally irrelevant, 32 declarations test deterministic diagnostics without repeated whole-template scans.
