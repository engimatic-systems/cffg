---
id: cff-gvch
status: open
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
2. State the policy for identical repeated declarations. Preserve them if
   harmless repetition is supported; otherwise document and test the rejection.
   Do not rely on declaration order to resolve differences.
3. Preserve supported single-declaration behavior and its existing diagnostics.
   Tests cover both orders of conflicting roles, conflicting evaluator details,
   identical repetitions under the stated policy, and serialized reconstruction.
4. Before implementation, name the validation owner in the New IR field and
   New factor or rule family architecture rows. Core ModelError versus compiler
   CompileError and aggregation order are deliberate choices, recorded in this
   ticket and tested. Update cff-kfo7's compatibility baseline after the fix.
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

These suggestions are optional. Collect declarations per target, compare their
semantic fields, then construct the unique-target lookup only after validation.
Keep declaration validation separate from later ownership claims so discarded
metadata cannot become a hidden second interpretation of a factor.

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
