---
id: c1-a2cb
status: closed
deps: []
links: []
created: 2026-08-18T20:23:12Z
type: task
priority: 1
assignee: coopwilliams
parent: cff-uy48
tags: [paper-fidelity, compiler, safety]
---
# Reject reserved moment-matching beads before compilation

## Background in Plain English — WHY

The template schema reserves `ConstraintKind.MOMENT_MATCHING` for future expectation-propagation rules. The compiler currently accepts that bead and then ignores it. A serialized model can therefore claim a constraint that execution does not enforce. The paper-faithfulness release gate requires a loud rejection until a registered rule owns the message and objective terms.

## Outcomes / Acceptance Criteria — WHAT

1. Every compilation path rejects a `MOMENT_MATCHING` constraint with a specific `CompileError` before arena construction.
2. The error names the target factor and states that no executable moment-matching rule is registered.
3. JSON and bundle serialization continue to round-trip the reserved syntax so a future rule can activate it without a schema migration.
4. Direct tests prove that a round-tripped bead is rejected and cannot produce messages or a free-energy ledger.
5. The rejection is independent of numeric values, dtype, incoming-message support, and model scale because it occurs before runtime inputs exist.
6. `docs/ARCHITECTURE.md`, `docs/LIMITATIONS.md`, and `README.md` classify moment matching as syntax only with compile-time rejection.

## Methodological Suggestions — HOW

These suggestions are optional. An implementer may choose another sound method that satisfies the numbered outcomes with equivalent evidence.

- Add one structural guard beside the compiler constraint classification, after template validation and before factor grouping.
- Reuse the existing serialized enum value; do not add an approximate rule.
- Exercise both an ordinary template and a serialized round trip in the focused compiler or IR tests.

## Future work

A future expectation-propagation ticket may register executable moment-matching rules, message projections, and objective semantics. That work must replace this rejection with factor-specific capability checks; it is not part of this ticket.

## Definitions

- **Expectation propagation:** An approximate inference method that matches selected moments rather than complete marginal densities.
- **Moment-matching bead:** A CFFG constraint that replaces exact marginal equality with equality of selected sufficient-statistic expectations.
- **Reserved syntax:** A serialized construct retained for future compatibility but rejected by the current executor.


## Notes

**2026-08-18T20:23:20Z**

Planning handoff, 2026-08-18. Applicable change path: New IR field guard on an existing reserved enum. Authority stays in src/cffg/core/ir.py; compiler capability rejection changes in src/cffg/compiler.py. Validation, serialization/hash, plates, compose/edits, arena, bundle, and agent persistence are checked and remain N/A except the existing serialization round trip. Duplicated execution pairs host/JIT, flooding/scheduled, single-instance/vmap, and current/legacy checkpoint are N/A because rejection precedes arena construction. Hostile-input checklist: sparse or zero support N/A; improper or vacuous messages N/A; float32 N/A; and 10–100× fixture scale N/A. All are structure-only rejection before numeric inputs.

**2026-08-18T20:25:24Z**

Completed 2026-08-18. compile_template now rejects every reserved MOMENT_MATCHING bead before arena construction and names the factor. JSON round-trip remains intact and the focused test tests/test_serialize.py passed (9 tests). README, ARCHITECTURE, and LIMITATIONS state the syntax-only boundary. No runtime execution pair applies because no arena is built.
