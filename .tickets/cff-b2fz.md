---
id: cff-b2fz
status: open
deps: [cff-xabn]
links: []
created: 2026-09-13T00:02:42Z
type: feature
priority: 2
assignee: coopwilliams
---
# Preserve compact plates through template scoping and merging

## Background in Plain English — WHY

The cff-xabn persistence audit found that core.compose.scoped and merge discarded compact plates, including declared learnables inside them. cff-xabn adds a rejection that names affected learnables and directs callers to expand_plates first. That prevents silent loss but leaves compact composition unavailable.

## Outcomes / Acceptance Criteria — WHAT

1. Scoping and merging preserve compact plate declarations, wiring, tied and per-copy blob ownership, and learning identities.
2. Name collisions fail before any input template changes.
3. Expanded results agree with composing the already-expanded operands, including learned parameter sources and targets.
4. JSON, structure hashing, and bundles preserve the composed compact form.

## Methodological Suggestions — HOW

The implementer may choose another sound method. Use the New IR field change-path row. Specify how a scope prefixes plate names, parent hubs, fragment names, and shared blob owners before implementation. Test tied and per-copy learnables and scoped names with multiple segments.

## Future work

This follows cff-xabn and removes its expand-before-composition limitation. No other future work is identified.

## Definitions

A compact plate stores one fragment plus copy dimensions and wiring rules. Expansion materializes the individual copies. A tied blob is shared by copies; a per-copy blob is owned separately by each copy.
