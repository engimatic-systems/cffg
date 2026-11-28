---
id: cff-b2fz
status: closed
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

## Notes

**2026-11-28T04:18:18Z**

Implementation handoff (New IR field row). Scope the plate key and its shared owner namespace. Keep fragment names, wiring, factor sources and per-copy blob paths local. Ground the parent hub as scoped-plate/hub, tied blobs and learners as scope/name, and private owners as scoped-plate/index/name. Check direct and generated cross-operand collisions before returning. Exercise JSON/hash, grounding, compiler learning descriptors and schema-2 bundle rewriting against flat-first composition. Keep the existing validation error for nested plate declarations. Local rules, JIT/host/batch execution, numerical guards, checkpoint format and edit operations receive the same grounded declarations and are N/A for implementation. Explicitly check repeated and multi-segment scopes, cross-operand conflicts, zero numeric parameter change, and a 64-copy case.

**2026-11-28T04:27:24Z**

Implementation complete. Compact scoping records shared_scope while leaving all fragment-local references intact. Disjoint merge checks declared and tied names without expanding; an overlapping plate prefix can require temporary grounding to verify exact collisions. The million-copy regression intercepts expansion. Declared host port bonds preserve compact plates. Generated copy and Broadcast ports require explicit expansion first; nested declarations remain under cff-6s31. Schema-1.1 JSON identifies the new shared scope, old schema-1 bytes and hashes are preserved, and the actual a774a54 parser rejected 1.1 before decoding. Schema-2 bundles preserve logical targets, sources, exact posterior/prior arrays and the compact model. Final focused uv pytest gate covered test_compose, test_serialize, scoped learned bundle and 64-owner checkpoints, HMM chain, frozen Broadcast JSON/bundle and nested guard: 40 passed in 2.21 seconds. git diff --check and tk dep cycle clean. Parent owns the required aggregate short-suite gate and PR integration.
