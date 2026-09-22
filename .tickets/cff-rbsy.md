---
id: cff-rbsy
status: closed
deps: [cff-rtuf]
links: []
created: 2026-09-22T19:17:33Z
type: task
priority: 1
assignee: coopwilliams
parent: cff-3c6f
---
# Reconcile the limitations document with current capabilities

## Background in Plain English — WHY

The limitations document is unusually candid, but parts of its framing are stale. Its header still says the document was assessed before implementation even though it now contains many as-built updates. Its early message-rule section can also be read as saying moment matching is only a rejected placeholder, while endpoint-aware Gaussian moment matching and nonlinear Gaussian factors are now executable. Closed research chains and current open defects need to be distinguished from delivered capabilities.

A newcomer uses this document to decide whether an experiment result is trustworthy. The document must describe the current branch consistently and separate hard mathematical limits, current implementation limits, known defects, and work that was deliberately not delivered.

## Outcomes / Acceptance Criteria — WHAT

1. docs/LIMITATIONS.md has a current as-of statement for the proposed branch and no longer describes the whole document as pre-implementation.
2. The moment-matching text distinguishes rejected legacy factor-only declarations from supported endpoint-aware Gaussian moment matching.
3. The document agrees with current code and the capability audit for discrete GFE, Gaussian, exponential-family, hard-constraint, hybrid, recurrence, rSLDS, nonlinear, neural, learning, Colony, visualization, persistence, and estimation boundaries.
4. Administratively closed not-pursuing work is not described as a production capability.
5. Known open correctness defects that affect experiment inputs or interpretation remain explicit, including negative fixed GFE probabilities and malformed fixed mu3 particles.
6. Resource-estimator, Colony, plate, schema-migration, long-horizon, and zero-noise neural boundaries are stated only where they materially qualify experiment claims.
7. Historical measurements remain labeled with their setup and are not presented as universal limits.
8. The final one-paragraph summary matches the current capability envelope and points readers to the future README capability table.
9. Documentation validation and git diff --check pass; runtime tests are N/A because the ticket changes no behavior.

## Methodological Suggestions — HOW

These suggestions are optional. Another clear organization is acceptable.

- Preserve the hard-versus-soft distinction but add a short guide explaining known defect versus unsupported feature versus named approximation.
- Edit in place. Do not create a replacement document.
- Cross-check current factor declarations, public Engine readouts, closed-ticket completion notes, and open tickets. Do not infer delivery from closed status alone.
- Remove obsolete future tense when an item is implemented, while preserving the exact approximation or failure boundary.
- Applicable architecture change path: canonical limitation documentation only; no runtime row applies. Compiler/runtime, host/JIT, flooding/scheduled, single/vmap, and current/legacy pairs are N/A because no executable contract changes.

## Future work

- Fixes for the documented open defects stay in their existing tickets, including cff-34gi, cff-6n9c, cff-j7mm, cff-rqqc, cff-wbhu, cff-b2fz, cff-1xig, and cff-u06e.
- The downstream README ticket gives newcomers the compact capability view and links here for full boundaries.
- No new mathematical capability is part of this ticket.

## Definitions

- Administratively closed: A ticket was ended without delivering its proposed production capability.
- Hard limitation: A boundary inherent to the chosen mathematical framework.
- Known defect: Implemented behavior that accepts or reports something it should reject or classify differently.
- Legacy declaration: An older serialized form retained only for migration or explicit rejection.
- Named approximation: A documented method whose result has a narrower claim than exact inference or evidence.
- Soft limitation: A current implementation or engineering boundary that later work could remove.


## Notes

**2026-09-22T20:43:39Z**

Completed on cff-rbsy-limitations. Replaced the pre-implementation status with a 2026-09-22 implementation review; defined named approximation, unsupported feature, and known defect; reconciled executable endpoint-aware moment matching with rejected legacy payloads; explicitly marked the D12 production chain as not delivered after not-pursuing closures; and bounded EF, hybrid, recurrence, neural, persistence, compact plates, Colony, resource estimation, rendering, fixed GFE inputs, and fixed mu3 particles. The final summary now matches the implemented discrete/continuous/hybrid envelope and points newcomers back to the README capability table.

Validation: uv run pytest -q tests/test_hygiene.py (3 passed); tk dep cycle (none); git diff --check. Runtime tests are N/A because this branch changes canonical documentation and ticket state only.
