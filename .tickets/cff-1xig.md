---
id: cff-1xig
status: open
deps: [cff-jilz]
links: []
created: 2026-09-13T15:23:02Z
type: task
priority: 2
assignee: coopwilliams
tags: [serialization, bundle, compatibility]
---
# Define the stable CFFGZ compatibility and migration contract

## Background in Plain English — WHY

Cooper clarified on 2026-09-13 that CFFGZ may break compatibility during alpha, but the stable format must preserve older supported bundles. The schema-2 learning archive in cff-enhs and its workflow proof in cff-jilz should not acquire an accidental permanent compatibility promise.

## Outcomes / Acceptance Criteria — WHAT

1. Before the format is declared stable, the documentation identifies the supported historical schemas and the version boundaries that require migration or permit refusal.
2. A migration preserves declared learning identities, immutable priors, posterior values, point arrays, ownership, shapes, and dtypes when the older format contains those facts.
3. A format missing learning provenance is never silently promoted to resumable state; its permitted capabilities and refusal rules are explicit.
4. Immutable representative fixtures prove each promised compatibility path and rejection boundary through the public loader.
5. README and ARCHITECTURE describe versioning and migration ownership; LIMITATIONS states unsupported historical cases.

## Methodological Suggestions — HOW

The implementer may choose another sound method. Use the Learnable or checkpoint feature change path. Inventory released schemas before promising compatibility, retain fixtures from their original producers, and test migrations against named array state rather than archive bytes. Decide the stable support window before implementing migrations. Alpha schema 1 support is not required by this ticket until that policy is decided.

## Future work

This ticket follows cff-jilz. It does not block alpha schema-2 storage or resume proof. Future schema-specific migrations may be split into dependent implementation tickets once the support policy exists.

## Definitions

- Compatibility contract: the documented promise about which older archives a release can load and which information it preserves.
- Migration: a validated translation from an older serialized representation to the current one.
- Learning provenance: evidence identifying the update semantics and immutable priors that produced a posterior.


## Notes

**2026-09-13T15:24:35Z**

Dependency companion (prerequisite --> dependent), checked against tk dep tree --full, tk ready, and tk blocked:

cff-enhs (ready, in progress)
  --> cff-jilz (blocked by archive mechanics)
        +--> cff-cz6v (current learning-resume epic)
        +--> cff-1xig (future stable compatibility policy)

The policy and epic completion are parallel downstream work; the policy does not block either alpha PR. tk dep cycle reports no cycles.
