---
id: cff-1xig
status: in_progress
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

**2026-11-28T04:18:18Z**

Change path: Learnable or checkpoint feature, limited to a source and fixture audit plus alpha documentation. Compare original and current public bundle readers, point arrays and complete learning inventories. Checkpoint decode has an independent contract. No IR field, arena layout, resolver, state/readout, message rule, host/JIT, flooding/scheduled or single/vmap change is proposed. The ticket remains in progress until Cooper chooses a stable support floor.

**2026-11-28T04:20:45Z**

Historical inventory at a774a54 (2026-11-27)

- The original writer, 8644e59, wrote manifest schema "1", model JSON, point arrays at fixed paths, and per-blob hashes. Commit 39ae83b added explicit member paths and support for plates under the same version. Neither writer stored priors or posteriors. The recurrence archive tests/fixtures/rslds-base-4d790231.cffgz came from the second layout. Its SHA-256 is c0adf08883f4e6e283dce3af8ac890d51049b15211afbb58400f6bc42e0fe58f. It contains four learning declarations and eleven point arrays. Declaring a learner does not identify the evidence it accumulated.
- Commit 97824a2 briefly allowed verified schema-1 recurrence points for inference with resumable=False. Commit 1234380 introduced manifest schema "2" and required complete learning records. Cooper's policy clarification in 950a115 explicitly removed the older import requirement. Model JSON still uses its independent schema "1". GitHub has no releases and origin has no version tags at the audit date. The package metadata says 0.1.0. None of these sources designates a stable support window.
- The checked-in recurrence ZIP is an immutable producer fixture. The flat refusal test constructs its input by editing a new ZIP. The representative schema-2 rSLDS workflow generates a fresh ZIP during testing. No schema-2 ZIP is pinned in this Git tree. A historical test that keeps passing on newly saved bytes alone cannot establish that future versions read these particular old bytes.

Original-producer checks. Ran the old writers from git archive of their exact source revisions in the existing uv environment. At 8644e59, used build_gridworld(k=5), save_bundle, and that revision's load_bundle. At 1234380, used the revision's test_bundle_learning._learner in float32 and float64 and its public save/load functions. The latter two fixtures have two learnable owners with different prior and posterior values. Four more schema-2 archives came from an actual learning update at fd8bfd1. Verified each input against agent-share/cffg-6wpk/inventory.json.

At a774a54, public load_bundle refused both original schema-1 layouts with unsupported schema '1'. It read all six schema-2 archives. All 48 point arrays and 80 prior/posterior arrays matched the NPY members exactly in shape, dtype, and value bytes. The restored learning descriptors matched every archived field, and the restored priors were read-only. The examples cover NIW, MNIW and component-MNIW learning. This checks restoration of these files; it does not certify every schema-2 producer, a numerical next update, or an executing checkpoint. The six schema-2 results and the two refusals are recorded with input digests at ~/agent-share/cffg-stable-bundle-compatibility/result.json. The same directory retains the three newly generated original-writer archives. The four earlier learned archives remain with their producer inventory.

Cooper's decision before adopting any stable contract:

1. Choose the first guaranteed producer. Start with bundles written by the first stable release, or include schema-2 alpha archives back to a named producer such as 1234380. The latter requires immutable supported fixtures for the historical template features and learning identities included in that choice.
2. Decide separately whether to provide an inference-only import for either schema-1 layout. Scope it by flat vs explicit paths, plate support and permitted model semantics. Even a verified point model needs fresh, explicit priors to start a new learning history. It cannot resume the original one.
3. Name the stable release boundary and how long its supported versions remain readable. Keep version refusal explicit for unsupported archive, model or learning identities. Archive migration must verify the original inventories and preserve the supported named data and its owners.

The README matrix describes current alpha behavior. The loader and stable support policy are unchanged. Retain this ticket in progress for Cooper's decision. The five focused cases for authentic recurrence refusal, flat refusal, complete f32/f64 restore and empty learning inventory passed in 1.28 s. No full-suite run applies to this documentation and ticket change.

**2026-11-28T05:17:06Z**

Cooper chose the first stable writer as the guaranteed floor on 2026-11-27. Every alpha import, including schema 2, is best effort. The current schema-1 refusals stand. Prior/ownership history cannot be reconstructed from point arrays. README, ARCHITECTURE and LIMITATIONS now record this policy and the bundle loader owns migration. The historical audit above remains evidence for those eight particular inputs. The remaining gate is release-bound: identify the actual first stable producer, archive immutable fixtures from it and verify their model and learning inventories through the public loader. No stable writer exists in this tree yet; no alpha producer is promoted. No further floor decision is pending.

**2026-11-28T05:17:52Z**

Post-decision validation: 5 README checks passed, tk dep cycle reports no cycles, and git diff --check passes. Documentation and ticket changes only.
