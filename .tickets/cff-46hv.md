---
id: cff-46hv
status: closed
deps: []
links: [cff-6pmw]
created: 2026-07-30T15:22:40Z
type: chore
priority: 3
assignee: coopwilliams
---
# Make ARCHITECTURE an as-built living document

## Plain English

ARCHITECTURE describes the system that exists today, so contributors can trust unqualified statements as current behavior. Future-facing material moves into a clearly labelled target-design section at the end. Dependency and file-format policy is made internally consistent with the implementation and Cooper’s decisions.

## Decisions

Keep Distrax/TFP unadopted. Canonical parameter blobs are .npy and safetensors; .npz is reserved for agent checkpoints because its ZIP container does not support ordinary array memory mapping. Treat ARCHITECTURE as a living as-built document, with proposed behavior cordoned off at the end.

## Acceptance Criteria

ARCHITECTURE identifies itself as a living as-built document. Every unqualified architectural claim before the final target-design section describes implemented behavior; future-only material appears under an explicit final Target design heading. The stale whole-loop lax.scan and per-instance RNG claims are corrected or moved. Distrax/TFP policy is consistently unadopted. Parameter blobs are documented as .npy/safetensors and .npz as checkpoint-only. No new Markdown file or behavior change; git diff --check and tk dep cycle pass.


## Notes

**2026-07-30T15:29:39Z**

Final handoff (2026-07-30): converted ARCHITECTURE from a proposal/current-state mixture into a living as-built record with one explicit final Target design section. Corrected current IR, serialization, compiler, rule-dispatch, Colony, resource-estimator, visualization, and dependency claims against the repository. Recorded .npy and safetensors as parameter-blob formats, .cffgz as .npy-backed model bundles, and .npz as RollingAgent checkpoint-only because ZIP members are not ordinarily memory-mapped. Kept Distrax/TFP unadopted and removed the conflicting recommendation. README now matches the format and Colony execution contracts. Change path: documentation-only; executable duplicate surfaces N/A. Validation: 26 concrete source/test anchors exist, Markdown fences balance, hygiene/ticket-dashboard tests 4 passed, git diff --check and tk dep cycle passed. No full suite per the documentation-only session policy. No gotchas.
