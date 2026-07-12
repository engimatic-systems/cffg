---
id: cff-afo0
status: closed
deps: []
links: []
created: 2026-07-12T15:10:41Z
type: chore
priority: 1
assignee: coopwilliams
external-ref: gh-21
---
# Repair stacked PR ancestry after squash merges

Back up the affected refs, reconstruct ef-engine through cff-j4mu-inference-status onto current main without changing their source trees, update the existing PR branches with force-with-lease, verify GitHub diffs/mergeability, and close redundant PR #20.

## Design

Use dated remote backup refs before mutation. Rebase each layer with exact old-parent boundaries. Require source-tree equality at every layer and full-suite success before updating live refs. Do not modify main.

## Acceptance Criteria

Backup refs exist remotely; PR #21 and PRs #10-#17 have correct stacked ancestry and intended diffs; source trees match their original tips except recovery ticket metadata; full test suite passes; PR #20 is closed as redundant; main is unchanged.


## Notes

**2026-07-12T15:11:36Z**

Pre-repair refs recorded and published as tags under recovery-backup-20260712/pre-stack-repair/: main=1072f6b77f4c2459d8a11bb3c3613b53fb2eaf75 ef-engine=af8d9f9c36484dbfe99a2bfbd7038fc781c2c4c4 hybrid=e35b4599bde440ee433f8bfb9fca218c19f12500 gauss-vmp=19770c6638d6040ffacb63d3e2f69803a391c80b plates=c473a0df4f25bcfad1b9a26c133886d23b993cc5 constraints=2d2aa5b4d7908f6078f16b14ca46754206209c26 eval-notes=9471f06c6fe87ac5d80e4f32a152c2e354c50443 adversarial-review=99bf5be80cb41262ad25041672ec211ed87faed3 cff-hp3y-verified-repair-1=eb9fd1cfc69ce3dc5cad2d31dece32b3a24f5117 cff-j4mu-inference-status=e9b0ee5bbd126a4b4eca1dccbc87d5e25ed9ca4e cff-vamy-bundle-plates-pr18-head=c8c3084c8d2cf70302e39346aa61bb3fad4df1d9 exp-families-deleted-tip=6f368c02f6ee32b949577cbe08f624e4fde63481

**2026-07-12T15:13:36Z**

Recovery completed. Published rollback tags under recovery-backup-20260712/pre-stack-repair/. Rebased nine layers onto main 1072f6b without conflicts: ef-engine 6 commits, hybrid 6, gauss-vmp 4, plates 2, constraints 2, eval-notes 2, adversarial-review 1, cff-hp3y-verified-repair-1 1, cff-j4mu-inference-status 4. Every reconstructed source tree matched its original tip exactly. uv run pytest on final reconstructed tip: 1115 passed, 4 dependency deprecation warnings. Updated all nine live refs atomically with force-with-lease; GitHub reports PR #21 and #10-#17 MERGEABLE/CLEAN with intended diffs. Closed redundant PR #20. main remains 1072f6b. Review/merge order: #21, then #10 through #17; use Create a merge commit and retarget each next PR to main before merging.
