---
id: cff-aub0
status: closed
deps: [cff-qu2j]
links: []
created: 2026-10-01T15:27:09Z
type: chore
priority: 2
assignee: coopwilliams
tags: [pcgym, validation]
---
# Refresh the PCGym library checkout for targeted PR-stack validation

## Background in Plain English

Cooper wants the PCGym experiment runner to validate the current CFFG PR stack. The experiment workspace still contains an older library checkout. Replace only that nested library checkout with a clean copy of the branch Cooper requested, while preserving the existing experiment scripts, data, results and environment.

Requested source: branch cff-9raw-pr-chain-guide at ead32f14c54511ccf818b2e911796f973198773b. Library target: /Users/ravd/experiments/cffg-testing/pc-gym/00/cffg/cffg. The parent directory is the experiment workspace, not the library repository.

## Outcomes / Acceptance Criteria

1. The library target contains a clean checkout at the exact requested source revision.
2. The previous checkout remains recoverable at a recorded backup location outside the experiment workspace.
3. Experiment scripts, results, feedback, environment and dependency configuration are unchanged.
4. The copied tracked files match the requested source revision.
5. The handoff states where the validator can install the library without claiming that validation has already run.

## Methodological Suggestions

Optional: create a fresh local Git clone in a temporary staging directory, verify its revision and tracked-file content, then move the old checkout to a backup before moving the new checkout into place. Avoid copying virtual environments, caches or unrelated untracked files into the new library copy. Another equally safe method is acceptable.

Architecture change paths and duplicated execution surfaces are N/A: this is an external checkout replacement, with no library implementation, serialization or test-selection changes. No library test suite is required. Validate checkout identity, file contents and preserved experiment configuration.

## Future work

The experiment runner will install from the refreshed workspace and perform targeted validation separately. This task does not run experiments or implement additional library features. No deferred implementation work is identified.

## Definitions

Checkout: files and Git history for one repository revision. Revision: a Git commit identifier naming an exact saved state. PR stack: the linked pull requests being reviewed together. Workspace: the experiment's project directory and local library dependency.


## Notes

**2026-10-01T15:27:15Z**

Before replacement: source branch cff-9raw-pr-chain-guide, exact HEAD ead32f14c54511ccf818b2e911796f973198773b; existing nested checkout is clean at fce60bd177ed8991f5919901728a13974e4055b6 on cff-ucvx-readme-capability-map. The enclosing pyproject.toml declares cffg as an editable uv workspace member at cffg, so preserve that relative path and do not replace the parent directory. Architecture change paths and duplicated execution pairs are N/A. No installation, targeted experiment or test-suite run is authorized by this staging task.

**2026-10-01T15:28:28Z**

Done: replaced only /Users/ravd/experiments/cffg-testing/pc-gym/00/cffg/cffg with a fresh independent Git clone of cff-9raw-pr-chain-guide at ead32f14c54511ccf818b2e911796f973198773b, the branch/revision present when Cooper requested the refresh. Verified the copied and requested Git trees both equal 34f84ed29ec5a6d02528375d7ba89f9be15ede2a, the installed-path checkout has no working-tree changes, and git diff --exit-code HEAD passes. The complete old checkout, including its local environment and caches, was moved rather than deleted: /Users/ravd/experiments/cffg-testing/pc-gym/00/cffg-refresh.8Nu08C/previous-cffg; its HEAD remains fce60bd177ed8991f5919901728a13974e4055b6. The fresh library copy does not inherit old virtual environments or caches. Parent pyproject.toml, uv.lock and FEEDBACK.md SHA-256 values match the pre-swap checks; parent experiment environment, scripts and results were not modified. The existing uv workspace source still resolves cffg to the same nested path. Next: the experiment runner can run uv sync --reinstall-package cffg from /Users/ravd/experiments/cffg-testing/pc-gym/00/cffg and carry out its targeted validation. This staging task ran no installation, library tests or experiments; no numerical validation success is claimed. No checkout-refresh work remains.
