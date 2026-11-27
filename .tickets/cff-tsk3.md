---
id: cff-tsk3
status: in_progress
deps: []
links: []
created: 2026-11-27T04:31:05Z
type: feature
priority: 1
assignee: coopwilliams
---
# Explain Studio controls and measurements while Option is held

## Background
Cooper wants to learn what the controls and measurements mean while working in Studio, including on a remote computer.
## Outcomes
Holding Option or Alt over any substantive control or value outside the graph displays an accurate plain-English explanation. Releasing the key dismisses it. Include dialogs, live results, the command palette and unavailable controls. Draft every explanation in one catalog. Have Gemini Carbon write the displayed sentences in batches, review each entry against its implementation, and return factual errors to Carbon for revision.
## Method
Architecture change path: Studio graph editing and collaboration, browser shell and static assets. Check dynamic and static rendering paths, pointer and keyboard access, dialog layering and private-host delivery. Numerical engine, serialized graph, API mutations and inference execution N/A. Cooper waived the full suite and requested targeted GUI validation only.
## Future work
Additional controls can register help in the same catalog.
## Definitions
Carbon is the Gemini model selected in the installed agy client. Option is the Mac label for Alt.


## Notes

**2026-11-27T04:35:39Z**

Inventory complete. The single Studio catalog holds 289 explicit drafts covering toolbar commands, inspector and forms, picker, tours, review activity, run controls, eight storage arenas, compilation timings and executable budgets. The agy CLI verified gemini-4-carbon-high and is processing 15 entries per batch. Fields, shared controls and readouts use stable semantic identifiers.

**2026-11-27T04:46:00Z**

Gemini 4 Carbon via agy wrote the final text from the shared drafts in 20 batches using gemini-4-carbon-low. High mode passed the initial connection check but timed out on batch requests, so the authoring run used the low reasoning setting of the same Carbon family. Reviewed every first response against its rendered target and implementation. Approved 273. Returned 16 with specific corrections and reviewed Carbon’s rewrites in two more batches. Previous wording and reasons remain beside the affected entries. Compared all 289 final strings verbatim with successful CLI responses and checked the required prefix and 1–3 sentences.
