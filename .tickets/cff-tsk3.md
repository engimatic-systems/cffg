---
id: cff-tsk3
status: closed
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

**2026-11-27T04:47:09Z**

Landed and deployed Option/Alt hover and Tab-to-focus help on the shared Studio launcher. One preloaded catalog supplies approved copy to static controls, forms, dynamic jobs and native modal dialogs. Release of Alt, Escape or loss of window focus dismisses it. Targeted checks only, per Cooper. The checked-in tools/check_studio_help.py passed in Chromium and its equivalent passed in WebKit. Disposable Playwright runs physically displayed all 289 approved entries, including one real benchmark, a step-controlled inference trace, indexed file import, and fixtures for uncommon worker results. Files: /tmp/cffg-studio-qa/option-help-final.log and option-help-final-host.log. The live Chromium check followed the agent-share launcher to private HTTPS port 8098, matched the delivered catalog byte-for-byte, and verified a Profile value and picker tooltip. JavaScript syntax, tk dep cycle and git diff --check passed. No outstanding work.
