---
id: cff-4kzc
status: closed
deps: []
links: []
created: 2026-11-27T21:33:54Z
type: feature
priority: 1
assignee: coopwilliams
---
# Guide agents through authoring tours that frame related graph elements

## Background
Cooper wants to hand a model to a coding agent for a narrated tour, then use Left and Right to follow the explanation across the canvas. Existing tours attach one stable target per stop and the view only sometimes recenters.
## Outcomes
A model-specific instruction can be copied for an agent. A stop accepts an overview, one reference or a group in a single model or plate scope. Playback visibly frames and highlights its targets and supports keyboard navigation, rapid paging, reduced motion, renamed elements and missing targets.
## Method
Architecture row: Studio model editing and experiments. Check agent CLI and API metadata transactions, project export and import, browser playback and canvas culling, tours dialog and documentation. Engine inference and native bundle numerics N/A. Validate with targeted tour GUI and authoring workflow checks per session waiver.
## Future work
Freehand camera paths are outside this request.
## Definitions
A reference identifies a named graph item. A stable anchor lets the tour keep finding it when its name changes.


## Notes

**2026-11-27T21:41:42Z**

Agent tour commands can use ordered refs within a single scope. The focused Studio tour regression passed for group validation, rename, project exchange, and revision edits. The edit-scoped CLI also published and reimported a grouped tour in a disposable library.

**2026-11-27T21:45:27Z**

Shipped Tours → Copy agent prompt, keyboard paging, scoped group highlights and a camera that fits visible targets around the narration. Removed references are named and remaining members are shown. Reviewed and approved all three new Gemini Carbon tooltips against the implemented behavior. Targeted validation: Studio tour pytest passed, edit-scoped CLI preview/apply and project round-trip passed, Chromium and WebKit desktop/phone playback passed for rapid paging, reduced motion, plate scope, rename and partial deletion. Long narration and 3,276-row culling passed; a live agent edit updated an open plate stop over SSE. Private agent-share launcher, real rSLDS deep link and clipboard/fallback were exercised after installing release 8262031d0188fd65. The full suite was waived for this GUI session.
