---
id: cff-qh8e
status: in_progress
deps: []
links: []
created: 2026-11-27T10:50:40Z
type: task
priority: 2
assignee: coopwilliams
---
# Show tooltip explanations without the authoring prompt

## Background
Cooper uses “This element is” as the prompt for Gemini Carbon to complete. Reading that preamble on every hover delays the explanation.
## Outcome
Every Studio tooltip starts with the completion, capitalized for display. Keep the reviewed catalog and the rest of each explanation intact.
## Method
Architecture change path: Studio model editing and experiments. All hover and focus targets share the browser tooltip renderer, including native dialogs. Model serialization, engine execution and agent API changes N/A. Validate the catalog and targeted browser behavior.
## Future work
New approved entries use the same display rule.


## Notes

**2026-11-27T10:52:31Z**

The shared renderer removes the exact leading prompt and capitalizes the first character of the completion, including the text exposed to assistive technology. The catalog retains Carbon’s reviewed original. Targeted WebKit and Chromium gesture/dialog checks passed.
