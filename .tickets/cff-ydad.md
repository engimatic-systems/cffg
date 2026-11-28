---
id: cff-ydad
status: closed
deps: []
links: []
created: 2026-11-27T04:55:42Z
type: bug
priority: 1
assignee: coopwilliams
---
# Make Studio hover help respond to Mac Option input on remote clients

## Background
Cooper sees no tooltips on macOS after tabbing to and hovering over Studio controls. The shipped controller arms only on one particular modifier-key event. Existing browser checks pressed Playwright Alt after entering the page.
## Outcomes
Recognize left and right Mac Option and the modifier flags carried by pointer events, including when the page missed the original key press. Retain Tab-to-focus, dismissal and ordinary shortcuts. Give the existing footer hint a keyboard-accessible toggle so the feature also works when the client intercepts modifier keys. A fresh navigation after deployment must load one consistent UI release.
## Method
Architecture change path: Studio model editing and experiments, browser input, shell and static delivery. Test keyboard and pointer orders, Mac alternate modifier reporting, native dialogs and release changes using targeted GUI tests. Numerical execution, model serialization and agent API changes N/A.
## Future work
No additional model or engine work is needed.
## Definitions
Option is the macOS Alt modifier. A pointer event reports modifiers held when the pointer moved, independently of key events delivered to the page.


## Notes

**2026-11-27T05:04:28Z**

Native-input research. Question: do ordinary left/right Option keys have a different DOM identity on Mac, or can the page miss the keydown? On macOS 15.7.5 a private, offscreen AppKit WKWebView reported AltLeft and AltRight, altKey true, AltGraph false when it had keyboard focus. A native modifier-bearing pointer move reached Tours even when keyboard focus elsewhere withheld the keydown. The previous controller showed no tooltip. The patched controller showed the approved text and dismissed it on unmodified motion. Sources: /tmp/cffg-studio-qa/option-native-shell-before.log and option-native-shell-after-latest.log. High confidence in this reproduced failure mode; the remote client itself was not instrumented. Chromium and WebKit also reproduced missing keydown when Option was held before page entry.

**2026-11-27T05:06:18Z**

Deployed build 2f97afbc629907a8 to the existing private port 8098 and updated the agent-share launcher. The before/after test against the deployed service saw zero tooltips before and one after for a trusted pointer move with Option held before page entry in both Chromium and WebKit. Tested both native Mac Option keys, keyboard focus in a modal, pointer-only activation/dismissal, the footer toggle, Escape and DOM replacement. The GUI check exercised all 290 approved explanations across 17 workflows including a real benchmark and live trace, with zero failures. The new button explanation was drafted in the catalog, authored by Gemini Carbon via agy, and reviewed. Verified upgrade from a long-cached pre-help shell in Chromium/WebKit, the live agent-share redirect, versioned assets, Profile and file picker. Evidence: /tmp/cffg-studio-qa/macos-option-verified.log and option-native-shell-after-latest.log. Only targeted GUI checks were run per session waiver. No outstanding work.
