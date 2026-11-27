---
id: cff-ydad
status: in_progress
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
