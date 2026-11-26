---
id: cff-xikb
status: closed
deps: [cff-cz82, cff-7noc, cff-w551, cff-5mf2, cff-fw2v]
links: [cff-0266]
created: 2026-11-26T19:44:48Z
type: epic
priority: 1
assignee: coopwilliams
tags: [studio]
---
# Build CFFG Studio for shared model editing and experiments

## Background in Plain English — WHY

Cooper and coding agents need one place to edit, discuss and run the same models. Today the library supplies individual pieces and Cooper uses the remote file share to read results.

## Outcomes / Acceptance Criteria — WHAT

1. A remote user can open a blank model or a current rSLDS starter, edit it entirely by keyboard and recover any saved revision.
2. A coding agent can inspect, critique and propose changes through the same compile-checked commands, and neither party can silently overwrite a newer edit.
3. The shared catalog can import discovered models, stream reproducible experiments and compare measured resource use.
4. A user can request a simulator, inspect its code, and use reproducible generated observations in a model run.

## Methodological Suggestions — HOW

Follow the delivery tickets and the stack proposal in docs/todo-notes/cffg-studio.md. Remove that future-work note when this epic lands and move the delivered contracts into the existing canonical docs.

## Future work

The linked implementation tickets deliver the workstation. Model families continue to be owned by the CFFG engine.

## Definitions

CFFG means constrained Forney-style factor graph. An rSLDS is a recurrent switching linear dynamical system. A revision records one complete accepted edit.


## Notes

**2026-11-26T19:49:23Z**

Delivery handoff from the stack proposal. The arrows point from prerequisite to dependent. Each terminal feature feeds this epic. Two new roots and the existing estimator repair can proceed in parallel.

cff-0266 proposal (closed)
  +--> cff-x6c0 model library + transactions (ready)
  |      +--> cff-w551 remote entry + discovery -----------+
  |      +--> cff-7noc agent critique and review -----------+
  |      +--> cff-cz82 keyboard editor [also needs 7piw] ---+
  |      +--> cff-djod runs + observations                  |
  |             +--> cff-fw2v simulator --------------------+
  |             +--> cff-5mf2 profiler [also needs j7mm] ---+
  +--> cff-7piw rename, resize, init (ready) --> cff-cz82  |
cff-j7mm estimator repair (existing, ready) --> cff-5mf2   |
                                                          +--> cff-xikb

Verified with tk dep tree --full cff-xikb, tk ready, tk blocked and tk dep cycle. First reviewable workflow: create an rSLDS from the remote browser, reopen it and have an independent agent retrieve the same revision. The proposal is at docs/todo-notes/cffg-studio.md. Retire the note when the epic lands.

**2026-11-26T21:13:28Z**

The private workstation is deployed and agent-share launches it. The model library contains a blank starter, linear Gaussian tracker and rSLDS. README.md documents the owner experience, agent transactions and host installation. ARCHITECTURE.md records the workstation and resource contracts. The proposal todo note has been retired. Production generation, deterministic simulation and profile/replay succeeded. The dashboard-backed default run at 943f8be passed 2762 tests, 335 opt-in deselected. Measured wall 1231.168s. Two Git integration scenarios measured 19.916s and 14.606s; moving those passing scenarios to the opt-in slow suite gives estimated default wall 1196.646s. Both remain available with uv run --extra studio pytest -m slow tests/test_studio.py. Selection-only changes reuse this passing result. No outstanding deployment step.
