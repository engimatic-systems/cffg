---
id: cff-gnpn
status: closed
deps: [cff-q0ep]
links: []
created: 2026-07-05T22:12:45Z
type: task
priority: 1
assignee: coopwilliams
parent: cff-br53
---
# Transactional edit-action API

Edit dataclasses (AddVariable/AddFactor/RemoveFactor/AddConstraint/DeclarePort/...), apply_edits validates the bundle result and leaves original untouched on failure.


## Notes

**2026-07-05T22:17:19Z**

Landed edits.py: 10 action types, apply_edits = deepcopy + apply-all + validate, all-or-nothing (EditError pre-validation, ModelError post). Tests cover rollback on both failure modes, the GFE<->BFE one-bead ablation flip (structure hash returns to baseline), RebindBlobPath hash-invariance. 30 tests.
