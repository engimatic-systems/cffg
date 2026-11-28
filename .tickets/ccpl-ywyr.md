---
id: ccpl-ywyr
status: closed
deps: []
links: []
created: 2026-11-28T06:57:23Z
type: bug
priority: 2
assignee: coopwilliams
parent: ccpr-ojza
---
# Keep node constraints attached to their factor when bonding a same-named variable

Variables and factors have separate names. A node can share a name with a variable. Bonding two ports or exposing a singleton Broadcast currently rewrites every constraint with that spelling, and a valid mean-field declaration can then point to a nonexistent node.

## Design

Use the existing constraint target categories in the core composition and plate-expansion change paths. Address lowering must make the same distinction. No inference rule or numeric tolerance changes.

## Acceptance Criteria

Chain and singleton Broadcast both keep a mean-field constraint on its original factor while moving the delta constraint on the variable. Validate both resulting templates.


## Notes

**2026-11-28T06:57:39Z**

Before the fix, the public expansion failed core validation on both controls: mean_field targeted the nonexistent p/hub after a singleton Broadcast and p/0/z_out after a chain. Both independent target assertions now pass. The compact compiler also keys renaming by constraint kind.
