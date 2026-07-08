---
id: cff-2wse
status: closed
deps: []
links: []
created: 2026-07-08T02:57:59Z
type: task
priority: 1
assignee: coopwilliams
parent: cff-ql2n
---
# viz: diagram IR — derive uncompressed CFFG annotation from Template

cffg/viz/scene.py-ish: typed diagram IR (FactorBox, Wire, Bead, Terminal, CompositeBox, Ellipsis marker), each carrying source Template element IDs. Derivation per Part I §5: default = joint internal bead per node + empty bead per edge, wires extended through borders for marginalisation; MEAN_FIELD -> per-edge border beads; DELTA -> ∂ bead on edge; data ports -> filled ∂ terminal; param_blobs -> black square terminals; P_SUBSTITUTION -> square bead on substituted role; GFE composite pairs (reuse compiler detection logic) -> dashed box.

