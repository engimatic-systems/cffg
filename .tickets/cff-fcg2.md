---
id: cff-fcg2
status: in_progress
deps: [cff-2wse]
links: []
created: 2026-07-08T02:57:59Z
type: task
priority: 1
assignee: coopwilliams
parent: cff-ql2n
---
# viz: compression passes 1-4 (p.19 recipe) on diagram IR

Four pure diagram->diagram transforms: (1) summarise bead chains by terminating beads; (2) remove empty internal beads on nodes with default factorisation; (3) remove internal edges on such nodes (careful: skip when a connected edge is pointmass in the model, cf. f_d in Fig 16/17); (4) push remaining internal beads to node borders. render level 0-4 selects prefix of passes; level 4 also shrinks boxes (Fig 19).

