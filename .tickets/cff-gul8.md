---
id: cff-gul8
status: open
deps: [cff-cj5h]
links: [cff-cj5h, cff-fxmj, cff-hp3y, cff-4oq8]
created: 2026-07-10T03:31:28Z
type: feature
priority: 3
assignee: coopwilliams
---
# Epistemic sensing readout: expected information gain over candidate DATA ports

GFE composites score actions inside a fixed graph; there is no primitive for 'which unobserved DATA port is most informative to open next'. Consumers currently compute EIG in numpy from marginals + expected-log params (works — an EIG glimpse policy measured +27 accuracy points over random at fixed sensing budget — but it bypasses the engine and only covers the naive-Bayes case). Add an Engine-level EIG readout: for each candidate unbound discrete DATA port, the mutual information between a query edge and that port's variable under current beliefs (closed form from node beliefs for tree-local structure; document scope limits). Relates to the MI-term diagnostic in cff-cj5h.

## Acceptance Criteria

eig(state, query_edge, ports) matches hand-computed MI on a small fixture; the naive-Bayes-style case matches the external numpy computation; documented scope (tree-local, discrete v1).

