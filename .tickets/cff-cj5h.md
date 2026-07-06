---
id: cff-cj5h
status: open
deps: []
links: []
created: 2026-07-06T00:37:22Z
type: task
priority: 2
assignee: coopwilliams
---
# Ergonomics: epistemic lint, MI-term diagnostic, MAP tie-breaking

From the BFE-ablation discussion: (1) compile-time lint — goal priors on future observations with zero P-substitution beads => warn 'agent will not explore; intended?' (KL-control is legitimate, so warning not error); (2) expose the per-node MI/epistemic contribution from the FE ledger as a queryable diagnostic so 'why is my agent boring' is a number, not 30 trials of observation; (3) randomized or epsilon-temperature tie-breaking for MAP action selection — deterministic argmax turns indifference into frozen-at-start. Rationale: the P-bead is the objective declaration; silent-but-disappointing omission is the requires_grad class of footgun, mitigated by templates-carry-beads + machine-checkable blueprint.

