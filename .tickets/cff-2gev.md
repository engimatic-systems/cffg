---
id: cff-2gev
status: open
deps: [cff-hp3y]
links: []
created: 2026-07-10T03:34:54Z
type: feature
priority: 3
assignee: coopwilliams
parent: cff-mwjo
---
# GFE-TM novelty weight: anneal with accumulated count mass

Measured in a learning stack: with near-uniform Dirichlet counts the dynamics-novelty channel dominates the goal channel — early plans chase count mass (task success 0.40 vs 0.80 for the learn-only ablation over the first 25 episodes) while buying only marginally faster model-error reduction; and the GFE-TM composite costs ~4x a plain GFE act() every step whether or not novelty remains. Add an annealed novelty weight (decaying with accumulated count mass, or per-(state,action) saturation) plus the option to drop back to the plain composite once counts saturate — recovering goal performance while keeping early directed exploration. Semantics stay optimistic-bound (LIMITATIONS S1c); the weight is a Q-side constraint knob, document its GFE interpretation.

## Acceptance Criteria

Annealed arm matches learn-only ablation's task success within noise after counts saturate while reaching lower model KL earlier; per-step cost falls back to plain-GFE level once annealed out; S1c updated.

