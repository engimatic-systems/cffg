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

Corrected post-cff-hp3y Hallway rerun (L=8, H=10, 120 sweeps, seed 0,
25 episodes): with near-uniform Dirichlet counts the dynamics-novelty channel
dominates the goal channel — task success is 0.16 versus 0.76 for the learn-only
ablation, and visitation-weighted transition KL is also worse (2.650 versus
2.534). The earlier pre-fix 0.40/0.80 success comparison and claimed marginal
KL benefit are invalid and must not be cited. The GFE-TM composite also costs
~4x a plain GFE act() every step whether or not novelty remains. Add an
annealed novelty weight (decaying with accumulated count mass, or per-
(state,action) saturation) plus the option to drop back to the plain composite
once counts saturate. The controlled arm must recover goal performance without
claiming useful exploration unless it also improves model KL. Semantics stay
optimistic-bound (LIMITATIONS S1c); the weight is a Q-side constraint knob,
document its GFE interpretation.

## Acceptance Criteria

Annealed arm matches the learn-only ablation's task success within noise after
counts saturate; any claim of useful early exploration requires lower model KL
than that ablation at a preregistered window; per-step cost falls back to plain-
GFE level once annealed out; S1c updated.
