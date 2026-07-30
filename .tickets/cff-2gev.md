---
id: cff-2gev
status: open
deps: [cff-hp3y, cff-4oq8, cff-b47g]
links: [cff-weff]
created: 2026-07-10T03:34:54Z
type: feature
priority: 3
assignee: coopwilliams
---
# Continual transition curiosity under nonstationarity

## Plain English

Transition curiosity remains available whenever learned dynamics become
surprising again. This ticket replaces one-way novelty annealing with a
generative account of forgetting, change points, or latent regimes that can
restore uncertainty. The default stationary model keeps its Bayesian
epistemic objective active rather than assuming curiosity expires on a
schedule.

## Rationale

The existing learned GFE-TM novelty term is information gain under the current
posterior. Its practical influence naturally shrinks as a stationary transition
model becomes certain, without an extra coefficient declaring that curiosity
is over. A monotone count-based weight cannot increase when the environment
changes, so it confounds a compute shortcut with a claim about the agent's
generative model.

Before another experiment is preregistered, choose and derive an explicit
nonstationarity mechanism. Candidate models include principled evidence
discounting, change-point inference, and latent transition regimes. Compare
their semantics and computational cost separately from the epistemic
objective. Any optimized path must preserve the declared objective and policy
ordering within a stated error; it must not silently replace curiosity with
plain planning.

## Historical Evidence

Closed PR #42 retains the preregistered monotone-annealing protocol, and closed
PR #43 retains its implementation and invalid run. They were deliberately not
merged. The run reached an outer mixed-GFE limit cycle before the comparative
experiment began, so it says nothing about annealing efficacy. The independent
solver problem is linked as cff-weff and is not a dependency or an M3 blocker.

The corrected flat Hallway result remains negative evidence about that model:
with near-uniform Dirichlet counts, dynamics novelty produced 0.16 task success
versus 0.76 for the learn-only ablation and no model-error improvement. It does
not establish a schedule for a factored or nonstationary model.

## Hostile Inputs

- Exact log-zero support and newly possible transitions must have defined
  posterior and novelty behavior.
- Vacuous priors must remain proper or fail loudly.
- Float32 guards must derive from the active dtype.
- Evaluation must include parameter blocks 10--100 times larger than the small
  corridor fixture and at least one environment change after apparent
  saturation.

## Acceptance Criteria

- The design states which generative assumption permits old evidence to lose
  force and derives its transition-learning update.
- The stationary limit recovers the existing full-strength Bayesian GFE-TM
  objective; the default does not silently decay or disable epistemic value.
- After a change, the selected model can regain uncertainty and direct
  information-seeking behavior toward the affected transition parameters.
- Compute-cost optimization and scientific claims about curiosity have
  separate metrics and acceptance thresholds.
- A replacement experiment is preregistered only after the mechanism, solver
  requirements, hostile-input behavior, and independent model-error metric are
  fixed.
