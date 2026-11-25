---
id: chr-kzwu
status: in_progress
deps: []
links: []
created: 2026-11-25T06:05:31Z
type: feature
priority: 1
assignee: coopwilliams
parent: cff-pyy9
---
# Run a categorical regime and its child on separate clocks

## Background in Plain English

A slow hidden state can change which transition a faster process follows. Several observations of the faster process provide evidence about the same slow state. A controller must carry their correlation and avoid counting the slow prior again when evidence returns to it.

## Outcomes / Acceptance Criteria

1. A versioned declaration binds two named categorical templates and their ports and records a positive integer number of fast ticks per slow period. External probability arrays and both templates reconstruct the same runtime.
2. Fast updates recover the likelihood of each regime without its prior. Parent and child marginals and the joint Bethe free energy agree with direct enumeration in both supported precisions.
3. Support remains exact with impossible events and zero priors. Contradictions, nonfinite data and exhausted budgets have distinct outcomes and preserve the last completed tick. A vacuous observation carries a constant likelihood.
4. Child transitions and slow periods advance only on completed environmental events. A checkpoint resumes between boundaries, binds the declared model and its numerical inputs, and rejects unknown versions.
5. A declared goal update can be staged and committed at the episode boundary. Repeating inference within a trial adds no counts. At least ten tiny-oracle periods or ticks exercise the log-domain implementation.
6. The parent integration ticket receives the public interface, executable oracle, timing tradeoffs, and any unsupported composition.

## Methodological Suggestions

Use a variable elimination schedule on the declared categorical parent and regime-selected child factors. Retain the joint separator over the current regime and fast state. Compare a node-local ledger with independently enumerated path weights and stress contradictory observations, zero probabilities and float32.

The architecture change paths are Runtime numerical or status invariant and Learnable or checkpoint feature. This dedicated exact schedule owns its status, ledger and versioned checkpoint. Ordinary Engine, Colony, flooding, fixed-point iteration, stochastic factors and their legacy checkpoint formats keep their own execution and state. The two individual templates retain the existing IR and factor types. The hierarchy declaration is a separate versioned artifact.

## Future work

The parent epic cff-pyy9 owns the behavioral comparison, shallow contingent policy tree, shared documentation and suite gate.

## Definitions

A regime is the slow categorical state that selects the fast transition table. A cavity is evidence returned to a variable after excluding its incoming prior. A separator is the joint distribution carried across an elimination boundary. Bethe free energy is the sum of local expected negative log factors and entropy corrections. On this tree it equals the negative log evidence. A checkpoint stores the model state between calls.

