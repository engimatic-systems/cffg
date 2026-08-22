---
id: cff-ifbr
status: closed
deps: [cff-fbmr]
links: []
created: 2026-08-22T15:57:08Z
type: task
priority: 1
assignee: coopwilliams
parent: cff-0oqo
tags: [rslds, learning, readout]
---
# Use structured recurrence pairs for Gaussian and learning statistics

## Background in Plain English — WHY

Structured categorical messages are not enough if learning still reads a
product of singleton beliefs. D9 requires one consecutive-state pair belief to
own the at-risk and signed-success masses used by the Gaussian drive message,
the coefficient statistics, diagnostics, and conjugate update.

This ticket changes the public readout and learning boundary after cff-fbmr has
installed coherent structured inference state.

## Outcomes / Acceptance Criteria — WHAT

1. The engine exposes each recurrence factor's normalized pair belief with its
   source and destination marginals, exact support, and propriety status.
2. `gaussian_node_stats` and the recurrence pair readout agree on the Gaussian
   drive moments and categorical marginals. No readout recomputes a pair as an
   outer product.
3. `recurrence_weight_stats` computes exposure, signed-success, linear, and
   quadratic rows from the retained pair belief exactly as D9.12--D9.14 state.
4. The Gaussian-drive message used during inference and the between-trial
   coefficient update use the same pair-derived exposure and signed-success
   arrays bit-for-bit at their shared boundary.
5. `Learnables.replace`, accumulated updates, source filtering, tied factors,
   and `RSLDSLearning` pool the structured rows without changing prior
   ownership or counting one sequence twice.
6. Literal zero pair mass gives exact zero natural statistics and retains the
   prior for an unexposed source/stick. Improper, non-finite, asymmetric, or
   materially non-positive-semidefinite inputs fail loudly with existing
   diagnostic predicates.
7. An independent K=2/K=3 enumeration matches the pair readout and pooled
   natural statistics. Float32 and a K=8/T=100 batch remain finite and within
   dtype-scaled tolerances.
8. A regression fails if any production source silently returns to singleton
   products, including renamed factors and explicit `LearnableSpec.sources`.

## Methodological Suggestions — HOW

These suggestions are optional. The implementer may choose another sound
method.

- Make the pair belief the input to one shared at-risk/signed-mass helper used
  by inference and learning.
- Preserve the current direct Gaussian covariance readout and its symmetry/PSD
  diagnostics.
- Compare pooled posterior natural parameters against an independent NumPy
  update, not only the rolled coefficient mean.

## Future work

cff-63dd will prevent legacy product-statistic posteriors from silently posing
as structured posteriors. cff-h0hx will retain pair evidence and rerun the
development recovery gate.

## Definitions

- **At-risk mass:** Pair probability for a source regime and any destination
  that reaches a given decision-list stick.
- **Signed-success mass:** Success probability at one stick minus half its
  at-risk probability; this is the linear Pólya–Gamma statistic.
- **Natural statistics:** The linear and quadratic terms added to a Gaussian
  coefficient posterior.

## Notes

Architecture handoff: `Engine.gaussian_node_stats`, the new pair readout,
`Engine.recurrence_weight_stats`, `cffg.learn`, `hybrid_learning`,
`RSLDSLearning`, and their source/tied-blob validation are applicable. Compiler
message dispatch, scheduling, BFE, GFE, IR editing, and visualization are N/A
except for parity checks against cff-fbmr state.

Duplicated surfaces: inference/learning, direct/pooled rows, renamed/explicit
sources, single/batch, exact-zero/dense support, and float32/x64 must agree.

**2026-08-22T17:49:37Z**

Implemented by 303492e.

Public API: `Engine.recurrence_pair_beliefs(state)[factor]` returns
`pair_probability[K,K]` with source rows and destination columns;
source/destination marginals; literal pair/row/column support; direct drive
mean/covariance/second moment; `omega`, `exposure`, `signed_success`; and scalar
`proper`. `gaussian_node_stats` carries the same fields plus the existing
in/out aliases. `recurrence_weight_stats` derives `exposure`,
`signed_success`, `linear`, and `quadratic` from `pair_probability` only.

Replaced production singleton-product paths in `gaussian_node_stats`,
`recurrence_weight_stats`, `Learnables` `pg_weights` pooling, and the
recurrence-smoothing ownership diagnostic. Independent K=2/K=3 NumPy checks
match pair and natural parameters; inference/public/learning exposure and
signed-success agree bit-for-bit. Renamed explicit sources, tied pooling
without double counting, restricted source filtering, immutable replacement,
accumulation, exact-zero prior retention, direct covariance diagnostics,
float32, and JIT K=8/T=100 are covered.

Validation: combined affected selection 112 passed in 239.83s; follow-up
source-filter selection 5 passed in 23.75s; earlier focused
compatibility/oracle selections 39 passed. `git diff --check` and `py_compile`
pass. The default short suite was intentionally deferred until the full
structured-recurrence core stack is integrated, as authorized in the ticket
handoff.

Deferred boundary remains cff-63dd: the pair is deterministically
rematerialized from current engine state and is not yet independently
persisted or migration-versioned.
