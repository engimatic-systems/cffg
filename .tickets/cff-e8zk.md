---
id: cff-e8zk
status: open
deps: []
links: []
created: 2026-07-08T16:50:10Z
type: feature
priority: 2
assignee: coopwilliams
parent: cff-ahps
---
# Constraint/optimization factors: min-sum semiring + hard-constraint library

Traditional constraint satisfaction / optimization under uncertainty. Two parts. (1) SEMIRING MODE: the discrete BP rules are sum-product einsums over (logsumexp, +); min-sum/max-product replaces logsumexp with max — same arenas, same gather/scatter, a mode flag on GroupPlan like bp/vmp. Ties into the existing delta/MAP machinery (harden phase becomes exact Viterbi on trees). (2) HARD-CONSTRAINT FACTOR LIBRARY: alldifferent, exactly-k/at-most-k, table constraints, linear (in)equalities over discrete domains — each a structured factor whose messages are computed combinatorially (never materialize the full potential tensor; alldiff BP has known permanent-based updates, use the Huang-Jebara style approximation and document it). Uncertainty enters by mixing soft probabilistic factors and hard constraints in one graph — that IS optimization under uncertainty in this framework. Later: continuous linear constraints on Gaussian edges (equality projections are exact). Test tier: mini-Sudoku / graph-coloring exact on small instances vs enumeration; weighted-CSP MAP vs brute force. DELEGATION: Fable specifies semiring semantics + FE story in min-sum mode (FE ledger is sum-product-specific: min-sum reports objective value, not FE — must be explicit in the ledger API); Opus implements the mode flag + constraint messages against enumeration tests; Sonnet surveys toulbar2/PGMax MAP benchmarks for parity targets.

## Acceptance Criteria

Min-sum on trees == Viterbi/brute-force MAP; mini-Sudoku solved via BP+decimation; mixed soft+hard graph returns MAP consistent with enumeration; ledger API distinguishes FE from objective value.

