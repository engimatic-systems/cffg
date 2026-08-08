# rSLDS milestone refinement questions

**Primary ticket:** `cff-6tek`  
**Related tickets:** `cff-l7y4`, `cff-r6bl`, `cff-y4nx`

Delete this note when the tickets answer these questions and their dependency
records reflect the answers.

## `cff-6tek`

1. What exact address type replaces the current one-key `Arena.member_of`
   entry for factors such as `gaussian_linear`, whose matrix, offset, and
   precision occupy separate parameter keys?
2. Which exact factor and group families must receive member addresses? Please
   enumerate them and mark excluded families N/A.
3. What is the complete `set_params` contract: accepted role names, return
   value, tied-blob behavior, covariance-to-precision conversion, dtype and
   shape validation, and atomic failure behavior?
4. Which factor types must NIW and MNIW support: `gaussian`,
   `gaussian_linear`, and/or `gaussian_mixture_linear`? What recovery and
   hostile-input outcomes review each supported pair?
5. Can this ticket be rewritten into the required background, numbered
   outcomes, methodological suggestions, future work, and definitions
   structure before implementation starts?

## `cff-l7y4`

1. What exact outer variational objective, including its parameter-prior term,
   controls convergence and divergence?
2. What precise matching rule makes the rSLDS and SLDS parameter budgets
   comparable when their transition structures have different free-parameter
   counts?
3. Must shared-emission learning work in both distinct-trial continual
   accumulation and one-sequence coordinate replacement, or only in the new
   one-sequence fitting API?

## `cff-r6bl`

1. Does criterion 17 depend on `cff-j7mm`, which owns correctness of the
   non-discrete resource estimator, or does `cff-r6bl` own the necessary
   estimator repair?
2. Should speedups and inference regressions compare with the historical
   single measurements (`723.14` and `30.58` seconds), or with a freshly
   measured generic-path distribution using the new benchmark protocol?
3. What numeric or dtype-derived tolerances make criteria 7 and 9 independently
   reviewable?

## Sequencing and ticket policy

1. What remaining outcome in `cff-twzc` must land before `cff-y4nx` can close
   and unblock `cff-r6bl` and `cff-l7y4`?
2. Which heading is canonical for future tickets: AGENTS.md's
   `## Plain English` or the refinement template's
   `## Background in Plain English`?
