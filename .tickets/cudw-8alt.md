---
id: cudw-8alt
status: in_progress
deps: []
links: []
created: 2026-11-28T13:55:33Z
type: task
priority: 2
assignee: coopwilliams
parent: cff-g7rc
---
# Reduce cold derivative compilation for disjoint categorical writes

## Background in Plain English

A certified solver evaluates the gradient of its residual merit even when it can accept a candidate without taking a trust-region step. With the shared propagation optimization at cc6b66b, median cold XLA compilation of that diagnostic alone took 0.646 s for the T-maze and 0.481 s for dynamics curiosity. Packed categorical rule outputs write to fixed scalar addresses. Without a uniqueness guarantee JAX computes an extra set of winner IDs when differentiating an overwrite.

## Outcomes / Acceptance Criteria

1. Preserve every original status, certificate, control, step count, state digest and reported quantity of both retained fixtures in x64, both host and compiled controllers. Preserve the exact support face, masks, current and mapped free energies, nonfinite derivative rejection, sparse/vacuous support and dtype-dependent tolerances.
2. Disjoint static destinations receive the uniqueness guarantee. Duplicate, wrapped or out-of-range addresses keep the original operation. Compare actual values, Jacobian-vector products and transposed derivatives with ordinary JAX indexed writes under float32/64 and a plate at least 10x the size of a toy case.
3. Pre-edit prediction: in five alternating fresh processes per revision and fixture, default host controller on CPU/x64, five synchronized warm repeats, no persistent cache, reduce median cold compilation of jit(diagnostics) by at least 10% on each case. Require positive cold wall improvement on both with at least 3% geometric-mean reduction, and at most 5% geometric-mean warm regression. Report each observed value; the broader 2x goal remains open. The compiled controller has a separate exact parity check. Do not retime the original propagation change.
4. Check the small-unit comparison and numerical failure surfaces, and run the dashboard short suite at the final code head before opening review.

## Methodological Suggestions

Use the Runtime numerical or status invariant change path. The discrete transport owner feeds flooding, mixed synchronous and scheduled categorical writes, and JIT, vmap and reverse-mode differentiation. Confirm native priors/observations and mask metadata remain runtime values. No structural cache, changed derivative equation, solver mode, semiring, JAX flags, new IR, output fields or checkpoint migration. Use the unchanged qualified worker from csp-t7n3 for the original two fixtures and verify every reported cold and warm result including the derivative diagnostics. Existing nonfinite-gradient regressions from #236 apply. Record tracing versus compiled costs separately.

## Future work

The parent cff-g7rc retains all wider latency and control gates. The root diagnostic alone is less than half of the cold public solve.

## Definitions

A scatter writes array elements at specified addresses. Its transpose tells gradient calculations which input or write received each output contribution. A JIT compiler translates an array function on first use. BFE means Bethe free energy, the scientific objective checked at the candidate and its successor.

