---
id: cff-0xjq
status: open
deps: [cff-ao02, cff-7194]
links: []
created: 2026-08-21T03:07:56Z
type: task
priority: 1
assignee: coopwilliams
parent: cff-qulu
tags: [rslds, validation, numerics]
---
# Choose the outer convergence certificate from a development replay

## Background in Plain English — WHY

The recorded heterogeneous failures may combine three mechanisms: a residual
that measures the wrong representation, a budget that stops slow but healthy
progress, and a full parameter-replacement map that does not contract. These
mechanisms require different fixes.

A bounded development replay must separate them before `cff-qulu` changes the
convergence certificate or adds a safeguarded step.

## Outcomes / Acceptance Criteria — WHAT

1. A development-only command rejects final seeds and records legacy,
   canonical runtime-map, and truth-free functional residuals with per-family
   attribution.
2. The replay records residual-vector direction, inner work, map evaluations,
   and coherent terminal status for one declared rSLDS tier and same-K SLDS
   control.
3. The evidence decides whether the production certificate changes and whether
   budget, globalization, or gauge handling owns the next repair.
4. The raw result and decision are manifested and reproducible from committed
   code.
5. `cff-qulu` receives a handoff with the selected certificate, falsifiers, and
   measured budget boundary.
6. The work generates and inspects no registered final seed.

## Methodological Suggestions — HOW

These suggestions are optional. The implementer may choose another sound
method that satisfies the outcomes.

- Use calibration seed 3101, replicate 1, only after compact tests pass.
- Start with shared-initialization rSLDS and the same-K control under a bound of
  32 to 48 map evaluations.
- Run the second tier and K=5 control only when the first pair cannot decide the
  mechanism.
- Treat a stable functional residual with an unstable runtime residual as a
  separate gauge or canonicalization finding, not permission to hide it.

## Future work

`cff-qulu` implements the selected certificate and any justified safeguarded
outer step. `cff-0oqo` re-evaluates source recovery only after that repair. A
separate ticket will own gauge handling if the replay shows that it is needed.

## Definitions

- **Development replay:** A diagnostic run on a declared nonfinal seed that may
  guide implementation choices.
- **Gauge:** A parameter change that leaves the represented probability model
  unchanged.
- **Residual-vector direction:** The angle or cosine between successive outer
  update vectors, used to detect oscillation or a two-cycle.

## Notes

Architecture handoff: this ticket consumes the CFFG observed-only residual
contract and the sibling experiment's matched-control terminal contract. IR,
compiler, VM, schedules, kernels, ledger, and serialization are N/A.

Duplicated surfaces: one rSLDS tier, the same-K control, legacy/runtime/
functional residuals, phase work, and terminal status must stay aligned before
additional arms run.

Legend: prerequisite --> dependent

```text
[ ] cff-ao02 canonical residual (ready) --+
[ ] cff-7194 coherent SLDS result (ready) +--> [ ] cff-0xjq replay
                                                        |
                                                        +--> [ ] cff-qulu
                                                                  |
                                                                  +--> [ ] cff-0oqo
```
