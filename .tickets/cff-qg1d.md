---
id: cff-qg1d
status: closed
deps: []
links: []
created: 2026-08-30T16:00:47Z
type: feature
priority: 1
assignee: coopwilliams
parent: cff-0qfy
tags: [consolidation, learning, hybrid]
---
# Port hybrid declared learning onto current main

## Background in Plain English — WHY

PR #64 adds declared conjugate learning for switching Gaussian dynamics and
recurrent stick-breaking weights. The useful delta is based on an old roadmap
branch and is mixed with an old ticket close and generated runtime history.
This ticket recreates the behavior directly on `main` commit `d805135` without merging
that branch.

The applicable architecture change-path row is **Learnable or checkpoint
feature**. `LearnableSpec` and `cffg.learn` own declarations and family
dispatch. The hybrid adapter owns factor-specific posterior shapes and
statistics.

## Outcomes / Acceptance Criteria — WHAT

1. The production, test, and canonical-documentation behavior from PR #64 is
   present relative to `main` commit `d805135`.
2. `gaussian_mixture_linear` component MNIW learning pools tied soft statistics,
   preserves exact zero responsibility, validates component axes, and writes
   matrix, offset, and covariance roles atomically.
3. `recurrence_link` Pólya–Gamma learning pools tied natural statistics,
   preserves source-regime and stick axes, and writes weights and offsets
   atomically.
4. Both families work through public `Learnables` accumulation, replacement,
   reporting, rollover, source selection, and checkpoint interfaces without
   factor-name conventions.
5. Invalid shapes, improper or non-finite statistics, and failed writes cause
   a loud failure before posterior or parameter mutation.
6. Sparse or zero support remains exact, float32 values stay finite with
   dtype-scaled guards, and the retained uncomfortable-scale fixture completes
   without a dense state-dependent transition tensor.
7. The change-path surfaces agree: validation and compilation resolve the new
   families; parameter addresses and engine readouts remain the write and
   statistic authorities; checkpoint encoding covers the new posterior axes.
8. Duplicated surfaces are checked as follows: flooding and scheduled source
   statistics agree; learning remains host-side, so host/JIT mutation is not a
   new pair; single-instance behavior is covered and no new `vmap` learning
   surface is introduced; current checkpoints round-trip and unsupported legacy
   forms fail explicitly.
9. `README.md`, `docs/ARCHITECTURE.md`, and `docs/LIMITATIONS.md` describe the
   supported pairings and remaining limits as they apply on `main` commit
   `d805135`.
10. Old `.tickets/cff-dqk2.md` state and old runtime measurements are absent.
    Dashboard test registration is retained only where the new focused test
    file requires it.

## Methodological Suggestions — HOW

These suggestions are optional. The implementer may choose another sound
method.

- Reapply the four production commits from PR #64 without the old ticket-close
  commit, then resolve conflicts one logical commit at a time.
- Keep hybrid family code in a focused module and compare it with the direct
  MNIW and Pólya–Gamma posterior objects used as test oracles.
- Run `tests/test_hybrid_learnables.py` after each behavior-bearing commit.

## Future work

cff-0qry depends on this ticket and will declare these learning families on the
rSLDS model and fitter. No other hybrid family is added here.

## Definitions

- **Declared learning:** A serialized template names the statistic sources,
  posterior family, and parameter targets for one learned group.
- **MNIW:** Matrix-Normal-Inverse-Wishart, a conjugate posterior for a linear
  matrix, offset, and covariance.
- **Pólya–Gamma learning:** The Gaussian conjugate update for recurrent
  stick-breaking logistic coefficients after augmentation.
- **Tied statistics:** Sufficient statistics pooled from factors that share the
  same serialized parameter blobs.

## Notes

**2026-08-30T16:06:49Z**

Ported PR #64 as four new atomic commits on main d805135. Retained production code, test coverage, canonical documentation, and the required EPIC_FILES registration. Excluded the old cff-dqk2 ticket state and tools/test-suite-runtimes.json. Focused validation: uv run pytest -q tests/test_hybrid_learnables.py -> 17 passed.
