---
id: cff-fkuj
status: in_progress
deps: [cff-04mb]
links: []
created: 2026-09-25T07:56:07Z
type: bug
priority: 1
assignee: coopwilliams
parent: cff-03dn
tags: [pcgym, numerics]
---
# Validate Gaussian-mixture moment collapse before exposing messages

## Background in Plain English

cff-03dn found that safe-inverse placeholders can escape from Gaussian-mixture moment collapse. Component precisions passing a check does not prove that their computed covariances, means, collapsed covariance or final message are valid. Overflow or ill conditioning later in the calculation needs its own outcome.

This ticket establishes checked collapse. Producer-side numerical-rank admission is separate downstream work; do not tighten the shared Gaussian propriety predicate here.

## Outcomes / Acceptance Criteria

1. Moment collapse returns explicit validity with its output information vector and precision. Checks cover normalized supported weights, admitted component moments, collapsed covariance and final finite proper output.
2. Prior-mixture, linear-forward and linear-backward callers all consume that validity; no safe inverse placeholder becomes an accepted outgoing message.
3. Preserve the existing vacuous fallback when components are known not to be normalizable yet. A failed numerical collapse of otherwise admitted components is distinguishable from this legitimate initialization case and surfaces a truthful failure.
4. Independent multidimensional moment oracles verify ordinary proper mixtures in float32/float64, including exact zero selector support and diffuse but proper components.
5. Nonfinite means, invalid covariance calculations and representational overflow cannot return plausible positive precision. Host/JIT and single/vmap agree; at least tenfold component count and differentiable proper-case gradients are covered.
6. Existing quadrature, backward-weight and energy tests pass. Parameters, objective, global propriety and run-control decisions are not changed to force convergence.

## Methodological Suggestions

A typed auxiliary CollapseResult is one possible method; another sound method is acceptable. Reuse branch-safe linear algebra only behind checked validity. Architecture: Runtime numerical/status invariant and New factor/rule family. Local owners: mixg._collapse and its three callers. Host/JIT, flood/scheduled and Engine/Colony share those owners; validate transport for a real failing factor. Current/legacy checkpoints N/A, because no stored schema changes. Continue the validated chain and run focused tests plus one final short suite.

## Future work

- cff-03dn consumes this foundation before completing producer-side rank admission.
- cff-u9un validates the retained hybrid planner after arithmetic prerequisites.

## Definitions

Moment collapse replaces a mixture by one Gaussian with matching first two moments. A proper Gaussian has positive-definite covariance. A branch placeholder is a dummy matrix used only to keep an unselected compiled calculation safe. Vacuous means no information. JIT is compiled execution; vmap batches instances; dtype is numeric storage precision.



## Notes

**2026-09-25T07:56:22Z**

Pre-implementation handoff: Runtime numerical/status invariant and New factor/rule family rows. One local checked-collapse owner used by mix_forward, mixlin_forward and mixlin_backward; their registered cross-domain execution covers host/JIT, flood/mixed, Engine/Colony and vmap. New auxiliary internal validity, not a persisted schema. Known non-normalizable components retain existing vacuous fallback; numerical failure after admission must become nonfinite outgoing evidence, tagged by existing failure transport. Tests will cover actual public factor attribution, proper independent moments and JVP/VJP finite behavior, literal/sentinel zero weights, malformed values, float32 overflow and K32/dimension-scale. No objective or default-run policy changes.
