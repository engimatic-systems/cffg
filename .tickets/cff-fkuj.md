---
id: cff-fkuj
status: closed
deps: [cff-04mb, cff-5o7r]
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
2. Prior-mixture, linear-forward and linear-backward callers all consume that validity; no safe inverse placeholder produced inside moment collapse becomes an accepted outgoing message. Producer-side intermediate validity belongs to cff-61ag.
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

**2026-09-25T08:03:43Z**

Review question -> answer -> source -> confidence: Can inactive extreme components contaminate backward scores or shared-cavity gradients? Yes in the initial patch; fixed by retaining original selector support and borrowing an active component solely for masked arithmetic before component construction. Both NEG and literal -inf now have caller-level forward/backward/prior and jacrev regressions. Admission explicitly concerns active support, so an inactive rank-deficient component no longer discards valid active information. Independent read-only review found no remaining placeholder escape inside _collapse. Upstream component-construction inverses remain cff-61ag, explicitly recorded in its acceptance criteria; checked-collapse validity cannot certify an upstream hidden placeholder. Source: failure_provenance review and bounded float32 two-component reproductions. Confidence high.

**2026-09-25T08:32:36Z**

Implemented in a9b4842: checked CollapseResult validates computed moments and output; all three callers preserve exact selector support before component/score/gradient arithmetic. Known non-normalizable active components retain vacuous initialization; arithmetic collapse failure surfaces nonfinite outgoing evidence, not an inverse placeholder. Review confirms no remaining leak inside collapse. Final focused83 tests passed, including independent multidimensional moments,K32/d30,f32/f64,overflow,literal/sentinel zeros,vmap,forward/reverse derivatives,public Engine/Colony attribution,quadrature/switching parity and neural hybrid gradients. Final full gate2535 passed,83 deselected; pytest1210.25s,measured1226.099s. cff-5o7r moves four behavioral cases to slow, estimated1196.045s/2531tests, reusing the passing result per policy. No remaining work in checked-collapse scope. Next ready:cff-61ag producer-rank and intermediate-validity admission; cff-03dn and cff-ectw intentionally stay open.

**2026-09-25T08:34:07Z**

Published PR #171: https://github.com/engimatic-systems/cffg/pull/171, based on #170.
