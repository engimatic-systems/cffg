---
id: cff-61ag
status: open
deps: [cff-fkuj]
links: []
created: 2026-09-25T07:56:07Z
type: bug
priority: 1
assignee: coopwilliams
parent: cff-03dn
tags: [pcgym, numerics]
---
# Preserve numerical-rank evidence when admitting Gaussian mixture components

## Background in Plain English

cff-03dn found that reconstructing a projected rank-one precision can produce a tiny positive eigenvalue. Strict positivity of that rounded matrix must not override a direction that the projection explicitly discarded. Backward messages can similarly gain apparent rank when forming A.T G A.

The shared Gaussian propriety predicate deliberately supports valid widely scaled diagonal systems. This ticket therefore owns rank evidence at the message producer, not a new global condition-number cutoff.

## Outcomes / Acceptance Criteria

1. Schur projection retains its dtype-relative spectral admission evidence before matrix reconstruction; linear-forward mixture normalization requires certified full output rank on that branch.
2. Proper-input covariance-form branches preserve well-scaled weak information and do not inherit an unrelated Schur cutoff.
3. Backward rank admission uses the generating factor of A.T G A, including retained informative subspaces and coordinate equilibration; singular G may still inform every input dimension.
4. Rotated exact nullspace, proportional-column maps and coordinate-rescaled proper cases agree with independent algebraic/moment oracles in float32/float64.
5. Exact-zero selector support, vacuous inputs and at least tenfold dimension/component scale have explicit outcomes; supported JIT/vmap and proper-branch gradients remain sound.
6. No global covariance floor, model-parameter change or shared propriety cutoff is introduced. Checked-collapse outcomes distinguish initialization from arithmetic failure.
6.1. Component producers validate actual computed covariance/precision intermediates before admitting a component, including the proper-input covariance-form branches. A safe inverse placeholder from an invalid predicted covariance cannot reach checked collapse as plausible component information.
7. Update canonical derivation/limitation contracts and executable anchors. The retained planning fixture is handed to cff-u9un; this ticket does not claim complete planner convergence by itself.

## Methodological Suggestions

Another sound approach is acceptable. Consider returning retained spectral factors/rank from gauss._project_psd, using those factors before forming backward Gram matrices, and equilibrating columns before rank tests. Inspect d=100 mixed-scale Gaussian tests before changing any predicate. Architecture: Runtime numerical/status invariant and New factor/rule family. Host/JIT, flood/scheduled, single/vmap and Engine/Colony consumers apply. Checkpoint schemas N/A. Run focused tests and one final short suite.

## Future work

- cff-03dn closes after this and checked collapse meet the complete arithmetic contract.
- cff-u9un tests initialization, finite free energy and safeguarded planning after repair.

## Definitions

Rank counts independent informed directions. A Schur complement eliminates Gaussian variables. A Gram matrix is a factor multiplied by its transpose. Equilibration rescales coordinates for numerical analysis, not the probabilistic model. JIT is compiled execution; vmap batches instances; dtype is numeric storage precision.

