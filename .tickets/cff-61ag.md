---
id: cff-61ag
status: closed
deps: [cff-fkuj, cff-q6zc]
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


## Notes

**2026-09-25T08:34:23Z**

Pre-implementation handoff: Runtime numerical/status invariant and New factor/rule family rows. Preserve shared _pd_ok and ordinary Gaussian rule outputs. Add producer-owned PSD projection facts (retained eigenvalues/basis and validity before reconstruction); keep old _project_psd matrix API as wrapper. Forward Schur branch requires retained full output rank. Proper-input covariance branch validates computed input/noise/predictive covariance and output precision without applying Schur rank cutoff. Backward tests rank of a generating factor (spectral factor of G times A), with dimensionless row/column equilibration before SVD; never use rounded Gram positivity as rank evidence. Rank facts are nondifferentiated auxiliary predicates. Distinguish valid but non-normalizable active components (vacuous fallback) from invalid numerical construction (nonfinite outgoing evidence). Sparse selector support uses the checked-collapse preprocessing. Required pairs: host/JIT, flood/mixed, Engine/Colony, single/vmap and derivatives; checkpoint schemas N/A. Hostile tests planned: exact rotated nullspace A=(3,4).T, proportional-column backward A, supported subspace with singular G but full input rank, diagonal unit rescaling, improper/indefinite inputs, f32/f64 and d>=30/K>=32. Independent moment/nullspace oracles and saved horizon15 replay remain the authority, not a covariance floor or relaxed status.

**2026-09-25T08:40:27Z**

Research review: can backward coordinate equilibration restore a discarded null direction? Yes: W=I, Lambda_o=outer([4,-3],[4,-3])/25, A=[3,4].T gives retained G eigenvalues [0,.5], but F A rounds to [0,-4.44e-16]. Naive rescaling admitted false precision1.11e-16. f32 [2,3] likewise. Remedy under test: componentwise product-error masking before equilibration, separate finite-product validity, unconditional output precision finiteness. Source: failure_provenance read-only local experiment and gaussian_components.py; high confidence, both dtype regressions retained. No global precision cutoff.

**2026-09-25T08:43:47Z**

Second review mechanism: an SPD noise matrix can rotate a subtractively formed G eigenvector slightly into an exact incoming nullspace; componentwise FA product bounds alone cannot remove pre-existing eigenvector error. Implemented the identity Null(G)=Null(Lambda_o) for SPD W: proper Lambda_o checks rank(A), otherwise project through the retained incoming eigenspace with dtype-relative eigenspace uncertainty before equilibration. Maintained f32/f64 diagonal-null fixtures reproduce the old false admissions. Proper d2 isotropic matrix VJPs also exposed pre-existing NaNs from an unused repeated-eigenvalue Schur branch; that unselected branch now receives a distinct-spectrum dummy before eigendecomposition, never as returned information. Independent matrix Jacobian oracles retained.

**2026-09-25T08:46:24Z**

Final read-only review passed: randomized exact-null scans across64 SPD noise matrices in each dtype had zero false admissions/invalid results; singular mixed-scale incoming precisions admitted proper outgoing components; proper d2 matrix and precision gradients finite both directions. Congruence equilibration applies only to rank analysis. Focused validation:30 new rank cases; earlier53 mixture/collapse/hybrid tests,57 rank/collapse/neural tests and56 Gaussian/hybrid/readout/schedule tests passed during development. Independent covariance, nullspace and Jacobian oracles retained. Full short-suite gate follows final code.

**2026-09-25T09:14:18Z**

Complete: rank evidence and producer intermediate checks implemented with independent proper moments, rotated/aligned nullspaces, singular informed subspaces, congruence-rescaled units and scalar/matrix derivatives. Full final code gate2561passed/87deselected, dashboard wall1254.886s; cff-q6zc moves7 unchanged behavioral cases totaling59.300s, estimated default1195.586s, preserving all30 new numerical cases. No rerun after marker-only adjustment. Canonical architecture/derivations/limitations updated; no shared propriety cutoff or model covariance changes. Handoff: cff-u9un owns retained H15 integration, neighboring/f32/Colony/horizon checks; bounded research already finds undamped convergence16. PR is next link after#171.

**2026-09-25T09:14:51Z**

Published as stacked PR https://github.com/engimatic-systems/cffg/pull/172, base cff-03dn-checked-collapse (#171). Working tree clean after publishing; next ticket cff-u9un.
