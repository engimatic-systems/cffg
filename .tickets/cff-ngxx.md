---
id: cff-ngxx
status: in_progress
deps: [cff-xwtz, cff-6rav, cff-xq4y]
links: []
created: 2026-08-22T19:52:31Z
type: bug
priority: 1
assignee: coopwilliams
parent: cff-0oqo
tags: [rslds, inference, learning, trustworthiness]
---
# Correct coupled rSLDS variational learning bias

## Background in Plain English — WHY

The D9 source-heterogeneous seed-5000 native fit still misses the unchanged 0.150 recovery ceilings. cff-xwtz now rules out the two local categorical mechanisms that were the leading suspects: after a fixed-parameter refinement, CFFG singleton, consecutive-pair, and both-direction categorical messages match an independent exact chain smoother to at worst 1.40e-9 under a 5.96e-8 tolerance, and the production dynamics-selector message matches an independent Gaussian integral to 4.64e-11.

Replacing inferred continuous beliefs with the generating continuous trajectory and actually rerunning categorical smoothing does not restore recovery while learned parameters remain installed: worst-source and source-contrast errors remain 0.203 and 0.309. The same exact smoother with generating dynamics and recurrence parameters passes at 0.033 and 0.040. The remaining owner is therefore the coupled variational fit that learned biased parameters, not D9 pair algebra or label/affine alignment.

## Outcomes / Acceptance Criteria — WHAT

1. Authenticate the manifested cff-xwtz seed-5000/native post-fit artifact before using development truth or executing any new coordinate.
2. A pure cross-parameter exact-smoothing oracle holds generating continuous states fixed and separately substitutes learned versus generating initial-regime, dynamics, and recurrence parameter blocks. It reports regime accuracy, pair-count error, source/gate exposure and signed-success error, and the unchanged 0.150 recovery scores for every declared block combination.
3. The first parameter block whose generating substitution restores both caps owns the next production change. If no single block restores both caps, preserve the interaction result and name the smallest multi-block coupling that does. Do not select an owner from effect size alone.
4. For the owned block, independently reproduce the production learning natural parameters from the same factor-local beliefs and distinguish an algebra/readout defect from an internally correct but biased assumed-density or mean-field objective.
5. The production repair, if an algebra defect is found, has a compact oracle that fails before and passes after the change. If the current objective is internally correct, derive and document the smallest reusable structured hybrid variational boundary before changing engine behavior; do not relabel an approximation limitation as an implementation bug.
6. Any behavior change preserves D9 source-row/destination-column pairs, exact-zero support, pair-derived Gaussian/learning masses, BFE ownership, synchronous/scheduled parity, float32, non-finite failure, and a K=8, T>=100 non-cubic path.
7. A prospective truth-free checkpoint version distinguishes the changed inference/statistic semantics from current D9 evidence. Historical artifacts remain immutable and cannot resume under changed semantics.
8. Only after compact oracles pass, rerun the registered source-heterogeneous seed-5000/native development pilot under the frozen 0.150 caps. Stop and preserve evidence on the first status/cap failure; do not run adapters, SLDS controls, calibration seeds, or final seeds.

## Methodological Suggestions — HOW

These suggestions are optional. The implementer may choose another sound method.

- Start with four truth-state exact-smoothing arms: all learned; generating dynamics only; generating recurrence plus initial-regime only; and all generating. Add one-at-a-time initial-regime isolation if the recurrence substitution changes the first boundary.
- Compare component-MNIW dynamics statistics against an independent Gaussian joint integral before proposing a larger structured hybrid region.
- If each local statistic is algebraically correct but the block interaction remains biased, derive a named structured variational family whose continuous and categorical responsibilities share one objective. State its bound and free-energy ownership before implementation.

## Future work

A passing repaired development pilot unblocks cff-h0hx requalification. Calibration and final parity evidence remain separate downstream work.

## Definitions

- **Cross-parameter oracle:** A post-fit diagnostic that swaps one learned parameter family for its known development generating value while holding the data and other families fixed.
- **Assumed-density inference:** Replacing a mixture distribution with a single Gaussian moment match so inference remains compact.
- **Coupled variational bias:** A parameter error caused by optimizing an internally consistent approximation whose categorical and continuous beliefs cannot represent the exact joint posterior.

## Notes

Architecture handoff: inspect docs/ARCHITECTURE.md rows for gaussian_mixture_linear assumed-density messages, structured recurrence categorical inference, component-MNIW and PG learning, rSLDS result construction, persistence identity, and the experiment consumer. Ordinary categorical BP, adapters, SLDS controls, calibration, and final execution are N/A until a compact owner is proven.

Hostile-input decisions: exact categorical zeros stay exact; improper/non-finite Gaussian or categorical evidence diverges loudly; tolerances scale with dtype; K=8 and T>=100 must remain O(T*K^2) on the categorical side and must not enumerate switching paths.

Falsifier: this ticket is not owned by coupled variational learning if the cross-parameter oracle shows that learned parameters with generating continuous states pass both 0.150 caps, or if an independent local-message/statistic oracle finds a bounded algebra mismatch large enough to explain the recovery error.

Legend: prerequisite --> dependent

```text
[x] cff-xwtz exact categorical and clamped-state diagnosis
      |
      +--> [ ] coupled variational learning repair (this ticket)
                  |
                  +--> [ ] cff-h0hx fresh development requalification
                              |
                              +--> [ ] cff-0oqo bounded recovery
```


**2026-08-22T19:54:11Z**

Implementation start from the authenticated cff-xwtz checkpoint. The first phase is diagnostic-only: enumerate the predeclared learned/generating initial, dynamics, and recurrence block substitutions with generating continuous states held only inside post-fit evaluation. No fitting or production change begins until a single block or smallest block interaction restores both frozen recovery caps and its local learning statistics are independently audited.

**2026-08-22T20:23:24Z**

Diagnostic handoff (2026-08-22): authenticated post-fit evidence is committed in experiment branch agent/cff-ngxx-cross-parameter at 1799f4d. No fit, adapter, control, calibration seed, or final seed ran.

The complete learned/generating initial x dynamics x recurrence cube used source_heterogeneous seed5000/native/replicate0 and the existing D9 checkpoint. All-learned failed: worst-source 0.202719, worst contrast 0.309087, M-step budget_exhausted at 160. Initial-only substitution was unchanged and failed. Two one-block substitutions independently restored both 0.150 caps: generating recurrence with learned dynamics scored 0.038173/0.043341 and converged at 145 coordinates; generating dynamics with learned recurrence scored 0.032903/0.040220 and converged at 134. Full generating scored 0.033182/0.039697. The minimal owner is therefore the dynamics-recurrence boundary; neither block alone is uniquely selected.

Independent local audit then reconstructed fixed-parameter beliefs without fitting. D3 responsibility/Syy/Syx/Sxx and component-MNIW posterior hyperparameters all met dtype-scaled tolerances across 5,119 dynamics factors. D9 exposure/linear/quadratic and PG eta/precision/posterior mean all met tolerance across 5,119 recurrence factors. Largest normalized local differences were 4.64e-11 for D3 factor statistics and 3.35e-16 for D9 factor statistics. This falsifies a bounded local algebra/readout repair and selects the internally correct but coupled-objective-biased classification. D9 remains in place.

Artifacts: cross results sha256 5faa01c9b6eb052b8fa088096cc4c9c0d34ed80c014f3160907d9b9d7b87ab76, manifest 28932488884ce5c59fbebeea2b52828bc6c5b0e4f9589d77831286da6f6aaf90; local audit results 153fa4ed93326270cead6ea9c8f87cb03739d96b5189389695f52350ab187816, manifest a73b646d27dc208d65fef644206de511f7c587a9bf690d0556302b7912ea399f. Cross wall/RSS 15.42s/634,814,464 bytes; local reconstruction/audit 103.78s/2,005,106,688 bytes.

DERIVATIONS.md D10 now derives the smallest reusable repair: one q_Z categorical chain and one q_X Gaussian Markov trajectory, with expected dynamics unaries, pair-derived recurrence terms, singleton-weighted dynamics learning, one conservative objective, and explicit D3/D9 ledger suppression. Production behavior is intentionally unchanged.

Legend: prerequisite --> dependent

[x] cff-ngxx authenticated diagnosis (this handoff)
      |
      +--> [ ] cff-bmax pure D10 coordinate authority (ready)
                  |
                  +--> [ ] cff-e10u compiler/engine dispatch
                              |
                              +--> [ ] cff-6rav learning/persistence identity
                                          |
                                          +--> [ ] cff-ngxx seed5000/native pilot
                                                      |
                                                      +--> [ ] cff-h0hx requalification

cff-ngxx now depends on cff-6rav and remains in_progress. Resume here only after the three implementation tickets land; then run the already registered development pilot and stop on the first status/cap failure.

**2026-08-23T00:26:28Z**

Core D10 authority, dispatch, four-family learning, fit coherence, and schema-v2 persistence are integrated through active CFFG commit 4b9f9e3. The remaining acceptance gate is a fresh prospective seed-5000/native D10 pilot under the unchanged 0.150 caps. The experiment must pin that integrated descendant, require exact four-family D10 dispatch and schema-v2 topology/observation/statistic identities before generation, use damp=0 and D10 work semantics, and reject every D9 checkpoint rather than importing or relabeling it.

**2026-08-23T01:08:56Z**

Fresh prospective D10 pilot falsifier (2026-08-22). Experiment code commits 137cd3b and e485bed; immutable artifact commit 0ccf659. Root results SHA-256 50a814dfe38a77f58b60d738268614e3a1296f61075256afa842888ec76d6c99; root manifest SHA-256 3e61a5a4c705069b0bd0b249cf2a56a0c75f90f3794c02e2cbd335ae644a85d0. Checkpoint JSON/NPZ SHA-256 b1ab8def2b71f2c0b45cfcdfb490e61835c32087a049e8f1ce64818e841d1267 / cb2bff6005948eb2bebeb25479db0dc77adba94aca95a0ac5d49d83dba30aba4. Both manifests and the registered validator pass.

Only source_heterogeneous seed5000/native_end_to_end/replicate0 ran under CFFG 4b9f9e3, exact four-family D10 schema v2, T6400/train5120, prior .001, outer cap19, inner cap4000, damp0, retries0, and registered streams 1500000000/1500040200/1500040400. The fit converged in 8 outer maps and 295 total inner iterations; all 8 hidden solves converged in one undamped attempt, certified and returned coordinate/7 agree, final runtime residual is 0.087052, wall is 245.83 seconds, and peak RSS is 5,111,627,776 bytes. No shared tier, expansion, adapter, control, calibration seed, or final seed ran.

The unchanged recovery gate failed: worst-source RMSE 0.206239 and worst source-contrast RMSE 0.317652 versus 0.150. The D9 pilot was 0.204837/0.318117 from the same observations and initialization, so D10 changed neither claim-bearing error materially. D10 pair and pooled statistics agree within tolerance, posterior replacement is coherent, alignment is well-conditioned, and failing source/gate cells are well covered. This falsifies D9 factorization as the sole heterogeneous-recovery bottleneck, but it does not yet distinguish a shared local optimum from a limitation of the D10 variational family.

Decision memo, ranked by scientific reach:

1. Richer joint q(z,x), such as a switching Gaussian-mixture trajectory. This is the strongest route if the no-fit discriminator says the D10 objective itself disfavors recovery-compatible points. A passing development cohort could support the heterogeneous extension claim, but this is a new inference family with new objective, persistence, and scaling proof obligations.
2. Alternate initialization or bounded global search under the unchanged D10 objective. This is narrower and appropriate only if a recovery-compatible retained parameter point has a better D10 objective or its update direction leads away from the current terminal. A pass would support heterogeneous capability under a registered initialization policy, not universal optimizer robustness.
3. Stop the heterogeneous extension and proceed with classic homogeneous parity only. This preserves the narrower original-paper architecture/task claim while explicitly leaving the stronger source-heterogeneous capability unsupported.

Smallest no-fit discriminator before choosing A or B: authenticate the existing D9 and D10 seed5000 checkpoints; at the D10 learned point and the already declared generating-dynamics, generating-recurrence, both-generating, and bounded label-aligned interpolation points, run fixed-parameter D10 inference, evaluate the same normalized D10 objective, and compute but do not install one M-step direction. If a recovery-compatible point has a lower D10 objective or a direction leading away from the learned terminal, local basin/global search remains plausible. If the converged learned point has the best D10 objective and every recovery-compatible point is worse while local statistics stay coherent, the D10 family/objective is the likely limiter. Estimated wall time is 15-30 minutes with roughly the pilot memory footprint; main risk is an ambiguous objective path, which would require a user decision before any alternate-initialization fit.

Keep D10 in production. It repairs formal coordinate, statistic, objective, and persistence ownership and passes independent engine oracles. Reverting would restore known D8/D9 inconsistency without improving the measured recovery. Leave cff-ngxx in_progress pending the user/root choice; do not tune or launch another fit.

**2026-08-24T05:49:02Z**

Objective-path handoff (2026-08-23). Authenticated evidence is experiment commit 16399ae9d0b7f23da9d2cef5fa61db4dcd48600a from reviewed code 6a022fccbd57f5a62dc2a8d4566b446be20fc2be. All 13 fixed-parameter D10 points converged twice with coherent statistics, but the registered outcome is ambiguous because the recurrence chart projection exceeded its 0.05 fidelity ceiling at 0.285718. The two recovery-compatible points passed at 0.051243 worst-source and 0.047136 worst-contrast error, yet neither beat the learned training objective; paired held-out simultaneous intervals also crossed zero. This does not justify an initialization/global-search run, and it does not establish that the D10 family is the limiter. D10 remains the production authority and should not be reverted. Decision paths: A, measure a direct bound gap on the authenticated fixed checkpoint, the smallest no-outer-fit discriminator of approximation quality; this could support or weaken a variational-family diagnosis but would not itself recover heterogeneous functions. B, design a richer joint q(z,x) switching-Gaussian family; this broadens the architecture and requires new objective, persistence, hostile-input, scaling, and requalification obligations before any parity claim. C, defer the heterogeneous extension and proceed with the faithful classic parity claim only, explicitly excluding source-heterogeneous recovery. Keep cff-ngxx in progress pending that user decision; run no further fit under this handoff.
