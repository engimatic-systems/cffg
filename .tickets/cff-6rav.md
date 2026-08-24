---
id: cff-6rav
status: closed
deps: [cff-e10u]
links: []
created: 2026-08-22T20:21:34Z
type: task
priority: 1
assignee: coopwilliams
parent: cff-ngxx
tags: [rslds, learning, persistence, trustworthiness]
---
# Version D10 learning and checkpoint semantics

## Background in Plain English — WHY

cff-e10u installs one D10 categorical/Gaussian coordinate for an eligible
recurrent switching component. Learning and persistence must use that exact
state. Reusing the old D3 component responsibilities or accepting a D9-only
checkpoint would silently give the new inference path old learning semantics.

This ticket makes the semantic boundary explicit. It preserves historical
D8/D9 evidence as immutable history and prevents it from resuming as D10.
Model-only templates contain no learned runtime state and may run under the
current documented implementation after recompilation.

## Outcomes / Acceptance Criteria — WHAT

1. Component-MNIW dynamics learning uses only D10.5 singleton-weighted
   Gaussian marginal and lag moments. PG recurrence learning uses only D9.14
   pairs and moments from the same installed D10 coordinate. Initial-regime
   and emission learning read that coordinate as well.
2. Independent calculations reproduce every pooled natural-parameter block
   before update. Exact-zero singleton or pair mass retains the applicable
   immutable prior coordinate. Improper, non-finite, asymmetric, or
   shape-inconsistent evidence fails before replacement.
3. `Learnables`, `RSLDSLearning`, `fit_rslds`, accumulation, replacement,
   tied sources, factor renaming, and atomic parameter writes all use one D10
   statistic authority without double counting a factor occurrence.
4. A new executable/checkpoint identity distinguishes D10 inference,
   objective, component-MNIW statistics, PG statistics, and source ownership
   from D9. Checkpoints round-trip dtype, shapes, values, learnable
   hyperparameters, installed-coordinate identity, and component topology.
5. A D8, D9, missing, unknown, wrong-factor, wrong-source, wrong-shape,
   non-finite, or tampered learned/runtime checkpoint cannot resume D10 by
   default. Any named migration must preserve its old meaning and may not
   relabel old statistics as D10.
6. Model-only templates and bundles with no posterior or resumable runtime
   state remain loadable and compile under the D10 semantics implemented at
   this ticket's proposed head.
   The unchanged serialized structure hash is not accepted as an executable
   statistic identity.
7. The rSLDS fit result preserves coherent installed parameters, D10 beliefs,
   objective/status traces, and evaluated/proposed/installed/certified/returned
   coordinate roles. Outer convergence remains a separate certificate and
   cannot convert an inner D10 failure into success.
8. Focused persistence, learning, and rSLDS tests cover float32, exact zero,
   non-finite evidence, K=8/T>=100 readout, and historical compatibility. The
   final CFFG short suite passes at the proposed integrated head.

## Methodological Suggestions — HOW

These suggestions are optional. The implementer may choose another sound
method.

- Version the smallest checkpoint boundary that can resume inference or
  learning. Do not broaden generic serialization for model templates that
  contain no runtime state.
- Make D10 readout arrays explicit in the checkpoint rather than inferring
  lag moments or pairs from endpoint marginals.
- Keep recurrence source rows and destination columns in the existing D9
  order. Give D10 dynamics statistics a separate named identity even when
  array shapes match historical D3 statistics.
- Update existing README, ARCHITECTURE.md, and LIMITATIONS.md sections in the
  same commits as the behavior they describe.

## Future work

When this ticket lands, cff-ngxx can run the registered
source-heterogeneous seed-5000/native development pilot against the unchanged
0.150 caps. A passing pilot unblocks cff-h0hx requalification. Adapters, SLDS
controls, calibration seeds, and final seeds remain outside this ticket.

## Definitions

- **Executable identity:** A versioned name for the inference and statistic
  behavior that produced resumable state.
- **Model-only template:** Serialized graph structure and parameter inputs
  with no posterior beliefs or learned runtime state.
- **Resume:** Continue inference or learning from stored runtime evidence
  instead of recompiling and starting a new run.

Architecture handoff: use both **New derived component coordinate** and
**Learnable or checkpoint feature** rows in `docs/ARCHITECTURE.md`. Current and
legacy decoding, learning sources, readouts, atomic writes, fit-result
construction, and model consumers apply. New IR spelling, compiler topology,
synchronous/scheduled numerical dispatch, and GFE are N/A here because
cff-e10u owns them.

## Notes

**2026-08-22T22:24:02Z**

Implementation start after fixed-parameter D10 dispatch passed the full suite and independent review. Learned D10 eligibility must land atomically with D10-owned readouts for all four learned families, fit evidence, parameter-write invalidation/rebuild, retry semantics, and versioned persistence. Until that complete boundary passes, learned templates remain on D3/D9.

**2026-08-23T00:25:27Z**

Completed D10 learning, fit, and persistence integration.

Implementation and evidence:
- Production implementation spans dffca82 through 3acc544; final validated repository head before ticket closure is d7288f2.
- Exactly eligible four-family rSLDS templates use one installed D10 coordinate for initial singleton, dynamics singleton/lag moments, recurrence source-row/destination-column pairs, and emission moments. Partial ownership remains on the explicit D3/D9 fallback.
- All point-parameter write paths invalidate/rebuild D10. RSLDSLearning.install now rejects a converged or otherwise advanced D10 coordinate even though legacy message arenas remain untouched.
- Fit attempt evidence distinguishes proper budget exhaustion from divergence. A failure before the first learned proposal returns coherent initializer state/points with posterior fields absent, so it cannot be resumed as learned evidence.
- Schema 2 binds executable cffg.structured-hybrid.d10/1.0.0, objective cffg.structured-hybrid.d10-jj-bfe/1.0.0, coordinate cffg.structured-hybrid.singleton-pair-gaussian-lag/1.0.0, four named statistic identities, full source/topology ownership, observation dimensions, dtype/shapes, hyperparameters, and exact value digests.

Compatibility matrix:
- Model-only templates/bundles: load and rerun under current D10 dispatch when topology is eligible.
- Current D10 learned state: round-trips only against matching schema-2 executable/statistic/topology ownership.
- Current non-D10 D9 recurrence state: remains schema 1 and may resume only under matching D9 semantics.
- D8, pre-manifest, D9-as-D10, missing, unknown, wrong factor/source/topology/observation dimension, wrong dtype/shape/value, non-finite, and tampered learned state: rejected before inference or replacement.
- Restricted Learnables cannot cross the D9/D10 statistic boundary in either direction.

Independent review found and verified fixes for attempt-evidence status, observation-domain checkpoint identity, exact-zero MNIW retention, D9/D10 cross-engine mixing, coordinate-0 posterior coherence, true-float32 probability and locally scaled moment consistency, arbitrary factor renaming, and D10 install freshness. Two reviewers cleared exact ed714f1 after hostile d=2/10/30, K=8/T=100, non-finite, one-sweep, rename, and tiny-oracle checks.

Validation:
- Focused affected surface before final repair: 173 passed, 12 deselected.
- Final repaired recurrence/D10/checkpoint/fit surface: 57 passed, 1 deselected in 153.79 seconds.
- Final project short suite at ed714f1: 1,520 passed, 40 deselected, zero failures in 703.843 wall seconds. Dashboard refreshed and recorded at d7288f2.
- No NASCAR fit, experiment artifact, adapter, SLDS control, calibration seed, or final seed ran.

Downstream handoff for cff-ngxx then cff-h0hx:
- Pin an integrated descendant containing production commit 3acc544 and bind the complete schema-2 D10 identity before any data generation.
- Do not relabel, import, or resume the historical D9 h0hx seed-5000 checkpoint; it remains immutable evidence and a fresh registered D10 development fit is required.
- Prospective experiment checkpoints must retain authenticated D10 singleton/pair beliefs, Gaussian mean/covariance/lag moments, pair-derived recurrence masses, all four learned posterior families, observation/config/stream/code identities, and coherent phase_work coordinate roles.
- Reject missing/D8/D9/product-marginal or topology-mismatched evidence before generation. Run only the already registered seed-5000/native development pilot first, preserve a falsifier and stop on status/cap failure, and run no adapter, control, calibration, or final seed under this handoff.

**2026-08-24T14:05:40Z**

Four-family D10 learning and persistence are published for review in draft PR #93: https://github.com/engimatic-systems/cffg/pull/93. Final core gate recorded 1,520 passed; scientific recovery remains a separate failed development gate.
