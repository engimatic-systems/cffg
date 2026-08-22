---
id: cff-ngxx
status: in_progress
deps: [cff-xwtz]
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
