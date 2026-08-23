---
id: cff-9skg
status: in_progress
deps: []
links: []
created: 2026-08-22T14:43:18Z
type: bug
priority: 1
assignee: coopwilliams
parent: cff-95ck
tags: [rslds, validation, numerics]
---
# Align matched SLDS returned-coordinate certification and evidence

## Background in Plain English — WHY

The matched SLDS control in the NASCAR experiment has the same returned-
coordinate defect that cff-95ck repaired for CFFG rSLDS. On its second
qualifying runtime-map residual, it installs the proposed parameter update,
breaks, and terminal-solves that unchecked proposal. The result can therefore
certify one parameter coordinate and return another.

The experiment's work accounting also treats only an installed replacement as
a completed map. After the core repair, a certified final map is complete even
though its proposal is deliberately not installed. Control fitting, budget
derivation, checkpoints, and aggregate validation must use the same coordinate
contract before another comparison is frozen.

## Outcomes / Acceptance Criteria — WHAT

1. A converged matched SLDS result returns beliefs, installed parameters, and
   learned posteriors from the evaluated coordinate that satisfied the unchanged
   0.1 runtime-map threshold for the second consecutive map.
2. The unchecked proposal from the certified map is not installed or returned,
   and its scale evidence remains available for diagnosis.
3. Exhausted and diverged controls preserve their terminal-state, retry,
   non-finite, and status behavior at the last installed coordinate.
4. SLDS phase and result schemas name evaluated, proposed, installed,
   certified, and returned coordinate identities with the same meanings as the
   CFFG rSLDS result.
5. Work accounting counts every completed runtime-map evaluation, including a
   certified final map whose proposal is not installed, and excludes the
   terminal hidden-state solve.
6. Selected-scale, classic-scale, outer-diagnostic, checkpoint, calibration,
   aggregation, and freeze validators reject inconsistent coordinate identities
   or map counts while retaining explicit compatibility for immutable schema-1
   artifacts.
7. Compact regressions cover rebound, exhaustion, non-finite proposals,
   float32, exact-zero support, and a long map sequence. No NASCAR-scale fit is
   required for the code gate.
8. A development-only replay remeasures control convergence and budgets before
   a fresh protocol is frozen; final seeds are rejected.

## Methodological Suggestions — HOW

These suggestions are optional. The implementer may choose another sound
method that satisfies the outcomes.

- Reuse the public coordinate-decision primitive from cff-95ck instead of
  duplicating the two-hit policy.
- Advance the current experiment schema and keep the old parser isolated for
  immutable artifacts.
- Update work consumers from phase role/identity, not from
  `coordinate_replaced` alone.

## Future work

`cff-95ck` closes after this experiment control follows the core coordinate
contract. A fresh calibration must remeasure both rSLDS and SLDS map budgets.
`cff-l5ag` and `cff-0oqo` remain separate scientific repairs.

## Definitions

- **Completed map:** One hidden-state evaluation followed by one complete
  undamped parameter proposal and runtime-map residual.
- **Returned coordinate:** The parameter values whose beliefs and posteriors
  are present in the public fit result.
- **Unchecked proposal:** A proposed replacement for which the next full map
  has not been evaluated.

## Notes

Architecture handoff: the sibling experiment's matched SLDS fit, phase
evidence, work accounting, checkpoint, budget, aggregation, and freeze
boundaries apply. The CFFG section 2d coordinate decision is reused but not
changed. CFFG IR, compiler, VM, schedules, kernels, ledger, factor equations,
and serialization are N/A.

Duplicated surfaces: same-K and closest-budget controls, converged/exhausted/
diverged results, phase serialization, every map-count consumer, historical
artifact validation, future calibration, and freeze must stay aligned.

Hostile-input decisions: exact-zero support remains exact; improper and
non-finite proposals fail loudly; float32 uses dtype-scaled guards; a synthetic
map sequence at least ten times the smallest fixture exercises accounting at
uncomfortable length.

Legend: prerequisite --> dependent

```text
[x] cff-95ck core rSLDS coordinate repair
      |
      +--> [ ] cff-9skg matched SLDS/evidence repair (ready)
                  |
                  +--> [ ] fresh calibration and freeze
```

**2026-08-22T15:19:02Z**

Experiment code integrated at commits 5c457b4 and 3bb5cd9. fit_slds now returns the certified evaluated coordinate, evidence serializes all coordinate roles, and map-count consumers count certified non-installed maps while excluding terminal inference. Immutable schema-1 parsing remains pinned and explicit. Root integration validation passed the affected 12-file selection and uv lock check. No fit, data, chain, calibration seed, or final seed ran. The ticket remains in progress until a development-only convergence cohort remeasures SLDS/rSLDS map budgets before freeze.

**2026-08-23T01:34:07Z**

Implementation handoff (2026-08-22): exact experiment base 552dcca already contained the core matched-SLDS returned-coordinate repair at ancestor 3bb5cd9. Independent review found and commit 0ef1875ba2ef8b5f2cf3173443907d31efe938a2 closes two remaining persisted-evidence gaps: current schema-2 budget consumers now require authoritative phase/fit work, bind coordinate identities to completed-map counts, and do not label inner exhaustion as outer right-censoring; live/serialized validators require certification and incomplete non-map phases to terminate the coordinate sequence. Selected scale, dual scale, calibration, checkpoint/diagnostic, aggregation, and freeze validation consume the stricter contract. Immutable schema-1 compatibility remains explicit. Verification: 71 focused coordinate/selected/calibration tests passed; 124 classic/diagnostic/checkpoint tests passed; an earlier broader affected selection passed 91 tests; uv lock check and git diff check passed; independent bounded post-fix review cleared with no blockers. No NASCAR-scale or development-cohort fit, adapter, calibration, final execution, or artifact rewrite was performed. AC1-7 are now code-complete. AC8 remains deliberately unsatisfied because this task prohibited the development convergence replay, so cff-9skg stays in_progress and cff-95ck replay/freeze remains blocked.
