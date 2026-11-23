---
id: cff-c5qf
status: open
deps: [cff-u9un]
links: [cff-ulbz]
created: 2026-09-25T09:19:45Z
type: task
priority: 1
assignee: coopwilliams
parent: cff-xa6z
tags: [pcgym, planning, decision]
---
# Decide the supported contract for partial-goal hybrid planning

## Background in Plain English

The repaired PCGym hybrid model converges, but its first action is unchanged when the horizon grows from 1 to 15 or only the final goal changes. The final action and free energy do change, so observations are not being lost. A goal on position alone supplies no direct velocity information. The existing Gaussian-mixture rule cannot normalize those backward components and deliberately sends a message with no information.

This is a modeling and approximation limitation, not a remaining arithmetic or damping defect. Earlier full-tilt expectation-consistency work in cff-ulbz was closed as not pursuing at Cooper direction; it is not an implemented alternative and must not be silently revived. Cooper has been asked whether to retain the limitation or authorize a separate investigation.

## Outcomes / Acceptance Criteria

1. Record Cooper decision: retain the documented limitation or authorize a bounded partial-goal planning investigation.
2. Preserve an independent small-horizon oracle and a changed-distant-goal experiment that distinguish numerical convergence from a useful earlier-action response.
3. If retaining the limitation, document which benchmark claims are unsupported and keep the existing approximation explicit.
4. If investigation is authorized, compare mathematically sound alternatives and create separate derivation and implementation tickets with dependencies before changing production inference. Do not add artificial full-state preferences or covariance floors to make the experiment pass.
5. Record whether historical cff-ulbz assumptions can be reused, without treating its closed status as completed implementation.

## Methodological Suggestions

Another sound approach is acceptable. Start from the maintained numeric fixture and independent local moment/ledger oracle in cff-u9un. Consider support-preserving improper messages or an explicitly chosen alternative projection only after the decision in criterion 1. For future implementation, use the New factor/rule family and Runtime numerical/status architecture rows and the full numerical robustness contract. This ticket itself changes no solver.

## Future work

- cff-wt1c uses this decision to state meaningful matched-benchmark claims. New derivation/implementation IDs are created only if the investigation is authorized.

## Definitions

A partial goal constrains only part of the state. A backward message carries evidence from future factors toward earlier states. A vacuous message carries no information. An oracle is an independent correctness reference. Expectation consistency is an alternative moment-matching method; full-tilt means matching the combined local factor and neighboring information before projecting.

## Notes

**2026-09-25T09:20:30Z**

Dependency snapshot from tk dep tree --full, tk ready and tk blocked; no cycles. Legend: prerequisite --> dependent.

[x] cff-vxza + cff-03dn + cff-04mb
  --> [in progress] cff-u9un (ready numerical integration)
      --> [blocked] cff-c5qf (also needs Cooper scope decision)
          --> [blocked] cff-wt1c --> cff-xa6z

Parallel ready roots include cff-7kpx nonlinear reproduction, cff-qu2j float32 exact certification, cff-ees0 finite hypothesis banks, and cff-nd1q bounded-action derivation. Historical cff-ulbz is a reference link only, not a completed prerequisite or an implicit revival.
