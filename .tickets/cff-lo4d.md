---
id: cff-lo4d
status: in_progress
deps: [cff-upih]
links: []
created: 2026-08-22T15:41:25Z
type: task
priority: 1
assignee: coopwilliams
parent: cff-l5ag
tags: [rslds, validation, statistics]
---
# Bind the fixed adapter initializer and requalify both-family mixing

## Background in Plain English — WHY

The cff-upih crossed study held the dataset, recurrence prior, Gibbs streams,
kernel, warmup, and retained-draw policy constant. Its observation-mapped
initializer produced four eligible heterogeneous chains, while the legacy
native initializer produced one accurate chain out of four and persistent
component collapse in another. This establishes initialization as the cause of
that native-tier failure and authorizes the staged
`pca_arhmm_mapped_decision_list_v1` policy.

The fixed policy cannot be inserted while continuing to call the reference arm
`native_end_to_end`: it supplies external observation-derived arrays and is no
longer the legacy adapter's per-chain native initialization. The prospective
protocol must describe the actual initialization, bind its artifact identity,
and re-establish mixing on both dataset families before a new calibration or
freeze can rely on it.

## Outcomes / Acceptance Criteria — WHAT

1. The prospective comparison protocol names the fixed observation-only
   adapter initialization policy and records its policy, manifest, family-
   specific named-array, data, seed, and numerical-runtime identities.
2. Within each family/dataset coordinate, every declared adapter chain receives
   the identical staged initialization arrays generated once from that
   coordinate's observations; no chain is selected, restarted, dropped, or
   initialized using generating states or parameters.
3. Tier names and report text accurately distinguish the two CFFG initialization
   streams without claiming that the reference adapter still performs native
   per-chain initialization.
4. One preregistered development gate exercises classic and source-heterogeneous
   data at their declared scales. For each family and tier, all four chains have
   no persistent component collapse and every family-authoritative variable has
   split R-hat at most 1.05 and effective sample size at least 50.
5. The paper Eq. 2, 4, 6, and 7 density/transition oracles remain unchanged and
   passing. Raw parameter diagnostics remain secondary, and classic
   counterfactual balanced-grid variables remain non-authoritative.
6. Checkpoint and resume validation reject a changed initializer artifact,
   family, tier, chain stream, sampler budget, reference source, or numerical
   environment before sampling. Calibration and final seeds are rejected before
   data generation.
7. The development artifact retains all chains and reports completed scientific
   misses without silently relaxing thresholds. It authorizes later calibration
   only if every declared cell passes.

## Methodological Suggestions — HOW

These suggestions are optional. The implementer may choose another sound method
that satisfies the outcomes.

- Extend the existing staged adapter-only runner instead of coupling this gate
  to a CFFG or control fit.
- Use a short funnel on one already-declared development coordinate, then expand
  only after its orchestration, mixing, occupancy, and provenance checks pass.
- If the classic held-out functional remains the sole marginal miss, extend a
  preregistered retained-draw budget rather than changing the Markov target or
  weakening the diagnostic threshold.
- Preserve historical cff-oyie and prior-screen schemas as immutable evidence;
  introduce a prospective schema and protocol identity.

## Future work

`cff-l5ag` consumes the passing gate and decides the adapter repair. A fresh
dual-family calibration, immutable freeze, and new final seeds remain separate
downstream work. This ticket does not change CFFG inference or repair
heterogeneous CFFG recurrence recovery.

## Definitions

- **Fixed observation-only initializer:** One deterministic set of starting
  arrays computed per family/dataset coordinate from its observations and a
  registered seed, then supplied unchanged to every chain in that coordinate.
- **Family-authoritative variable:** A mixing diagnostic whose estimand is
  identified by the named dataset family and allowed to control eligibility.
- **Development gate:** A preregistered non-final experiment that decides
  whether a configuration may advance to calibration.

## Notes

Architecture handoff: the sibling experiment's isolated reference-adapter,
initialization-policy, protocol, checkpoint, retained-draw, diagnostic, and
artifact boundaries apply. CFFG factor equations, IR, compiler, VM, schedules,
kernels, ledger, learning, and serialization are structurally irrelevant.

Duplicated surfaces: both dataset families, both CFFG tiers, all four Gibbs
streams, the nested Python 3.11 runner, initialization artifacts, checkpoints,
family-qualified diagnostics, component occupancy, aggregate authorization,
and report wording must remain aligned.

Hostile-input decisions: an initially empty component, exact-zero transition
support, saturated logits, a non-finite retained draw, a changed fixed-policy
digest, and one chain in a distinct basin fail loudly. The reference runtime is
locked to float64, so float32 is structurally irrelevant. Classic T=12,500 and
heterogeneous T=6,400 are the uncomfortable-scale cases.

Legend: prerequisite --> dependent

```text
[x] cff-w42g family-qualified mixing authority
      |
[x] cff-upih crossed initializer diagnosis
      |
      +--> [ ] cff-lo4d fixed-policy integration and gate (ready)
                  |
                  +--> [ ] cff-l5ag adapter repair
                              |
                              +--> [ ] fresh calibration and freeze
```
