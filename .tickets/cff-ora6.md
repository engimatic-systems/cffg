---
id: cff-ora6
status: closed
deps: []
links: []
created: 2026-08-18T16:50:22Z
type: task
priority: 1
assignee: coopwilliams
parent: cff-nkbj
tags: [rslds, validation, architecture]
---
# Complete executable classic rSLDS architecture mapping evidence

## Background in Plain English

The final rSLDS parity claim has two parts: the CFFG graph must represent the
classic NASCAR probability model, and it must perform roughly as well on the
registered task. Existing tests strongly compare the recurrent transition and
destination-indexed dynamics equations with independent implementations. The
initial categorical law, tied initial Gaussian, and shared emission are instead
supported by separate component tests and source inspection. That is good
engineering evidence, but it is weaker than the executable term-by-term mapping
required by cff-nkbj.

This ticket closes that narrow evidence gap. It does not change inference,
learning, generator parameters, or the final experiment protocol.

## Outcomes / Acceptance Criteria

1. One reviewable architecture table covers every term in the classic NASCAR
   joint distribution: initial regime, tied initial latent state, recurrent
   decision-list transition, destination-regime dynamics, and shared emission.
2. Each table row names the CFFG factor or graph composition, independent
   paper-equation adapter mapping, parameter conversion, and executable oracle
   or fixture that would fail if the probability law changed.
3. Direct numerical tests cover the mapped initial categorical probabilities,
   tied initial Gaussian parameters, and shared emission parameters and
   information-form likelihood, including their regime-sharing semantics.
4. The existing recurrent-transition and destination-dynamics oracles remain
   the authoritative executable evidence for those terms and are linked from
   the architecture table rather than duplicated.
5. The generator evidence distinguishes source correspondence from executable
   identity. The final wording does not call the local implementation the
   unmodified authors' generator unless a cross-runtime oracle proves that
   stronger statement.
6. A truth-boundary test or static interface check continues to prove that both
   learners receive observations and declared priors but not generating latent
   states, regimes, or parameters.
7. The resulting evidence supports only the source-homogeneous, shared-emission
   NASCAR subclass and explicitly excludes inference-algorithm identity,
   implementation identity, and general regime-specific emissions or initial
   laws.

## Methodological Suggestions

These suggestions are optional. The implementer may choose another sound method
that satisfies the outcomes.

- Extend the independent mapping tests in the sibling validation experiment so
  one fixture compares both sides of each missing parameter conversion.
- Keep a compact machine-readable mapping record next to the conversion code and
  render or cite it from the final CFFG architecture documentation.
- Use fixed canonical arrays and direct probability or information-parameter
  calculations. A long fit is neither needed nor useful for this ticket.

## Future work

cff-huxt freezes the task-performance protocol. cff-oyie executes it, and
cff-nkbj combines that bounded result with this architecture evidence. General
regime-specific initial and emission distributions remain future model work and
are not implied by classic NASCAR parity.

## Definitions

- **Architecture table:** A term-by-term correspondence between a probability
  model and the factors, parameters, and tests that implement it.
- **Classic NASCAR subclass:** The Linderman NASCAR model with recurrence rows,
  initial Gaussian laws, and emissions tied across source regimes.
- **Information-form likelihood:** A Gaussian likelihood represented by its
  precision matrix and precision-weighted mean.

## Notes

Pre-implementation handoff: `docs/ARCHITECTURE.md` change-path row is the
canonical rSLDS consumer-validation section. Core compiler, VM, schedule,
ledger, serialization, and factor ownership are N/A because this ticket adds
mapping evidence without changing their behavior.

Duplicated surfaces checked before implementation:

- initial categorical, tied initial Gaussian, decision-list recurrence,
  destination-indexed dynamics, and shared emission;
- CFFG model construction, paper-equation adapter construction, canonical
  generator mapping, and truth-free fit boundaries;
- float64 direct probability and information-form values; existing factor-level
  float32 and hostile-scale tests remain authoritative and are not weakened.

Hostile-input decisions:

- Sparse or zero support is represented exactly in the categorical mapping; the
  test must not replace zero probability with an arbitrary floor.
- Improper or vacuous messages are N/A because these are fixed-parameter mapping
  tests, not inference updates.
- Float32 inference behavior is N/A to this exact canonical mapping check; the
  existing numerical-robustness tests continue to own consumer dtype behavior.
- A 10--100x domain-size case is N/A because the claim is deliberately bounded
  to the canonical four-regime, two-latent, ten-observation NASCAR architecture.

Legend: prerequisite --> dependent

```text
[x] recurrence + destination-dynamics oracles
                  |
                  +--> [ ] cff-ora6 initial/emission mapping evidence (ready)
                                   |
                                   +--> [ ] cff-nkbj final parity audit
```

**2026-08-18T17:00:24Z**

Implementation handoff: experiment commit 61cec3f bumps the serialized architecture mapping to version 2, adds the five-term joint correspondence table, checks every initialization and prior array, preserves exact categorical zero support, and adds independent categorical, Gaussian-density, and emission-information oracles in the isolated reference runtime. The adapter provenance ledger now covers main-paper Eqs. 1-4 and the tied initial and emission surfaces. README wording distinguishes local generator correspondence from authors-code identity. Validation: modern experiment suite 110 passed in 68.32 seconds; isolated legacy suite 38 passed in 0.92 seconds; JSON and diff checks passed. No fit, generator, calibration artifact, or final seed was executed by this ticket.
