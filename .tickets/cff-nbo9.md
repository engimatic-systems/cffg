---
id: cff-nbo9
status: in_progress
deps: [cff-rk7i]
links: []
created: 2026-08-22T15:16:47Z
type: task
priority: 1
assignee: coopwilliams
parent: cff-0oqo
tags: [rslds, inference, validation]
---
# Locate recurrence responsibility bias with exact chain smoothing

## Background in Plain English — WHY

The cff-rk7i clamping ladder showed that generating joint regime-transition
responsibilities plus inferred continuous moments recover the heterogeneous
transition functions. Inferred responsibilities still fail when they are
paired with generating continuous states. The local cavity-based recurrence
joint differs from a product of marginals, but it also fails to restore
recovery.

This localizes the error to categorical recurrence inference, but it does not
yet say whether a message rule is wrong or the current edge-factorized
variational family is too weak. A compact chain with an exactly enumerable
regime posterior can compare edge marginals, factor-local joints, and globally
smoothed consecutive-regime joints before the production architecture changes.

## Outcomes / Acceptance Criteria — WHAT

1. An independent enumerator computes normalized single-regime and consecutive-
   regime posteriors for compact K=2 and K=3 recurrent chains with fixed
   continuous drives and observations.
2. The same models run through CFFG and compare categorical messages, edge
   beliefs, recurrence factor-local joints, and recurrence learning statistics
   with the enumerated authority under a named finite tolerance.
3. A correlated case proves that a product of edge marginals is not an
   acceptable substitute for the consecutive-regime posterior.
4. A second oracle introduces uncertain Gaussian drives and separately reports
   error from the Pólya-Gamma/Jaakkola-Jordan approximation and error from the
   categorical factorization.
5. The report issues one falsifiable ownership decision: a local categorical
   message defect, a learning-readout defect, or a variational-family limitation.
6. Any proposed production change states whether it preserves the existing
   CFFG recurrence factor's fixed point and Bethe free energy or introduces a
   structured consecutive-regime belief with a new derivation.
7. Exact-zero transition support, deterministic and vacuous messages,
   non-finite inputs, float32, K at least 8, and chain length at least 100 follow
   the numerical robustness contract.
8. The enumerator, model templates, parameters, tolerances, CFFG identity,
   numerical mode, results, and replay command are manifested. No NASCAR final
   or calibration seed is used.

## Methodological Suggestions — HOW

These suggestions are optional. The implementer may choose another sound
method that satisfies the outcomes.

- Start with fixed observed drives so the categorical chain is the only latent
  subsystem. Enumerate all K^T paths at a T small enough for an exact sum.
- Reconstruct the factor-local joint from incoming cavities and the recurrence
  potential, then compare it with both exact smoothing and the production
  product readout.
- Use quadrature or exact finite support for the uncertain-drive oracle rather
  than treating a generating drive value as inferred evidence.
- Do not implement a structured recurrence factor until the oracle identifies
  the missing dependency and its free-energy consequence.

## Future work

If a local rule or readout is wrong, create a core bug ticket that corrects it
and add the enumerated model as its regression. If the existing variational
family is the limitation, `cff-0oqo` must choose between a derived structured
belief repair and narrowing the heterogeneous recovery claim. Fresh calibration
and final evidence remain downstream.

## Definitions

- **Exact chain smoothing:** Enumeration or forward-backward computation of
  posterior probabilities over consecutive states using the whole sequence.
- **Factor-local joint:** A joint belief reconstructed from one factor's
  potential and its incoming cavity messages.
- **Variational family:** The set of approximate joint distributions that the
  inference method is allowed to represent.

## Notes

Architecture handoff: use `docs/ARCHITECTURE.md` section 2d's recurrence
message, energy, `gaussian_node_stats`, and `recurrence_weight_stats` boundaries.
The BFE and message-rule change paths apply if the oracle finds a production
defect. Compiler, VM, scheduling, kernels, ledger, and serialization are N/A
for a diagnostic-only implementation.

Duplicated surfaces: out/in categorical roles, local and chain-smoothed joints,
fixed and uncertain drives, message/readout/energy consistency, exact zeros,
float32/x64, K=2/K=3 oracles, and uncomfortable K/T cases must stay aligned.

Hostile-input decisions: literal support zeros remain zero; deterministic and
vacuous messages are explicit; non-finite values fail loudly; tolerances derive
from dtype precision; K=8 and T=100 exercise scale without exhaustive K^T
enumeration.

Legend: prerequisite --> dependent

```text
[x] cff-rk7i responsibility clamping diagnosis
      |
      +--> [ ] cff-nbo9 exact pairwise oracle (ready)
                  |
                  +--> [ ] recurrence inference/readout repair
                              |
                              +--> [ ] cff-0oqo recovery evidence
```
