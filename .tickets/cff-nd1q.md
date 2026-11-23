---
id: cff-nd1q
status: closed
deps: []
links: []
created: 2026-09-25T03:46:26Z
type: task
priority: 1
assignee: coopwilliams
parent: cff-xa6z
tags: [pcgym]
---
# Derive a bounded continuous-decision contract with an independent planning oracle

## Background in Plain English

The experiment plans with unconstrained Gaussian actions and clips the selected action at actuator limits. Clipping one action is not solving the constrained planning problem. The reward difference motivates a capability, but does not establish that bounds explain the entire gap.

Evidence comes from the PCGym report at `/Users/ravd/agent-share/PCGymCFFG/` and experiment code at `/Users/ravd/experiments/cffg-testing/pc-gym/00/cffg/`, reviewed against CFFG `fce60bd`. Copy only the minimal reproducible inputs into maintained fixtures; do not add external training artifacts or pickle dependencies.

## Outcomes / Acceptance Criteria

1. The design specifies the first supported continuous bound family, correlation assumptions and decision readout semantics.

2. Messages, normalizers, entropy and free-energy terms are derived consistently; any projection or truncation approximation is named.

3. A small box-constrained linear-quadratic planning oracle distinguishes bounded planning from post-hoc clipping.

4. The design defines infeasible/zero-width intervals, tail probabilities, improper/vacuous incoming information, float32 and larger action/horizon dimensions.

5. A matched controller ablation fixes dynamics, observations, objective, horizon and information access so reward effects are attributable to bounds.

6. DERIVATIONS gains a plain-English companion and executable mathematical anchors; the implementation ticket receives explicit supported/rejected cases.

## Methodological Suggestions

Choose the simplest mathematically valid v1 contract; do not assume independent clipping preserves correlated planning beliefs. Architecture: New factor or rule family and Runtime numerical or status invariant. Pairs host/JIT moments/normalizers, flooding/scheduled factor placement, single/vmap local rule, checkpoint N/A until new persisted identity is specified.

These are suggestions; another method is acceptable if it satisfies the outcomes. Work branches continue from the preceding validated PR in the PCGym chain, whose initial base is `cff-ucvx-readme-capability-map` (PR #161). Update the intended base before branching. Code changes require focused tests and one passing final short suite; documentation-only work validates its changed artifacts.

## Future work

- cff-fl7h — Implement bound-aware continuous planning from the approved derivation.

## Definitions

Box bounds: a lower and upper limit on each coordinate. Truncation: restricting a distribution to an allowed region and renormalizing. Normalizer: probability mass used to normalize that restriction. Linear-quadratic: linear dynamics and a quadratic cost. Ablation: a comparison changing one component.

Shared terms: CFFG means constrained Forney-style factor graph; JAX is the numerical array/compiler library; dtype is a numeric storage type; free energy is the model's inference objective; host/JIT compares Python-side and compiled execution; single/vmap compares one instance with vectorized instances; N/A means not applicable.

## Notes

**2026-11-22T03:39:42Z**

Architecture rows New factor or rule family and Runtime numerical or status invariant. This is a design and independent oracle, with no production rule or message serialization. Host/JIT, flooding/scheduled and single/vmap must agree for an eventual implementation; the derivation tests are independent host references. The contract specifies zero mass, degenerate intervals, vacuous inputs, float32 tails, correlated directions and 100 independent bounds. The matched ablation holds the frozen model, action readout, units and horizon fixed.

**2026-11-22T03:42:58Z**

Research question: what happens to the ledger and the state separator when an action box truncates a correlated Gaussian? The exact closed-region free energy increases by −log P(B) once; on an open separator the message includes P(B|x), usually non-Gaussian. Gaussian conditional lifting gives state moments without replacing the distribution. See D15.2–3, Tallis (1961), Manjunath & Wilhelm (2021), and Genz (1992). Independent verification on seed 2 tick 5 reconstructed the saved per-time means/covariances to 4.5e-14/3.9e-15 max error and found the first three actions correlated. Across 50 retained windows, all 14 adjacent pairs were correlated on every window. Confidence high for these closed frozen Gaussian identities and the measured fixtures. The H15 reactor is outside first-release bounded support.

**2026-11-22T03:46:02Z**

D15 now specifies a closed-region truncated-Gaussian sum-product readout, exact energy accounting and independent scalar/pair implementation envelope. On the two-step LQ example the clipped/MAP/conditional-mean first actions are 0, −0.5 and −0.31785829. All 45 mathematical cases passed; adding dashboard tests gives 58 focused passes. One held-out reactor window was also integrated offline across all 15 correlated commands with two orderings and 12 randomized scrambles per ordering. The first mean moves about −0.32 K despite being initially feasible. Forward/reverse fine estimates 296.14214/296.14453 K. This is an exploratory frozen-window integration, with approximate numerical uncertainty and no plant reward claim. Reproduce via tools/cstr_box_oracle.py STUDY/run --output NEW_FILE and ~/agent-share/cffg-reactor-paired/bounded-actions.html. The qualified short-horizon production work remains cff-fl7h.

**2026-11-22T04:12:00Z**

Final integrated numerical head 1bc7dce passed 2716 default tests with 194 deselected and 4 warnings. Process wall 1179.653 seconds, within the 1200-second budget. The suite included the 45 new mathematical anchors and the existing runtime library. Frozen H15 offline result independently reproduced within 1e-12 for all per-scramble pairs. Next ticket cff-fl7h implements the approved limited block contract.
