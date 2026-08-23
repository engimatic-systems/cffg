---
id: cff-xq4y
status: in_progress
deps: []
links: []
created: 2026-08-23T01:10:29Z
type: task
priority: 1
assignee: coopwilliams
parent: cff-ngxx
tags: [rslds, inference, diagnostics, trustworthiness]
---
# Discriminate D10 basin failure from variational-family bias

## Background in Plain English — WHY

The first source-heterogeneous D10 development fit converged, but it still missed both registered recovery ceilings. Its worst-source transition error was 0.206 and its worst source-contrast error was 0.318; each ceiling is 0.150. The earlier D9 fit on the same observations and initialization scored almost identically. D10 therefore fixed the formal inference and learning ownership defects without fixing this recovery problem.

We must now distinguish two explanations before changing production inference again. The fit may have stopped in a poor local basin even though the D10 objective has a better recovery-compatible solution. Alternatively, the D10 variational family or objective may prefer the biased solution it found. This ticket makes that decision with fixed-parameter, post-fit diagnostics. It does not run another outer fit or select a new initialization.

## Outcomes / Acceptance Criteria — WHAT

1. The diagnostic authenticates the immutable D10 seed-5000/native artifact, result, manifest, checkpoint, observations, streams, CFFG revision, D10 schema-v2 topology, and all four statistic owners before reading development truth or running inference.
2. The declared evaluation set contains the D10 learned point, the generating-dynamics substitution, the generating-recurrence substitution, the both-generating substitution, and a fixed bounded set of label-aligned interpolations between the learned and recovery-compatible points. The registration fixes the set before any result is inspected.
3. Every evaluation point runs fixed-parameter D10 inference under one shared numerical contract. The report records convergence status, work, normalized D10 objective, recovery errors, and the exact parameter identity for every point. A non-converged point remains censored and is not ranked as if it had a valid objective.
4. For every converged point, the diagnostic computes one complete four-family M-step direction but does not install it. It reports whether that direction moves toward or away from recovery-compatible parameters and retains the same pair, pooled-statistic, posterior-consistency, alignment, and coverage checks used by the pilot.
5. The result selects `local_basin_plausible` only if at least one recovery-compatible point has a lower valid D10 objective than the learned terminal, or if a coherent update-direction result independently shows that the learned terminal is not locally preferred. The result selects `variational_family_likely_limiter` only if the learned terminal has the best valid objective, every recovery-compatible point is worse, and the retained statistics remain coherent. All other outcomes are `ambiguous`.
6. The artifact is append-only, manifested, and independently replayable. It preserves each failed, censored, or non-finite point. It does not import or relabel D9 posterior evidence as D10 evidence.
7. No outer rSLDS fit, adapter chain, SLDS control, calibration seed, final seed, production code change, or new initialization search runs in this ticket.
8. The ticket ends with one explicit downstream recommendation: bounded D10 initialization/global search, a new richer joint q(z,x) inference-family design, or deferral of heterogeneous recovery while classic homogeneous parity proceeds.

## Methodological Suggestions — HOW

These suggestions are optional. The implementer may choose another sound method.

- Add a development-only experiment command next to the D10 requalification runner and reuse its authenticated observation/checkpoint loaders.
- Normalize the D10 objective by training transition count so every point is comparable at the same scale.
- Define interpolation in a valid family-specific chart: simplex or expected-log coordinates for categorical values, natural or Cholesky-safe coordinates for Gaussian families, and ordinary coordinates for unconstrained recurrence weights. Reject an invalid point instead of projecting it silently.
- Use the existing independent functional-recovery and four-family natural-statistic oracles for semantic replay.
- Stop after the fixed evaluation set. Do not adaptively add points based on intermediate results.

## Future work

- If the result is `local_basin_plausible`, create a separate registered development ticket for an observation-only initialization policy or bounded multi-start search under the unchanged D10 objective.
- If the result is `variational_family_likely_limiter`, create a separate derivation ticket for a richer joint switching Gaussian trajectory q(z,x), including objective ownership, persistence, hostile-input, and scaling obligations.
- If the result is `ambiguous`, cff-ngxx requires a user decision before another fit. Classic homogeneous parity may proceed independently, with the heterogeneous claim explicitly deferred.
- cff-h0hx and cff-0oqo remain blocked until a fresh development fit passes the unchanged 0.150 recovery ceilings.

## Definitions

- **D10:** The current structured hybrid coordinate with one categorical regime chain q(z) and one Gaussian Markov latent trajectory q(x), plus one shared objective and four-family learning statistics.
- **Local basin:** A locally preferred parameter region reached by the fitter even though another region has a better value under the same objective.
- **Variational family:** The restricted class of approximate probability distributions that inference is allowed to use.
- **Recovery-compatible point:** A declared parameter point whose fixed-parameter inference meets both 0.150 source-heterogeneous recovery ceilings.
- **M-step direction:** The parameter update proposed from one fixed set of inferred sufficient statistics. This ticket measures it but does not install it.

## Notes

Architecture handoff: this is a post-fit experiment diagnostic. The CFFG D10 compiler, engine, learning, objective, persistence, and model behavior are N/A unless the diagnostic uncovers a bounded implementation mismatch. Historical D9 and D10 artifacts remain immutable.

Hostile-input decisions: exact categorical zeros remain exact; invalid simplex, covariance, support, or non-finite evidence fails loudly; all guards scale with dtype; a censored fixed-parameter solve is preserved and excluded from objective ranking.

Legend: prerequisite --> dependent

```text
[x] cff-ngxx fresh D10 pilot falsifier
      |
      +--> [ ] cff-xq4y objective-path discriminator (this ticket)
                  |
                  +--> [ ] bounded D10 initialization/global search
                  |         or
                  +--> [ ] richer joint q(z,x) inference design
                  |         or
                  +--> [ ] defer heterogeneous recovery
                              |
                              +--> [ ] cff-ngxx / cff-h0hx requalification
                                          |
                                          +--> [ ] cff-0oqo heterogeneous evidence
```
