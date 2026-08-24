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
# Test whether recovery-compatible D10 points falsify the learned terminal

## Background in Plain English — WHY

The first source-heterogeneous D10 development fit converged, but it still missed both registered recovery ceilings. Its worst-source transition error was 0.206 and its worst source-contrast error was 0.318; each ceiling is 0.150. The earlier D9 fit on the same observations and initialization scored almost identically. D10 therefore fixed the formal inference and learning ownership defects without fixing this recovery problem.

The first draft of this ticket tried to distinguish a poor optimization basin from a limitation of the D10 variational family by comparing fixed-parameter D10 free energy on the training sequence. Independent pre-launch review falsified that two-sided decision. Even with an exact model family and a sound optimizer, fitted parameters normally score the data they were fitted on better than the generating parameters. A synthetic oracle reproduced this effect and made the draft classifier falsely recommend a richer variational family.

This ticket now asks a narrower question that the evidence can answer. It tests whether any declared recovery-compatible point has materially lower fixed-parameter D10 free energy than the learned terminal. Such a result would make a poor basin or learning objective mismatch plausible. The reverse result does not establish a variational-family limitation and must remain ambiguous. The diagnostic does not run another outer fit or select a new initialization.

## Outcomes / Acceptance Criteria — WHAT

1. The diagnostic authenticates the immutable D10 seed-5000/native artifact, result, manifest, checkpoint, observations, streams, CFFG revision, D10 schema-v2 topology, and all four statistic owners before reading development truth or running inference.
2. The declared evaluation set contains exactly 13 points: the D10 learned point and alpha 0.25, 0.50, 0.75, and 1.00 substitutions along the generating-dynamics, generating-recurrence, and both-generating paths. The registration fixes this order before any result is inspected.
3. Every evaluation point runs fixed-parameter D10 inference under one shared numerical contract. Each point also runs a predeclared tighter verification solve. The report records both statuses, work, normalized objectives, recovery errors, and exact parameter identities. A point is objective-rankable only when both solves converge.
4. Objective comparison tolerance is at least the measured absolute difference between the standard and tighter solve for both compared points, plus a dtype- and term-count-scaled accumulation guard. It cannot use a machine-rounding threshold below the observed solver spread.
5. Held-out predictive log likelihood is evaluated on the unused 1,280-step continuation with a fixed, preregistered set of common particle streams and particle count. Paired simultaneous intervals quantify Monte Carlo uncertainty. This result is descriptive generalization evidence; it cannot by itself establish a variational-family limitation or an outer-learning basin.
6. For every converged point, the diagnostic computes one complete four-family M-step direction but does not install it. It retains the pair, pooled-statistic, posterior-consistency, alignment, and coverage checks used by the pilot. `statistics_coherent` is true only when pair consistency, posterior consistency, and every proposed family are finite and valid. The direction is descriptive and does not control the ticket classification.
7. The result selects `local_basin_plausible` only when a recovery-compatible declared point has lower training D10 free energy than the learned terminal by more than the combined measured comparison tolerance. Every other complete result is `ambiguous`. The result schema contains no `variational_family_likely_limiter` outcome.
8. A converged point with incoherent statistics is preserved as valid negative evidence and forces `ambiguous`; it does not make artifact validation crash. Missing, censored, non-finite, tied, or contradictory evidence is also preserved and cannot receive a stronger classification.
9. The artifact is append-only, manifested, and independently replayable. It preserves every declared point and failure. It does not import or relabel D9 posterior evidence as D10 evidence.
10. Compact tests make forbidden execution executable assertions by replacing outer fitting, adapters, controls, calibration/final seed access, and new-initialization entry points with failures before running the diagnostic harness.
11. No outer rSLDS fit, adapter chain, SLDS control, calibration seed, final seed, production code change, adaptive point, or new initialization search runs in this ticket.
12. The ticket ends with one explicit downstream recommendation. A `local_basin_plausible` result recommends a separately registered D10 initialization or bounded-search study. An `ambiguous` result reports whether held-out prediction favors a recovery-compatible point, then leaves richer joint q(z,x), direct bound-gap measurement, or deferral of heterogeneous recovery as an explicit user decision.

## Methodological Suggestions — HOW

These suggestions are optional. The implementer may choose another sound method.

- Add a development-only experiment command next to the D10 requalification runner and reuse its authenticated observation/checkpoint loaders.
- Normalize the D10 objective by training transition count for reporting, while deriving the comparison guard from the unnormalized term count and the measured standard-versus-tight solve difference.
- Define interpolation in a valid family-specific chart: simplex or expected-log coordinates for categorical values, natural or Cholesky-safe coordinates for Gaussian families, and ordinary coordinates for unconstrained recurrence weights. Reject an invalid point instead of projecting it silently.
- Use the existing independent functional-recovery and four-family natural-statistic oracles for semantic replay.
- Reuse the existing held-out particle likelihood implementation with common random streams, but register the particle count, stream coordinates, multiplicity correction, and interval rule before execution.
- Stop after the fixed evaluation set. Do not adaptively add points based on intermediate results.

## Future work

- If the result is `local_basin_plausible`, create a separate registered development ticket for an observation-only initialization policy or bounded multi-start search under the unchanged D10 objective.
- If the result is `ambiguous`, cff-ngxx requires a user decision among direct bound-gap evidence, a richer joint switching Gaussian trajectory q(z,x), or deferral. A richer-family ticket must define objective ownership, persistence, hostile-input, and scaling obligations before production work.
- Classic homogeneous parity may proceed independently, with the heterogeneous claim explicitly deferred.
- cff-h0hx and cff-0oqo remain blocked until a fresh development fit passes the unchanged 0.150 recovery ceilings.

## Definitions

- **D10:** The current structured hybrid coordinate with one categorical regime chain q(z) and one Gaussian Markov latent trajectory q(x), plus one shared objective and four-family learning statistics.
- **Local basin:** A parameter region reached by the fitter even though a declared comparison point has a better measured value under the same fixed-parameter training objective.
- **Variational family:** The restricted class of approximate probability distributions that inference is allowed to use.
- **Recovery-compatible point:** A declared parameter point whose fixed-parameter inference meets both 0.150 source-heterogeneous recovery ceilings.
- **M-step direction:** The parameter update proposed from one fixed set of inferred sufficient statistics. This ticket measures it but does not install it.
- **Bound gap:** The difference between a variational free-energy bound and the exact or independently estimated log evidence for the same parameters and data.
- **Simultaneous interval:** An uncertainty interval corrected so that the complete registered set of comparisons has the stated coverage, rather than each comparison in isolation.

## Notes

Architecture handoff: this is a post-fit experiment diagnostic. The CFFG D10 compiler, engine, learning, objective, persistence, and model behavior are structurally unchanged by this ticket unless the diagnostic uncovers a bounded implementation mismatch. Historical D9 and D10 artifacts remain immutable.

Hostile-input decisions: exact categorical zeros remain exact; invalid simplex, covariance, support, or non-finite evidence fails loudly; all guards scale with dtype; a censored fixed-parameter solve is preserved and excluded from objective ranking.

Legend: prerequisite --> dependent

```text
[x] cff-ngxx fresh D10 pilot falsifier
      |
      +--> [ ] cff-xq4y one-sided objective-path falsifier (this ticket)
                  |
                  +--> [ ] bounded D10 initialization/global search
                  |         or
                  +--> [ ] direct bound-gap / richer joint q(z,x) design
                  |         or
                  +--> [ ] defer heterogeneous recovery
                              |
                              +--> [ ] cff-ngxx / cff-h0hx requalification
                                          |
                                          +--> [ ] cff-0oqo heterogeneous evidence
```

**2026-08-24T04:26:26Z**

Pre-launch independent review (2026-08-23). Source: /Users/ravd/agent-share/RSLDS/cff-xq4y-prelaunch-review-response.md. Answer: the draft two-sided training-BFE classifier is invalid. A compact exact-family oracle made fitted parameters beat generating parameters in-sample and falsely selected the variational-family verdict. The review also measured the registered objective tolerance below solver spread, found posterior consistency absent from coherence, found an incoherent-point validator/classifier contradiction, and found forbidden-path assertions missing. Confidence high: the reviewer supplied executable counterexamples and 16 focused tests passed. Decision: no launch at 1fd925c; replace the draft with a one-sided falsifier, measured solve uncertainty, descriptive held-out predictive evidence, complete coherence, ambiguity-preserving validation, and executable forbidden-path tests.
