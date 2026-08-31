---
id: cc4-vk3q
status: closed
deps: [cff-4jk1]
links: []
created: 2026-08-31T04:08:07Z
type: bug
priority: 1
assignee: coopwilliams
external-ref: gh-99
tags: [consolidation, rslds, numerics]
---
# Return the rSLDS coordinate actually certified by the outer map

## Background in Plain English — WHY

PR #99 can certify one rSLDS parameter coordinate, install that coordinate’s unchecked proposal, and return hidden-state beliefs from the unchecked proposal. A historical weak-prior fit exposed the defect: the qualifying residual was 0.0676 at a 0.1 tolerance, but the returned coordinate’s next complete runtime-map residual rebounded to 0.175. The fitter must return the coordinate whose full map was actually evaluated and certified. The repair must preserve the existing two-consecutive-map rule while avoiding an unnecessary terminal inference after convergence.

The applicable `docs/ARCHITECTURE.md` change-path row is **Runtime numerical or status invariant**. The shared coordinate-fit primitive owns coordinate identities and the two-hit decision. `cffg.models.rslds_fit` owns installed, certified, and returned rSLDS state coordination.

Duplicated execution surfaces: host/JIT is N/A because the decision is host-side and consumes the existing JIT inference result; flooding/scheduled inner inference retains the same engine contract; rSLDS is the production consumer and the experiment-only matched SLDS control is outside this repository stack; current/legacy checkpoint decoding is N/A because no serialized format changes. Converged, exhausted, diverged, and malformed-proposal paths all require explicit evidence.

## Outcomes / Acceptance Criteria — WHAT

1. A converged rSLDS fit returns hidden-state beliefs, installed runtime parameters, and learned posteriors from the same coordinate whose complete runtime-map residual was evaluated.
2. The first consecutive residual at or below tolerance installs its proposal; the second certifies and returns its evaluated coordinate without installing the unchecked proposal.
3. A residual above tolerance resets the consecutive-hit count, so a rebound cannot produce convergence.
4. A converged result avoids the separate terminal inner solve because the certified coordinate already has a converged inference result.
5. An exhausted fit still terminal-solves its last installed coordinate so returned beliefs and parameters agree.
6. A non-finite or malformed proposal produces divergence without installing or returning that proposal.
7. Exact-zero categorical support remains exact and existing improper-input behavior remains unchanged.
8. Float32 and float64 decisions use the caller-provided dtype-scaled tolerance without a new numeric floor or clamp.
9. A synthetic decision sequence with at least twenty complete maps covers rebound behavior at more than ten times the smallest two-map fixture.
10. Phase evidence identifies evaluated, proposed, installed, certified, and returned coordinates.
11. Focused coordinate-fit and rSLDS-fit tests pass, followed by one final default short-suite run at the updated PR #99 head.
12. Canonical architecture documentation describes the returned-coordinate contract and the absence of the terminal solve on convergence.

## Methodological Suggestions — HOW

These suggestions are optional. The implementer may choose another sound method.

- Semantically port the coordinate-decision behavior and regression coverage from historical commit `13276f0` without importing its branch ancestry or ticket state.
- Keep coordinate identity and two-hit selection in `cffg.models.coordinate_fit`; keep rSLDS-specific parameter installation and result assembly in `cffg.models.rslds_fit`.
- Return immediately from the converged outer iteration with the inference result for the evaluated coordinate.

## Future work

Changing from two qualifying maps to a one-map convergence rule is not part of this repair. No other future work is identified.

## Definitions

- **Coordinate:** One complete set of rSLDS initial, dynamics, recurrence, and emission parameter values.
- **Complete runtime map:** Hidden-state inference at one coordinate followed by the full undamped parameter proposal derived from those beliefs.
- **Evaluated coordinate:** The installed coordinate supplied to inference and to the complete runtime map.
- **Unchecked proposal:** The output of a runtime map that has not itself been installed and evaluated.
- **Rebound:** A below-tolerance map followed by an above-tolerance map.


## Notes

**2026-08-31T04:23:44Z**

Implemented in commit 74b31ff without importing historical ancestry. The first qualifying map installs its proposal; the second qualifying map returns the coordinate it evaluated and leaves its unchecked proposal uninstalled. Converged fits now avoid the third inner solve, while exhausted fits still terminal-solve the last installed coordinate. Focused validation: 34 passed. Final default short suite at production commit 74b31ff: 1,473 passed, 39 deselected, 0 failed in 579.57 pytest seconds and 588.183 dashboard wall seconds.

Wiggle-word review retained context-resolved uses: canonical=project-authority document; N/A=structurally irrelevant duplicated surface; preserve=semantic invariant; same=coordinate identity or unchanged engine contract as named; expose=reveal defect; support=literal probability support; complete=the enumerated full runtime map or coordinate; remaining=preserved behavior; certify=runtime certificate; converge=outer runtime status or explicitly named inner status; evidence=phase provenance or automated result; exact=literal zero support; match=the separately named experiment control; pass=tests succeed; sound=implementation judgment; test=automated artifact or execution; fit=repeated fixed-dataset parameter estimation; infer=hidden-state posterior; learn=parameter posterior; scale=dtype numeric magnitude; small=map-count fixture size; can=defect possibility; may=implementation permission; explicit_owner=the named coordinate_fit and rslds_fit components; all=the enumerated status paths; at_least=inclusive lower bound; only=ticket scope boundary; current=PR #99 repository reference; existing=behavior predating this repair. Capability ownership was checked separately; no required owner is omitted. No follow-up work or gotcha remains.
