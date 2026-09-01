---
id: cff-7ozr
status: open
deps: [cff-lpsn]
links: []
created: 2026-09-01T02:20:19Z
type: task
priority: 1
assignee: coopwilliams
external-ref: gh-84
---
# Clean-port rolling constraint-flip equivalence

## Background in Plain English — WHY

`RollingAgent` uses separate reusable planning and filtering windows instead of
mutating one graph from a future planning constraint to observed data after
each action. Old PR #84 supplied an independent exact reference showing that
these two constructions agree for the repository's declared fixed-A binary
discrete joint-boundary subclass. The evidence is useful even though it is
mostly tests and documentation, so this ticket ports it as the next layer of
the new chain.

## Outcomes / Acceptance Criteria — WHAT

1. An immutable one-slice reference implements the paper-literal transition
   from planning constraint to observed DATA without calling `RollingAgent` or
   its rollout builders.
2. Independent enumeration computes the first action posterior, exact filtering
   posterior, rolled boundary, and second action posterior.
3. The two-window agent and the paper-literal reference agree for the declared
   fixed-A binary discrete joint-boundary subclass.
4. A mixed-history graph independently checks the second plan and the derived
   objective identity between mixed-history and rolled formulations.
5. Tests preserve exact-zero support, report impossible evidence truthfully,
   run under float32, exercise an uncomfortable repeated-tick scale, and prove
   that the agent reuses exactly its declared planning and filtering structures.
6. README, architecture, and limitations state the exact boundary of the
   equivalence and do not generalize it to learned-A functional messages,
   continuous variables, structured boundaries, or arbitrary constraints.
7. The implementation is a semantic port from old PR #84 and remains independent
   of the production paths whose equivalence it tests.

## Methodological Suggestions — HOW

These suggestions are optional. An implementer may choose another sound method.

- Adapt the oracle and tests from old commit `fa167108`, but update public and
  internal calls to current-main interfaces.
- Keep the reference under tests. It is evidence, not a second production
  runtime.
- Applicable change path: documentation and independent execution evidence for
  the `RollingAgent` planning/DATA boundary. Host/JIT, flooding/scheduled,
  single-instance/`vmap`, and current/legacy checkpoint pairs are unchanged and
  therefore N/A.

## Future work

`cff-cfhi` clean-ports the public facade and strict checkpoints after this
evidence layer. Functional learned-A equivalence or broader boundary families
would require separate derivations and tickets.

## Definitions

- **Constraint flip:** Replacing a future planning constraint with observed
  data after the environment returns an observation.
- **Joint boundary:** A rolled categorical belief that keeps all state
  components together rather than factorizing them.
- **Mixed-history graph:** A reference graph that retains observed past and
  planned future slices in one structure.
- **Two-window runtime:** The separate planning and filtering templates reused
  by `RollingAgent`.

## Notes

**2026-09-01T02:23:06Z**

Planning audit, 2026-08-31. Capability owners are the independent test oracle and README, ARCHITECTURE, and LIMITATIONS claim boundaries; production runtime ownership is explicitly denied. Wiggle scan reviewed in context; retained candidates mean: fixed=provided A parameters; n_a=execution pairs proven inapplicable; preserve=literal zero support; public=documented claim; reusable=the two compiled agent windows; implement=reference construction or production path under comparison; support=probability support; remaining=old-PR capability; check=enumerated objective identity; equivalent/evidence/exact=mathematical equality checked by exhaustive finite enumeration and automated tests; oracle=independent test reference; sound=optional method quality; test=automated artifacts; learn=explicitly excluded learned-A scope; scale=repeated tick count; may=optional method; all=state components in the declared joint boundary; current=main interfaces or current-main documentation.
