---
id: cff-nkbj
status: in_progress
deps: [cff-oyie]
links: [cff-r6bl, cff-0f3g, cff-dlrj]
created: 2026-08-09T13:09:26Z
type: epic
priority: 1
assignee: coopwilliams
parent: cff-ahps
tags: [rslds, validation, learning, statistics]
---
# Epic: Confirm full source-specific rSLDS reference parity

## Background in Plain English

cff-l7y4 showed that CFFG can fit one synthetic NASCAR rSLDS from
observations at `T=10,000`. It did not compare CFFG with an observed-only
literature learner over multiple seeds. Later source review found that the
pinned authors' learner and the paper do not use the same full parameter
family: the paper has a separate continuous recurrence slope for each source
regime, while the executable learner shares slopes across sources. The pinned
NASCAR generator and learner also select different transition trees unless the
learner tree is supplied explicitly.

This epic coordinates a fair confirmatory comparison without narrowing CFFG's
source-specific model. cff-akmp locks the legacy runtime. cff-64ri adds an
independent paper-equation Gibbs adapter for the full model. cff-0l31 freezes
the mapping, harness, and statistical protocol. cff-oyie runs the final seeds
and reports the result. The split keeps environment repair, reference
semantics, scientific choices, and confirmatory evidence independently
reviewable.

## Outcomes / Acceptance Criteria

1. cff-akmp is closed with a reproducible decision-list legacy runtime and an
   observed-only recurrence-only smoke artifact.
2. cff-64ri is closed with an independent full source-specific
   paper-equation Pólya–Gamma Gibbs adapter and its analytic and collapse
   oracles.
3. cff-0l31 is closed with a hashed preregistration that fixes the complete
   final comparison before any final seed runs.
4. cff-oyie is closed with every registered seed, failure, aggregate,
   uncertainty interval, artifact, and supported conclusion reported.
5. The primary conclusion compares CFFG with the independent paper-equation
   adapter on the same full source-specific decision-list model.
6. The unmodified pinned authors' learner is reported only as a narrower
   recurrence-only secondary comparator. It is not described as a same-model
   reference for source-specific learning.
7. A literature-parity claim is made only if every preregistered primary
   non-inferiority test passes in both primary comparison tiers.
8. README.md, docs/ARCHITECTURE.md, and docs/LIMITATIONS.md agree about the
   model boundary, reference provenance, runtime statuses, results, and limits.
9. Balanced-tree and destination-softmax work remains linked future work and
   does not block this decision-list comparison.

## Methodological Suggestions

These suggestions are optional. The implementer may choose another sound
method that satisfies the outcomes.

- Treat the four child tickets as phase barriers. Do not move semantic adapter
  work into environment repair or implementation changes into final execution.
- Keep the primary same-model comparison separate from the useful but narrower
  unmodified-upstream baseline in every artifact and conclusion.
- Put any evidence-backed failure repair in a new ticket rather than changing
  the frozen protocol after confirmatory execution starts.

## Future work

cff-0f3g owns explicit tree topology, balanced-tree inference, gate sharing,
learning, and scale validation. cff-dlrj owns the first derivation decision for
a destination-softmax recurrence factor. Neither ticket is a prerequisite of
this epic. Real observations and other switching-system datasets remain
future validation after the synthetic NASCAR comparison.

## Definitions

- **Decision-list model:** A categorical transition that tests logistic gates
  in order and selects the first accepted destination.
- **Literature parity:** Passing every preregistered primary non-inferiority
  test against the independent implementation of the published full model.
- **Paper-equation adapter:** A separately versioned implementation of the
  published equations that is based on pinned legacy code but was not shipped
  unchanged by the paper's authors.
- **Primary comparison tier:** A same-model CFFG-versus-adapter comparison
  using either shared truth-free initialization or each workflow's native
  truth-free initialization.
- **Source-specific recurrence:** A transition whose continuous-state slope
  and offset are selected by the current discrete regime.

## Notes

**2026-08-10T03:10:00Z**

Roadmap decomposition after the reference-feasibility and CFFG-paper reviews.

Legend: prerequisite --> dependent

    [ ] cff-akmp (ready: locked legacy runtime)
              |
              +--> [ ] cff-64ri (paper-equation adapter)
                            |
    [x] cff-l7y4 -----------+
    [x] cff-garu -----------+--> [ ] cff-0l31 (frozen protocol)
    [x] cff-r6bl -----------+              |
                                           +--> [ ] cff-oyie (final evidence)
                                                         |
                                                         +--> [ ] cff-nkbj

    Parallel linked work: [ ] cff-0f3g balanced-tree epic
    Independent research: [ ] cff-dlrj destination-softmax derivation
    Roadmap parent, not a dependency edge: cff-ahps

Research question: what comparison remains faithful to both the CFFG papers
and the Linderman rSLDS paper? Answer: CFFG does not prescribe source-specific
or tied recurrence parameters; it requires the chosen model, variational
family, constraints, local messages, and free-energy terms to be explicit.
The Linderman paper's full model uses source-specific recurrence slopes, but no
audited unmodified `recurrent-slds` learner implements that full family. The
fair primary reference is therefore an independent source-specific
paper-equation adapter, with the narrower authors' learner retained as a
secondary nested-model comparator. Sources: `notes/pt1.txt`, sections 1 and 5;
`notes/pt2.txt`, sections 2.3 and 4; Linderman et al. 2017 Eq. 4 and supplement;
`recurrent-slds@383ddac` transition, state, and NASCAR learner sources.
Primary external sources:
https://proceedings.mlr.press/v54/linderman17a/linderman17a.pdf#page=3 ;
https://proceedings.mlr.press/v54/linderman17a/linderman17a-supp.pdf#page=1 ;
https://github.com/slinderman/recurrent-slds/blob/383ddac2472cdf3248c6e871baede9ecda6f14cb/rslds/transitions.py#L19-L126 ;
https://github.com/slinderman/recurrent-slds/blob/383ddac2472cdf3248c6e871baede9ecda6f14cb/rslds/states.py#L147-L201 ;
https://github.com/slinderman/recurrent-slds/blob/383ddac2472cdf3248c6e871baede9ecda6f14cb/examples/nascar.py#L421-L464 .
Confidence: high on the model and executable boundaries; medium-high on
legacy integration until cff-akmp and cff-64ri execute their tests.

**2026-08-10T03:20:00Z**

Post-rewrite wiggle audit retained clear uses with these intended meanings:
integrate=executing the legacy adapter inside its locked environment;
register=the preregistered seed and metric set; same=the explicitly named full
source-specific model; implement=the named CFFG, adapter, or upstream software;
support=a conclusion justified by the final evidence; complete=the enumerated
protocol or artifact set; ready=tk dependency readiness; remaining and
later=named future or historical work; close=ticket closure; evidence=the
confirmatory or provenance record; oracle=the adapter's analytic check;
parity and pass=the defined joint non-inferiority gate; reproduce=rerunning the
locked runtime; sound=an alternative method satisfying the outcomes;
test=an automated check, statistical test, or binary gate test as stated;
validate=the named scientific or scale evaluation; fit and learn=parameter
estimation; infer=balanced fixed-belief inference; scale=the named benchmark
dimensions; can=the demonstrated cff-l7y4 capability; may=optional method;
explicit_owner and named_ticket_handoff=the named cff-0f3g/cff-dlrj owners;
any, every, and only=the locally enumerated seeds, metrics, phases, or hard
claim boundaries; current=the mathematical source regime at time `t`; and
reproduce=the rerunnable environment artifact. Capability ownership was
checked separately: the epic owns the joint conclusion, while each child owns
one phase named in the background. Intelligible-ticket review: complete.

Classification-note additions: fixed=parameters held during the named
inference comparison; check=the adapter's analytic or executable authority;
estimate=observation-only parameter estimation; intended=the contextual
meaning assignments in this note.

The retained all quantifier means the explicitly enumerated twelve-ticket
authoring set.

Authoring handoff: all twelve new or rewritten tickets completed their current
wiggle scan; `tk dep cycle` and `git diff --check` pass. cff-akmp is the next
immediate ticket, and cff-4sor is the independent balanced-tree root. Gotcha:
cff-dlrj and cff-0f3g are links, not prerequisites of the decision-list
comparison.
