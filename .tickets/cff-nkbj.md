---
id: cff-nkbj
status: closed
deps: [cff-oyie]
links: [cff-r6bl, cff-0f3g, cff-dlrj]
created: 2026-08-09T13:09:26Z
type: epic
priority: 1
assignee: coopwilliams
parent: cff-ahps
tags: [rslds, validation, learning, statistics]
---
# Epic: Confirm source-specific rSLDS empirical non-inferiority

## Background in Plain English

cff-l7y4 showed that CFFG can fit one synthetic NASCAR rSLDS from
observations at `T=10,000`. It did not compare CFFG with an observed-only
literature learner over multiple seeds. Later source review found that the
pinned authors' learner and the paper do not use the same full parameter
family: the paper has a separate continuous recurrence slope for each source
regime, while the executable learner shares slopes across sources. The pinned
NASCAR generator and learner also select different transition trees unless the
learner tree is supplied explicitly.

This epic formerly coordinated a confirmatory comparison without narrowing
CFFG's source-specific model. Cooper discontinued that comparison on
2026-08-31 before its legacy runtime, paper-equation adapter, preregistration,
or final execution was delivered. Its four child tickets are closed as not
doing and their dependency links now record only the abandoned sequence.

Had it run, the comparison could have supported a bounded empirical
non-inferiority conclusion for its registered model, metrics, seed set, and
reference adapter. It did not run and supports no scientific or implementation
conclusion. Any future comparison requires newly reviewed tickets rather than
resuming this sequence.

## Outcomes / Acceptance Criteria

Not delivered. The criteria below are an archival description of the
discontinued comparison, not actionable work or accepted outcomes.

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
7. A source-specific empirical non-inferiority conclusion is made only if
   every preregistered primary test passes in both comparison tiers.
8. The conclusion states that same-model outcome non-inferiority does not
   prove algorithm parity, authors-code parity, or CFFG-paper fidelity.
9. README.md, docs/ARCHITECTURE.md, and docs/LIMITATIONS.md agree about the
   model boundary, reference provenance, runtime statuses, results, and limits.
10. Balanced-tree and destination-softmax work remains linked future work and
   does not block this decision-list comparison.

## Methodological Suggestions

These historical suggestions described the discontinued sequence. They are
not instructions for active work.

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
- **Empirical non-inferiority:** Passing every preregistered primary outcome
  test against the independent implementation of the published full model,
  within the registered margins and evidence set.
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

Legend: prerequisite --> dependent; [x] delivered earlier; [-] closed not doing

    [-] cff-akmp (reference runtime not delivered)
              |
              +--> [-] cff-64ri (adapter not delivered)
                            |
    [x] cff-l7y4 -----------+
    [x] cff-garu -----------+--> [-] cff-0l31 (protocol not delivered)
    [x] cff-r6bl -----------+              |
                                           +--> [-] cff-oyie (no final run)
                                                         |
                                                         +--> [-] cff-nkbj

    Parallel linked work: [ ] cff-0f3g balanced-tree epic
    Independent research: [ ] cff-dlrj destination-softmax derivation
    Roadmap parent, not a dependency edge: cff-ahps

    The four-ticket comparison path and this epic are archival. They are not
    ready work and do not authorize resuming the discontinued experiment.

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
comparison. Stacked draft PR: https://github.com/engimatic-systems/cffg/pull/62

**2026-08-13T05:19:12Z**

PR merge-gate correction (2026-08-12): narrowed the epic conclusion from literature or implementation parity to source-specific empirical non-inferiority on the registered outcomes. The epic now excludes authors-code parity and CFFG-paper message/free-energy fidelity from that conclusion. cff-64ri owns adapter provenance and independent conditional oracles; cff-0l31 owns direct CFFG recurrence message/energy oracles, Gibbs mixing diagnostics, calibrated margin ceilings, and the exact paired estimand; cff-oyie owns immutable execution and bounded reporting.

**2026-09-01T03:57:27Z**

Closed as not doing at Cooper’s direction on 2026-08-31. The confirmatory source-specific rSLDS comparison epic is discontinued. No implementation result or scientific conclusion from this epic is claimed. Its open child tickets remain separately open and unchanged; this closure does not close or complete them.

**2026-09-01T05:28:00Z**

Roadmap-state repair, 2026-08-31. The earlier closure left all four children
actionable even though their only joint outcome had been discontinued.
`cff-akmp`, `cff-64ri`, `cff-0l31`, and `cff-oyie` are now each closed as not
doing with explicit no-delivery handoffs. The background and outcomes label
the former implementation sequence as archival. Dependency links are retained
only to preserve the historical decomposition; they do not authorize resuming
the work. The separately linked balanced-tree and destination-softmax tickets
keep their own scopes and statuses.
The refreshed wiggle scan retains the recorded meanings; `deliver` and
`complete` distinguish missing outcomes from ticket closure, `could` describes
the counterfactual bounded claim, and `later` refers to historical review
order. Capability ownership is deliberately absent for the abandoned joint
outcome; any future owner must be established in newly reviewed tickets.
The diagram refresh retains the recorded wiggle meanings. In its archival
legend, `ready` is explicitly denied for the discontinued path and `deliver`
distinguishes the three earlier completed prerequisites from the four absent
experiment artifacts.
