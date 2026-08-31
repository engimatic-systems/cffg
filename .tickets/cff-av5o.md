---
id: cff-av5o
status: closed
deps: []
links: [cff-cz6v]
created: 2026-08-31T03:15:44Z
type: task
priority: 1
assignee: coopwilliams
---
# Reconcile resumable CFFGZ learning with the open roadmap

## Background in Plain English — WHY

Cooper wants every `.cffgz` file written by the schema that follows
`agent/cff-yoxz-bundle-specimen` to carry
enough model-level state to continue learning. Before adding roadmap work, the
project must separate what is already implemented, what an open specialized
ticket already owns, and what no ticket owns.

The audit starts from `agent/cff-yoxz-bundle-specimen`, which exposes the
present archive format. Closed tickets `cff-x0sv`, `cff-6tek`, `cff-qg1d`, and
`cff-s6yy` already supply declared learning, family-agnostic keyed
hyperparameter containers, immutable priors, the five learning families
registered in `cffg.learn.FAMILIES`, versioned checkpoints,
and recurrence statistic identity. Open tickets `cff-602p`, `cff-4c7t`,
`cff-9vab`, `cff-1o72`, `cff-pyy9`, and `cff-t2g3` own specialized topology,
training, hierarchy, or agent persistence. None makes `.cffgz` the model-general
model-level learning-resume artifact.

## Outcomes / Acceptance Criteria — WHAT

1. The ticket record identifies each part of the proposed resumable-bundle
   design as implemented, owned by an open specialized ticket, or missing.
2. Missing work is represented by one bounded epic and reviewable child
   tickets with explicit dependency links.
3. Existing roadmap tickets gain a dependency only when implementing them
   first would duplicate or bypass the new model-general persistence contract.
4. Every new or materially changed ticket passes a bounded wiggle-word scan
   after ambiguous claims are clarified, and the handoff records retained
   candidates with their intended meanings.
5. `tk ready`, `tk blocked`, and `tk dep tree --full` support a compact ASCII
   dependency tree that identifies ready roots, parallel specialized work,
   immediate blockers, and downstream epics.
6. `tk dep cycle` and `git diff --check` succeed.

## Methodological Suggestions — HOW

The implementer may use another sound audit method. Search open ticket text for
bundle, checkpoint, posterior, schema, and learning-state ownership. Read every
matching ticket that could own the model-general capability, then compare it
with the `LearnableSpec`, `Family`, `Learnables`, checkpoint, and bundle
contracts on `agent/cff-yoxz-bundle-specimen`.

This is a ticket-only roadmap change. The architecture change-path index,
runtime/compiler/serialization implementation, host/JIT, flooding/scheduled,
single-instance/`vmap`, current/legacy decoding, and the hostile-input checklist
are not implementation surfaces for this audit. The child tickets must name the
applicable **New IR field** and **Learnable or checkpoint feature** surfaces.

## Future work

Epic `cff-cz6v` and child tickets `cff-xabn`, `cff-enhs`, and `cff-jilz` own the
missing implementation. The linked specialized roadmap tickets keep their
existing model- or runtime-specific responsibilities.

## Definitions

- **Declared learning:** Parameter learning described by serialized
  `LearnableSpec` records and executed by `Learnables`.
- **Learning-resume artifact:** A saved model that can reconstruct the committed
  posterior state and accept evidence from a later trial.
- **Model-level state:** Priors, posterior hyperparameters, and installed point
  parameters. It excludes an unfinished inference sweep and agent clock state.
- **Specialized persistence:** Saved state whose meaning is limited to one
  topology, factor family, agent, or experiment rather than the generic bundle
  contract.

## Notes

**2026-08-31T03:24:29Z**

Roadmap coverage audit at base ca87069:

Implemented foundations: cff-x0sv owns serialized LearnableSpec declarations and generic keyed state; cff-6tek owns immutable priors, generic NIW/MNIW state, accumulation/replacement, and checkpoint restoration; cff-qg1d owns component-MNIW and PG family registration; cff-s6yy owns D9 recurrence identity and refusal of unidentified recurrence posteriors; cff-yoxz supplies the inspectable model-only specimen.

Open specialized ownership: cff-602p owns recurrence topology serialization; cff-4c7t owns tree-gate coefficient owners and pooled learning; cff-9vab owns balanced-tree builder integration; cff-1o72 owns neural optimizer/training state; cff-pyy9 owns hierarchy runtime state; cff-t2g3 owns per-patient agent replay; cff-ahps owns later factor and learning families.

Missing before this refinement: a family-independent versioned learning-semantic descriptor, CFFGZ storage for immutable priors and current posteriors, a save/load API that refuses schema-current learnable model-only output, and next-update equivalence across the five registered families. New epic cff-cz6v and children cff-xabn, cff-enhs, and cff-jilz own those four items.

**2026-08-31T03:24:29Z**

Legend: prerequisite --> dependent; ~ means a nonblocking overlap link.

[x] cff-6tek ----+
[x] cff-s6yy ----+--> [ ] cff-xabn (ready)
                           |
[x] cff-yoxz ------------>+--> [ ] cff-enhs
                                      |
                                      +--> [ ] cff-jilz
                                                  |
                                                  +--> [ ] cff-cz6v

[ ] cff-lv65 ----+
[ ] cff-xabn ----+--> [ ] cff-4c7t

[ ] cff-cz6v ----+
[ ] cff-4c7t ----+--> [ ] cff-9vab --> [ ] cff-0f3g

Parallel linked overlaps, not blockers:
[cff-cz6v] ~ [cff-602p, cff-1o72, cff-pyy9, cff-t2g3, cff-ahps]

Ready root: cff-xabn. Immediate blockers: cff-enhs waits on cff-xabn; cff-4c7t waits on cff-lv65 and cff-xabn; cff-9vab waits on cff-4c7t and cff-cz6v. Downstream roadmap epic: cff-0f3g.

**2026-08-31T03:24:29Z**

Retained bounded-scan candidates and intended meanings: generic=protocol-general or model-general contract; register=runtime family coverage; expose=inspection access to the specimen; implement=existing primitive, planned child work, or non-code roadmap change as stated; support=commands supply dependency evidence; ready=dependency-ready; close=work-item completion; material=scope impact; sufficient=state mathematically sufficient for next-trial resume; check=automated non-test command; evidence=observations added to a posterior; match=suitability of a candidate ticket; pass=non-test command success; sound=reasonable audit method; infer=fixed-parameter state inference; learn=new-evidence accumulation or integrated workflow; learnable=declared or registered fitting surface; trial=independent later dataset; can=capability guaranteed by the definition; could=possible ownership option; intended=explicit classification meaning; may=optional method; explicit_owner and named_ticket_handoff=the named tickets in the audit; every=each member of the named set; only=ticket or dependency scope; current=the named schema-current/legacy pair; existing=roadmap state at ca87069; later=ordered future trial; read=future audit requirement. Capability ownership was checked separately in the coverage note.

**2026-08-31T03:25:56Z**

Post-handoff scan additions: fixed=constant-parameter behavior already implemented; integrate=the public or internal consumer workflow named by an existing ticket; complete=the whole missing item set named in the coverage note; appropriate=suitability of the ticket boundary; equivalent=next-update behavior delegated to cff-jilz; test=the cff-yoxz artifact test; accumulate=new evidence from distinct trials; fit=the declared rSLDS parameter workflow.

**2026-08-31T03:26:14Z**

Final classification addition: public=documented consumer workflow named by an existing ticket.

**2026-08-31T03:26:47Z**

Handoff: completed the open-roadmap audit and added epic cff-cz6v with children cff-xabn, cff-enhs, and cff-jilz. Added cff-xabn as a prerequisite of cff-4c7t and cff-cz6v as a prerequisite of cff-9vab, with explanatory notes. Linked nonblocking specialized persistence work without changing its scope. Final bounded scans were rerun for all five new tickets and for the two dependency-modified recurrence tickets; retained meanings and capability owners are recorded in ticket notes. tk ready identifies cff-xabn as the new ready root. tk blocked shows the expected child and recurrence blockers. tk dep tree --full supports the recorded ASCII tree. tk dep cycle and git diff --check pass. No test suite was run because every repository change is ticket metadata only.

**2026-08-31T03:27:11Z**

Final handoff scan additions: expected=dependency state implied by the recorded tk edges; all=the enumerated five new tickets and two dependency-modified recurrence tickets.
