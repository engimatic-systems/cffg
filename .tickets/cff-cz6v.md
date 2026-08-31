---
id: cff-cz6v
status: open
deps: [cff-jilz]
links: [cff-av5o, cff-1o72, cff-t2g3, cff-602p, cff-ahps, cff-pyy9]
created: 2026-08-31T03:16:28Z
type: epic
priority: 1
assignee: coopwilliams
tags: [serialization, learning, checkpoint]
---
# Epic: Make current CFFGZ bundles resumable for declared learning

## Background in Plain English — WHY

The `.cffgz` format on base commit `ca87069` stores graph structure and
point-parameter blobs.
The same model may declare learnable parameters, but its immutable priors and
current posterior hyperparameters live in a separate `Learnables` object. A
user can therefore load the model and run inference but cannot continue the
same learning history from the bundle alone.

CFFG already represents each learning family's numeric state as named arrays.
This epic makes that generic state part of the portable model artifact. It does
not turn a model bundle into a snapshot of an unfinished inference run or a
rolling agent. Those exact runtime snapshots remain checkpoints.

## Outcomes / Acceptance Criteria — WHAT

1. Every `.cffgz` written by the schema-current public save API states whether
   the template declares learnables.
2. A schema-current bundle with declared learnables contains the immutable
   prior arrays, current posterior hyperparameter arrays, point-parameter
   blobs, and semantic descriptors needed to continue at the next committed
   learning boundary without caller-supplied priors.
3. The public save API refuses to write a schema-current learnable bundle when
   any required prior, posterior, semantic descriptor, or point-parameter blob
   is missing or incompatible.
4. A template with no declared learnables carries an explicit empty learning
   inventory and remains a complete fixed-parameter model.
5. Loading returns a model artifact that can reconstruct both an `Engine` and
   compatible `Learnables`; integrity and semantic checks finish before either
   object is exposed for mutation.
6. Original model-only bundles remain loadable through a documented legacy
   path that reports their inability to resume learning and never invents
   priors or posterior history.
7. The five registered families—Dirichlet, NIW, MNIW, component MNIW, and
   Pólya–Gamma weights—demonstrate save, load, and next-update equivalence.
8. Registering a later learning family provides the same persistent-state
   descriptor through the generic family contract instead of adding a new
   bundle layout.
9. `README.md`, `docs/ARCHITECTURE.md`, and `docs/LIMITATIONS.md` replace the
   model-only statement with the delivered capability and preserve the
   boundary between model-level resume and exact runtime checkpoints.

## Methodological Suggestions — HOW

The child tickets may choose another design that satisfies the outcomes.
Prefer one versioned semantic descriptor per declared learnable and role-labeled
array namespaces for immutable priors and current posteriors. Keep existing
point parameters in the ordinary blob namespace. Use the manifest to bind
every array's logical owner, role, shape, dtype, digest, and learning semantics.

Use the **New IR field** and **Learnable or checkpoint feature** change-path
rows. Review validation, JSON and structure hashing, plate scoping, composition
and edits, compiler learn plans, family registration, checkpoint migration,
bundle encode/decode, agent handoff, and public exports. Numerical message rules
do not change, but loaded point parameters must remain usable under host/JIT,
flooding/scheduled, and single-instance/`vmap` consumers. Current/legacy bundle
and checkpoint pairs require explicit tests.

## Future work

`cff-xabn` defines versioned learning semantics. `cff-enhs` stores and restores
the arrays. `cff-jilz` proves the complete workflow and closes this epic.

Linked tickets retain narrower responsibilities: `cff-602p` owns recurrence
topology migration; `cff-1o72` owns neural optimizer and training state;
`cff-pyy9` owns an executing hierarchy snapshot; `cff-t2g3` owns per-patient
agent replay; and `cff-ahps` owns future learning families. Those tickets must
reuse this contract when they save declared model learning, but this epic does
not absorb their runtime state.

## Definitions

- **Committed learning boundary:** A point after an atomic posterior update and
  point-parameter installation, with no partly applied statistic update.
- **Current posterior:** The hyperparameters after all learning evidence
  accepted before the bundle was saved.
- **Immutable prior:** The original hyperparameters used by replacement updates
  and retained unchanged throughout a learning session.
- **Model-level resume:** Starting a later inference or learning trial from the
  saved model parameters and posterior history.
- **Runtime checkpoint:** A snapshot that may also contain messages,
  observations, counters, pending rollovers, or random-number state.
- **Schema-current bundle:** A bundle written with the newest bundle schema and
  public save contract, as opposed to a legacy model-only archive.

## Notes

**2026-08-31T03:24:42Z**

Retained bounded-scan candidates and intended meanings: fixed=constant-parameter model; generic=family-independent protocol; preserve=compatibility or the model/checkpoint boundary; public=documented save/load contract or exported symbol; register=runtime family coverage and later extension API; reusable=public reuse by linked consumers; same=model identity, learning-history identity, or one shared descriptor contract as named; expose=return a mutation-safe public object; provide=ticket requirement for future family registration; complete=exhaustive model state or full child workflow; deliver=consumer outcome documented after implementation; remaining=preserved fixed-model behavior or deliberately excluded runtime snapshots; close=acceptance-based epic closure; check=input, integrity, or semantic guard; equivalent=behavioral next-update equality defined by cff-jilz; evidence=observations accepted into the posterior; exact=byte/value-complete runtime replay boundary; test=automated current/legacy checks; validate=input and artifact validation; infer=fixed-parameter state inference; learn=new-evidence accumulation or integrated declared-learning workflow; learnable=serialized declaration and registered fitting surface; trial=independent later dataset; can=required consumer capability; may=optional implementation or optional runtime contents; explicit_owner and named_ticket_handoff=the tickets named in Future work; all=the full accepted evidence history; any=negative scope over required state; every=each schema-current public save; only=legacy model-only limitation; current=runtime-active posterior or schema-current format; existing=pre-ticket point-blob namespace; later=future family registration or trial. Capability ownership was checked separately: cff-xabn owns semantics, cff-enhs archive mechanics, and cff-jilz workflow proof.

**2026-08-31T03:25:56Z**

Post-handoff scan additions: integrate=public save/load-to-update workflow; implement=planned child-ticket work; accumulate=new evidence from later trials; fit=registered family parameter estimation; intended=the locally stated classification meaning.

**2026-08-31T03:26:14Z**

Final classification addition: estimate=registered-family parameter estimation.
