---
id: cff-enhs
status: open
deps: [cff-xabn, cff-yoxz]
links: []
created: 2026-08-31T03:16:28Z
type: feature
priority: 1
assignee: coopwilliams
parent: cff-cz6v
tags: [serialization, learning, bundle]
---
# Store complete declared-learning state in CFFGZ bundles

## Background in Plain English — WHY

The bundle archive already stores `model.json`, point-parameter blobs, and
hashes. `Learnables.state_dict()` separately stores current posterior
hyperparameters, while `Learnables.initial_params` retains immutable priors
only in memory. A user who loads `.cffgz` must therefore supply priors again or
start a different learning history.

Ticket `cff-xabn` supplies semantic and array descriptors shared by every
registered family. This ticket makes those descriptors and arrays part of the
bundle format and public save/load workflow.

## Outcomes / Acceptance Criteria — WHAT

1. A schema-current archive contains `model.json`, the existing point-parameter
   blob inventory, a declared-learning inventory, immutable-prior arrays, and
   current-posterior arrays under collision-free member names.
2. The manifest binds every learning array to its learnable, prior-or-posterior
   role, hyperparameter key, semantic identity, shape, canonical dtype, and
   exact-value digest.
3. Saving a template with declared learnables requires one complete compatible
   learning state. Missing priors, missing posterior keys, extra keys, unknown
   identities, invalid values, or point blobs whose inventory, shapes, or dtypes
   disagree with their template declarations fail before the destination is
   created or replaced.
4. Saving a template without declared learnables writes an explicit empty
   learning inventory and otherwise retains fixed-parameter bundle behavior.
5. Loading validates ZIP member uniqueness and safety, model structure, point
   blobs, learning descriptors, all hashes, every prior and posterior, and
   cross-references before returning a usable object.
6. The loaded object exposes the template and point arrays and reconstructs
   `Learnables` with both immutable priors and current posterior values without
   caller-supplied learning arrays.
7. A save operation observes one committed learning boundary. It either stores
   a mutually compatible model, prior, posterior, and point state or fails
   atomically; it never records a partly applied update.
8. Original schema-1 model-only bundles remain loadable and are explicitly
   reported as non-resumable. Loading does not infer posterior hyperparameters
   from point estimates or fabricate immutable priors.
9. Scoped and plate-expanded names have reversible member encoding and cannot
   collide with model blobs, manifest members, or another learning-state key.
10. Tampered, missing, duplicate, non-finite, wrong-shape, wrong-dtype, unknown-
    version, and model/state-mismatch cases fail before runtime mutation.
11. Exact zero array entries round-trip bit-for-bit; float32 arrays retain their
    dtype; and an archive with at least 64 learnables loads without family-
    specific bundle branches.
12. `README.md` documents the public save/load and legacy capability check.
    `docs/ARCHITECTURE.md` documents member layout and ownership.
    `docs/LIMITATIONS.md` distinguishes next-trial learning resume from an exact
    inference or agent-runtime snapshot.

## Methodological Suggestions — HOW

The implementer may choose another sound API. Consider a first-class loaded
bundle object that carries template, point arrays, learning state, manifest,
and an explicit capability flag. Preserve a compatibility wrapper when it can
remain unambiguous; do not hide learning state inside the point-array mapping.
Write to a temporary sibling, validate it through the loader, and replace the
destination atomically.

Use the **Learnable or checkpoint feature** change path and the **New IR field**
path if `cff-xabn` adds serialized fields. Review bundle and checkpoint
encode/decode, legacy migrations, public exports, agent handoff, and the
compiler plans used to reconstruct `Learnables`. Message updates and free
energy do not change. Loaded point arrays must remain consumable by host/JIT,
flooding/scheduled, and single-instance/`vmap` inference; the persistence code
itself is host-side. Schema-current/legacy bundle and checkpoint pairs are directly
affected.

Hostile-input answers are criteria 10 and 11. Sparse support is preserved as
literal array values. Improper priors and posteriors are rejected by their
family validators. No numeric floor, ridge, or dtype conversion is introduced.
The 64-learnable archive exercises inventory scale and scoped-key handling.

## Future work

`cff-jilz` depends on this ticket and owns consumer-workflow resume equivalence
for every registered family. It also updates the test from closed ticket
`cff-yoxz` so the fixed-path inspection specimen is regenerated in the new
format.
Exact `RollingAgent`, hierarchy, and neural-optimizer snapshots remain with
`cff-t2g3`, `cff-pyy9`, and `cff-1o72`.

## Definitions

- **Capability flag:** Loaded metadata that says whether an archive contains a
  complete learning-resume state or only a fixed-parameter model.
- **Collision-free member name:** A reversible ZIP path encoding for which two
  logical state keys cannot produce the same archive path.
- **Point parameter:** The numeric array consumed directly by inference rules.
- **Posterior hyperparameter:** A parameter of a probability distribution over
  a learned model parameter, such as Dirichlet concentration or MNIW scale.
- **Schema-1 bundle:** The original model-only `.cffgz` layout.

## Notes

**2026-08-31T03:25:11Z**

Retained bounded-scan candidates and intended meanings: canonical=normalized dtype spelling; fixed=constant-parameter bundle; preserve=compatibility or bitwise sparse values; public=documented consumer save/load workflow or exported symbol; register=the enumerated runtime family coverage delegated to cff-jilz; same=archive-path equality prohibited for distinct keys; expose=public loaded-artifact fields; support=literal probability support represented by arrays; complete=exhaustive required learning state; remaining=legacy compatibility or deliberately excluded runtime snapshots; close=historical cff-yoxz work-item completion; check=legacy capability or input guard; equivalent=consumer-workflow next-update equality owned by cff-jilz; estimate=point parameter estimate from which posterior history must not be derived; exact=digest byte identity, literal zero, or exact runtime replay boundary; sound=reasonable API choice; test=the existing specimen test; validate=artifact and input validation; infer=forbidden metadata derivation from point values or fixed-parameter inference; learn=new-evidence accumulation and persistent workflow; learnable=serialized declaration or registered family; trial=next independent dataset; scale=64-item inventory size or the named MNIW scale hyperparameter; can=conditional API compatibility; may=optional API design; explicit_owner and named_ticket_handoff=cff-jilz for workflow proof; all=all manifest hashes and declared prior/posterior arrays; at_least=inclusive 64-learnable lower bound; every=each inventory member or registered family; only=in-memory or model-only limitation; current=runtime-active posterior or schema-current/legacy pair; existing=pre-ticket point-blob namespace. Capability ownership was checked separately: this ticket owns archive mechanics and restore construction, not semantic identity or end-to-end family proof.

**2026-08-31T03:25:56Z**

Post-handoff scan additions: end_to_end=consumer workflow delegated to cff-jilz; appropriate=the named archive-versus-workflow ticket boundary; evidence=observations retained in a posterior; accumulate=new evidence under the later workflow; intended=the locally stated classification meaning.

**2026-08-31T03:26:14Z**

Final classification addition: later=ordered downstream workflow.
