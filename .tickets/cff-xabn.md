---
id: cff-xabn
status: open
deps: [cff-6tek, cff-s6yy]
links: []
created: 2026-08-31T03:16:28Z
type: feature
priority: 1
assignee: coopwilliams
parent: cff-cz6v
tags: [serialization, learning, schema]
---
# Version declared-learning semantics and persistent array schemas

## Background in Plain English — WHY

`LearnableSpec` on base commit `ca87069` records a family name, target blobs,
and statistic sources. The `Family` registry knows the hyperparameter keys,
sufficient statistics, and rollover rule in Python. That is enough to execute learning,
but it does not give every saved posterior a serialized, versioned identity.

The recurrence checkpoint demonstrates the failure mode. D8 and D9 used the
same factor name and array shapes but different sufficient statistics, so the
project added a recurrence-only manifest to prevent a semantically wrong
restore. This ticket generalizes that safeguard and the persistent array schema
to every declared learning family.

## Outcomes / Acceptance Criteria — WHAT

1. Every declared learnable resolves to one namespaced, versioned
   identity for its posterior update and sufficient-statistic semantics.
2. The serialized model records the semantic identity that determines learning
   meaning. The structure hash changes when that identity changes and remains
   unchanged when only numeric prior, posterior, or point values change.
3. The generic `Family` contract describes its persistent hyperparameter keys
   and validates their roles, shapes, and dtypes without bundle code branching
   on a family name.
4. A resolved persistent-state descriptor binds the learnable name, family,
   factor type, statistic sources, parameter targets, parameter forms, and
   semantic identity to named immutable-prior and current-posterior arrays.
5. Plate scoping, composition, transactional edits, copying, JSON, and
   structure hashing preserve the descriptor or reject an invalid result with
   a diagnostic naming the affected learnable.
6. Unversioned declarations emitted on base commit `ca87069` migrate
   deterministically to documented
   identities for unchanged families. An unknown or incompatible identity
   fails before any posterior or point parameter changes.
7. The D9 recurrence identity is expressible through the family-independent
   descriptor.
   D8 or unidentified recurrence state remains rejected and is never relabeled
   as D9 evidence.
8. Validation rejects missing identities, duplicate persistent keys, unsafe or
   ambiguous scoped names, incomplete multi-array schemas, incompatible target
   roles, invalid shapes, and noncanonical dtypes before archive writing.
9. Focused tests cover all five registered families, nested scoped names,
   pre-ticket declarations, legacy migration, unknown versions, tampering, and a
   model with at least 64 declared learnables.
10. `docs/ARCHITECTURE.md` documents semantic identity and the family-independent
    persistent-state descriptor. `README.md` documents migration diagnostics,
    and `docs/LIMITATIONS.md` records any family that cannot satisfy the
    descriptor.

## Methodological Suggestions — HOW

The implementer may choose another sound representation. Consider a versioned
field on `LearnableSpec` plus a family-owned descriptor method. Keep semantic
identity in model structure, but keep concrete array values, shapes, dtypes,
and digests in the bundle state inventory. Reuse the recurrence manifest's
ownership checks while removing its role as a one-family schema authority.

Use both the **New IR field** and **Learnable or checkpoint feature**
change-path rows. Review core validation, JSON/hash, plates, composition and
edits, compiler `_resolve_learnables`, `Arena.learn_plans`, the family registry,
checkpoint encode/decode, legacy migration, bundles, and public exports.
Host/JIT, flooding/scheduled, and single-instance/`vmap` numerical execution are
structurally unchanged because this ticket defines and validates persistence
semantics. The schema-current/legacy declaration and checkpoint pairs are directly
affected.

Hostile-input answers: sparse probability support and vacuous messages do not
enter schema validation. Invalid or incomplete state descriptions fail loudly.
Float32 descriptors preserve canonical dtype identity without numeric floors or
casts. The 64-learnable case checks namespace and validation scale rather than
message-rule numerics.

## Future work

`cff-enhs` depends on this ticket and owns archive storage and restoration.
Open recurrence ticket `cff-4c7t` also depends on this ticket so coefficient
owners extend one family-independent semantic descriptor instead of creating another
recurrence-only manifest. Future families under `cff-ahps` and neural work under
`cff-1o72` must supply compatible semantic identities when they join declared
model learning.

## Definitions

- **Canonical dtype:** NumPy's unambiguous dtype spelling, including byte order
  and element width.
- **Persistent-state descriptor:** Metadata that states which named arrays
  constitute one learnable's prior and posterior and what semantics produced
  them.
- **Semantic identity:** A namespaced and versioned identifier for the update
  rule and sufficient statistics that give posterior arrays their meaning.
- **Sufficient statistic:** A summary of inferred variables and observations
  used by a conjugate posterior update.

## Notes

**2026-08-31T03:24:55Z**

Retained bounded-scan candidates and intended meanings: canonical=normalized NumPy dtype spelling; generic=protocol-general Family contract; preserve=semantic identity, descriptor identity, or dtype identity; public=exported persistence symbols; register=runtime family registry coverage; reusable=code reuse of recurrence ownership guards; same=value identity for preexisting factor names and shapes; support=literal probability support, which is structurally outside schema validation; remaining=unchanged hash or deliberately rejected recurrence state; sufficient=mathematical sufficient statistics; check=input guard for ownership or namespaces; evidence=posterior evidence under D9 semantics; sound=reasonable representation choice; test=automated schema artifact or execution; validate=input/schema validation; infer=state beliefs summarized by statistics; learn=parameter-update primitive or declared workflow; learnable=serialized declaration or registered family; rollover=point-parameter installation; scale=64-item namespace size; may=optional implementation method; explicit_owner and named_ticket_handoff=cff-enhs and cff-4c7t as named; all=the enumerated five registered families; any=negative mutation scope or any unsupported family; at_least=inclusive 64-declaration lower bound; every=each declared learnable or family; only=numeric-data-only hash exclusion or recurrence-only historical manifest; current=runtime-active posterior or schema-current/legacy pair. Capability ownership was checked separately: this ticket owns semantic descriptors and migration, not archive storage or recurrence coefficient owners.

**2026-08-31T03:25:56Z**

Post-handoff scan additions: implement=the planned descriptor and migration work; appropriate=the named semantics-versus-storage ticket boundary; intended=the locally stated classification meaning.
