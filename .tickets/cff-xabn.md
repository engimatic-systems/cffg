---
id: cff-xabn
status: in_progress
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

**2026-09-12T23:47:34Z**

Implementation handoff: using the New IR field and Learnable or checkpoint feature change-path rows. Affected: core declaration validation and JSON/hash; plates/composition and transactional edits/copies; compiler resolution and LearnPlan transport; Family persistence schema and Learnables checkpoint encode/decode; bundle pre-write declaration validation; public descriptor exports. Topology keeps declarations through the existing template snapshot; member_of supplies existing target addresses. Host/JIT, flooding/scheduled, and single-instance/vmap numerical execution and agent cadence: N/A (no message rules, statistics, or scheduling changes). Current/legacy declaration and checkpoint decoding are affected and need separate tests. Sparse/zero support and vacuous messages: N/A to metadata validation; invalid schemas fail before mutation. Float32: preserve canonical dtype without casting in persistence. Scale: exercise at least 64 scoped declarations. Working on the existing clean agent/cff-xabn-retry branch as Cooper requested; no branch creation or base switch.

**2026-09-13T00:03:33Z**

Research question: Which recurrence guards must generic persistence preserve? Answer: the older D9 manifest binds version, statistic identity, source/target/blob ownership, complete hyperparameter shapes, canonical dtype, and exact-value digests; unidentified v2/slash state and D8 remain rejected. Sources: learning_checkpoint.py, Learnables.load_state_dict, _agent_checkpoint.py, and test_recurrence_checkpoint.py. Confidence: high (read-only sub-agent implementation/test inspection). Integration found that agent inventories derive required fields from emitted learning state, so older envelopes need an explicit optional-metadata path; historical hashes are accepted only for the four unchanged family identities. The new descriptor additionally names each logical target factor. Discovered compact-plate composition silently discarded plate learnables: this ticket rejects that operation with affected names and an expand_plates diagnostic; cff-b2fz owns implementing compact composition.

**2026-09-13T00:06:41Z**

Implementation complete for review: semantic identity is transported by LearnableSpec and LearnPlan; the family-owned persistent-parameter schema resolves logical target factors/blobs/forms and named prior/posterior arrays. Generic checkpoints bind semantics, exact dtypes, shapes, and digests; old identified D9 and unchanged-family declaration/checkpoint/hash migration remain explicit. Descriptor construction and encoding validate before writing. Focused validation: 112 passed, 3 slow tests deselected across learning schemas, declared/ Gaussian learning, and bundles; recurrence/hybrid/plate/composition run passed 102. Functional-agent corruption regression passed with recomputed metadata so it still exercises stale planning-state rejection. Topology clarification: TopologyLayout snapshots factor/variable topology, not learning declarations; it needs no new field. Compiler resolution reads the expanded Template directly, then stores identity in Arena.learn_plans. Full short-suite gate remains pending.

**2026-09-13T00:10:50Z**

Final API audit updated the canonical-export test and the renamed-recurrence checkpoint ownership test for the generic manifest; both pass (2 passed in 8.72 s). The first short-suite attempt was stopped before completion when this audit found those stale expectations. It produced no gate result or dashboard update. One fresh short-suite run is now validating the final code and tests; the previous 1003.707 s wall-time measurement and 1260 s wait budget remain the basis for the wait.

**2026-09-13T00:28:47Z**

The first completed short suite reported 1734 passed, 3 failed, 41 deselected (989.07 s pytest; 1001.601 s measured wall time). All three failures were compiler-contract assertions pinning the pre-identity learning/agent structure hashes. Updated exactly those two expected hashes; schedule fingerprints and execution-layout expectations are unchanged. All 44 compiler-contract tests now pass in 2.08 s. A fresh final short-suite run will replace the failing dashboard result; no focused result is being represented as a full-suite pass.

**2026-09-13T00:46:09Z**

Completed on agent/cff-xabn-retry. Implementation commit: 534d783. All five registered families now declare versioned semantics and complete persistent array schemas. JSON/hash, compiler plans, scoped declarations, immutable-prior/current-posterior descriptors, generic checkpoint validation, legacy D9 and unchanged-family migration, agent envelopes, bundle pre-write validation, and public exports agree. Canonical docs describe migration and storage scope.

Final gate after the last test edit: uv run python tools/dashboard.py --run-suite passed 1737 tests, with 41 slow tests deselected and 4 existing dependency deprecation warnings. Pytest took 985.55 s; measured wall time was 998.071 s. The runtime history and ~/agent-share/cffg-tests.html are refreshed with zero failures. Log: /tmp/cff-xabn-short-suite-pass.log. Focused compiler-contract validation passed all 44 tests; schedule fingerprints and execution-layout assertions remain unchanged. No numerical rules changed, so slow behavioral runs were not needed.

No work remains for cff-xabn. Downstream cff-enhs owns complete CFFGZ learning-state storage/restoration. Discovered cff-b2fz owns compact-plate composition; until then, expand_plates before scoping or merging. The current flat checkpoint API still restores posteriors while retaining destination immutable priors, as documented. Main was not changed; Cooper requested this existing branch.

**2026-09-13T01:07:45Z**

Published PR #137 against main: https://github.com/engimatic-systems/cffg/pull/137 . The PR uses the existing 1737-passing short-suite result; subsequent changes only record validation/publication metadata and normalize ticket formatting. Final branch diff whitespace check and ticket dependency cycle check pass. No code changes or unchanged suite rerun were needed to publish.

**2026-09-13T01:43:34Z**

PR #137 bounded review fixes (comments 3998216389 and 3998216393): preserve verified original schema-1 recurrence bundles as model-only/non-resumable inputs without inferring posterior evidence; restore complete selected learnables under exact=False while exact=True still requires all destination learnables. Applicable change-path rows remain New IR field and Learnable or checkpoint feature. Affected pairs: current/legacy bundle parsing/hash checks and current/legacy checkpoint completeness; host/JIT, flooding/scheduled, single-instance/vmap numerics, parameter writes, and agent cadence are N/A. Both changes keep staged validation before mutation. Hostile inputs: missing selected arrays, extra undeclared arrays, tampered identity/digest, sparse fixed point values, and float32 preservation; 64-owner schema scale is already covered. Genuine base-revision bundle production is delegated as bounded compatibility research. One atomic commit per fix; final short-suite run after focused tests, then wait the complete suggested_wait_seconds without polling, per updated AGENTS.md.

**2026-09-13T01:46:09Z**

Review fix 1: generic manifest validation now accepts a subset of known owners when load_state_dict(exact=False), while exact=True retains complete-destination validation. Every supplied descriptor and posterior remains complete; arrays without supplied descriptors are rejected before mutation. New regressions cover selected NIW/MNIW posteriors under nested scopes, untouched unrelated posteriors, exact=True rejection, missing arrays/schema keys, extra owners, dtype tampering, and unknown identities. Focused schema/Gaussian/recurrence checkpoint run: 82 passed in 23.03 s. Final full gate remains pending both bounded fixes.

**2026-09-13T01:53:28Z**

Review fix 2: load_bundle now verifies the original stored model structure hash before selecting current declarations for a model-only archive, then retains blob inventory/digest verification. Old recurrence point parameters load for fixed-parameter inference; the returned manifest explicitly reports resumable=False. Standalone unversioned recurrence JSON and D8/unidentified posterior restoration remain rejected. Canonical documentation now distinguishes these paths.

Research question: can a genuine pre-change recurrence bundle reproduce fixed inference without assigning posterior provenance? Answer: yes. A research sub-agent archived exact base 4d7902317ef5007f90a36ea59bb276cf6866bfcb, ran that revision's rslds and production save_bundle/load_bundle in the existing uv-managed environment, and independently reproduced its inference probe twice. Source: that revision's src/cffg/bundle.py and src/cffg/models/rslds.py; checked-in artifact tests/fixtures/rslds-base-4d790231.cffgz (SHA256 c0adf08883f4e6e283dce3af8ac890d51049b15211afbb58400f6bc42e0fe58f). Confidence: high, direct execution. The regression uses the exact point values from test_recurrence_checkpoint._model, verifies the original archive hash and absence of posterior metadata, and reproduces the base free energy and regime marginal. Additional regressions reject model/blob tampering and attempts to restore D8 or unidentified recurrence posteriors.

Focused validation after the final relevant edit: uv run pytest -q tests/test_learning_schema.py tests/test_recurrence_checkpoint.py tests/test_bundle.py tests/test_serialize.py tests/test_compiler_contracts.py — 139 passed, 1 slow test deselected in 9.41 s. No new test files; dashboard EPIC_FILES already covers these files. Full short-suite gate remains pending. No numerical rules changed, so slow behavioral validation is not needed.

**2026-09-13T02:02:50Z**

Cooper requested an explicit AGENTS.md exception to general frequent-progress guidance during test-suite waits. The suite protocol now says to announce launch and budget once, then wait the full suggested_wait_seconds without periodic commentary or process/log/dashboard polling; user messages still receive responses. This documentation-only change was made while the already-launched final gate runs and does not invalidate that code/test result. Whitespace checks pass.

**2026-09-13T02:15:32Z**

Both bounded PR #137 requests are complete in separate commits: 31d4251 restores complete selected posteriors under exact=False; 97824a2 preserves verified legacy model-only recurrence bundles as non-resumable inputs. Cooper's requested quiet-wait exception is committed in c82ea33.

Final validation after the last code/test change: uv run python tools/dashboard.py --run-suite passed 1747 tests, with 41 slow tests deselected and 4 existing dependency deprecation warnings. Pytest took 993.36 s; measured wall time was 1005.804 s. The dashboard and runtime history are refreshed with zero failures. Log: /tmp/cff-xabn-review-short-suite.log. The suite was launched once and its output was first inspected after the full 1260-second suggested wait; no periodic suite polling occurred. Focused persistence validation passed 139 tests before launch.

No implementation work remains for these requests. Model-only archives still do not contain learning state; cff-enhs owns complete archive storage/restoration. D8 and unidentified recurrence posteriors remain rejected. Flat selected restores retain destination priors and untouched owners. cff-b2fz remains the separate compact-composition follow-up. Closing cff-xabn and updating PR #137 on Cooper's existing branch; main remains unchanged.

**2026-09-13T02:41:45Z**

Cooper authorized merging PR117–134 followed by PR137. The stack is now merged into main with the final reviewed PR134 file tree preserved exactly. Integration uses the New IR field and Learnable or checkpoint feature change-path rows. Move PR137's learnable role checks, logical factor names, semantic identity, and persistent schema validation from the old compiler body into resolution/addresses.py::resolve_learnables, retaining the stack's ParameterRuntime declarations and statistic-source selection. Compiled values and downstream arena assembly keep transporting the resolved fields; core JSON/hash and current/legacy persistence keep their reviewed behavior. Host/JIT, flooding/scheduled, and single-instance/vmap execution changes are N/A: no numerical rules change. Validation will cover resolver/assembly contracts and learning/checkpoint migration, followed by one fresh full short-suite run for the combined code. Retain the full-stack runtime measurement to budget that run.

**2026-09-13T02:43:36Z**

Resolved the compiler conflict by preserving main's facade and moving the schema additions into resolution/addresses.py. The final main tree was verified identical to the reviewed PR134 tree before this integration. Focused validation of resolver, assembly, compiled values, compiler contracts, all learning families, schemas, bundles, and recurrence checkpoints passed: 295 passed, 3 slow tests deselected in 69.60 seconds. The existing regression tests cover both review fixes on the merged architecture. git diff --check and tk dep cycle pass. Full short-suite validation is next; its runtime budget comes from the completed architecture stack measurement.
