---
id: cff-twzc
status: in_progress
deps: []
links: []
created: 2026-08-07T15:20:31Z
type: chore
priority: 1
assignee: coopwilliams
tags: [process, tickets, ambiguity]
---
# Ticket semantics: audit wiggle words and require explicit capability ownership

## Plain English

Make ticket capability claims mechanically reviewable instead of relying on overloaded prose. Preserve a reproducible two-pass vocabulary audit and provide bounded ticket views so agents never need the complete report in context. Require future tickets to state capability ownership, implementation layer, and evidence boundaries explicitly, then apply that structure to the remaining roadmap tickets before implementation resumes.

## Scope

The audit is pinned to the cff-y4nx branch corpus at c855af6 so v1 and v2 remain comparable. The refinement covers every open or in-progress ticket; closed tickets remain historical evidence except where an active ticket depends on interpreting one. This ticket owns ticket-language policy and ticket rewrites, not rSLDS implementation.

## Design

Applicable architecture change path: documentation/ticket process; runtime, compiler, serialization, host/JIT, scheduling, and checkpoint surfaces are N/A because no executable model behavior changes.

Each remaining ticket gets a capability-ownership ledger. Every referenced capability is classified as delivered prerequisite, owned here, delegated future, external, or unowned; non-current owners name a ticket or authority, and ticket relations agree with tk dependency direction. Reserved claim words such as end-to-end, generic, learning, implemented, validation, performance, and N/A receive the structured boundary required by their measured ambiguity.

The audit has a JSON Schema expressed as YAML and a query tool that projects one ticket, one candidate, or only sentinel classifications. The same tool scans a current ticket against the frozen lexicon after authoring, because the pinned contextual audit cannot contain later prose.

Roadmap refinement proceeds from open terminal tickets, which have no open or in-progress dependents, toward their prerequisites. One agent edits one assigned ticket at a time. A shared prerequisite is not assigned until each open or in-progress direct dependent has been refined and reported its aggregated downstream requirements; the top-level agent combines those reports into the assignment.

Numerical hostile-input cases are structurally N/A. Process-hostile cases are missing owners, list-level N/A dispositions, unnamed comparison universes, oracle-assisted validation, mixed cold/warm performance claims, a title whose scope exceeds its acceptance evidence, two agents editing one ticket, and a shared prerequisite refined before every downstream requirement arrives.

## Acceptance Criteria

1. V1 and v2 candidate lexicons and occurrence audits parse as YAML, retain exact corpus/lexicon hashes, and reconcile every raw match. The v2 report validates against its JSON Schema.
2. A command returns bounded historical slices by ticket, candidate, or sentinel classification, and a second command scans one current ticket without regenerating the pinned audit.
3. AGENTS.md requires the current-ticket scan and contextual refinement after ticket creation or material rewriting.
4. AGENTS.md defines required capability ownership, claim-layer, N/A-disposition, and validation-boundary fields without relying on prose keywords alone.
5. Every open or in-progress ticket is reviewed against the structure; applicable tickets are updated in place, and every delegated capability has a named accountable ticket or is explicitly unowned.
6. cff-y4nx is narrowed or expanded unambiguously, depends on this process ticket, and distinguishes fixed-parameter oracle inference from observed-only learned fitting.
7. tk dependency records agree with the ownership ledgers; the audit validator, schema validation, slice/scan smoke commands, tk dep cycle, and git diff --check pass. Runtime pytest is deliberately excluded because the query tool does not import or modify package runtime code.

## Notes

**2026-08-07T15:22:31Z**

Handoff 2026-08-07. Draft PR #57 on agent/cff-twzc-ticket-semantics, based on agent/cff-4apw-pg-learning. V1 and v2 lexical/context audits are committed at ba672d7 and compare the identical cff-y4nx corpus snapshot c855af6: v1 65 candidates/3,651 occurrences/28 ambiguous/238 unlisted/14 false positives; v2 87 candidates/4,300 occurrences/7 ambiguous/5 unlisted/10 false positives. The seven survivors are structural rather than missing vocabulary. Next: add the mandatory ownership/claim-layer/validation/N/A schema to AGENTS.md, review all open/in-progress tickets, and rewrite cff-y4nx before rebasing PR #56. Preserve PR #56 implementation and experiment evidence; do not restart from scratch. Gotcha: the audit corpus is intentionally pinned to the child commit c855af6 even though this process PR is upstream, so do not regenerate reports from the upstream working tree. Existing unrelated local AGENTS.md edits remain unstaged.

**2026-08-07T16:03:30Z**

Navigation increment complete. Added the v2 JSON Schema, bounded slice/scan/summary/validate commands, the downstream-first assignment protocol with a fan-in barrier, and the mandatory post-authoring scan rule. Dogfood scan before this note found 69 retained occurrences across 33 candidates. Intended-meaning groups: metalinguistic claim labels (end_to_end, generic, n_a, implement, deliver, validate, learn, performance); artifact and status claims (fixed=constant_parameter, preserve=semantic_invariant, same=object_identity, provide=public_api, complete=exhaustive, remaining=residual_set or preserved_state, close=work_item_completion, current/existing=repository_at_reference, later=ordering_constraint); evidence claims (check=automated_check, evidence=provenance_artifact, exact=byte_or_value_identity, oracle=generating_parameter_oracle, pass=algorithmic_pass or non_test_check_succeeds, reproduce=rerunnable_artifact, fit=estimate_parameters, infer=state_or_regime_posterior); and explicitly bounded ownership or quantifiers (explicit_owner=current_ticket_owner, all/every=named_class, only=ticket_scope_boundary, intended=semantic_purpose). Work and match are unambiguous Git-working-tree and regex-match senses; test is the pytest command name. The refinement pass changed the fan-in sentence so a shared prerequisite waits for every open direct dependent and receives their aggregated downstream requirements. Validation: draft-2020-12 schema validation passed; the internal validator reconciled 87 candidates, 4,300 occurrences, and 1,907 candidate-ticket pairs; slice and current-scan smoke commands, py_compile, uv lock --check, tk dep cycle, and git diff --check passed. Per Cooper, runtime pytest is excluded for this standalone tool; the already-started suite was interrupted after 961 passes and is not claimed as final validation. The broader ownership-ledger ticket refinement remains next.
