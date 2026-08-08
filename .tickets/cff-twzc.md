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
# Prevent milestone surprises with intelligible tickets

## Background in Plain English

Milestone work has reached its later stages before revealing missing capabilities, unclear ownership, or unstated requirements. These surprises forced completed work to be revisited. The root cause was tickets that did not clearly state the intended outcome, the work that motivated it, or the work delegated to another ticket. This ticket changes how tickets are written and reviews every active roadmap ticket so gaps become visible before implementation begins.

The repository already contains a reproducible audit of words whose meaning can shift with context. It also contains a bounded query tool so an agent can inspect one ticket without loading the complete audit. The audit identifies wording that deserves review; the intelligible-ticket structure makes the ticket's background, owned outcomes, suggested methods, and delegated work explicit.

This ticket owns the ticket-writing policy, the dependency-aware rewrite campaign, and the tools that support that campaign. Each rewritten ticket continues to own the engineering outcomes stated in its acceptance criteria. Closed tickets remain historical references and are not rewritten.

## Outcomes / Acceptance Criteria

1. The version 1 and version 2 candidate lexicons and occurrence audits parse as YAML.
2. The version 1 and version 2 audits retain their recorded corpus and lexicon hashes.
3. Every raw audit match has a recorded classification.
4. The version 2 report satisfies its JSON Schema.
5. The query tool returns a bounded historical view for one ticket, one candidate, or the recorded problem classifications.
6. The query tool scans one current ticket against the frozen version 2 lexicon without regenerating the historical audit.
7. The machine-wide Codex and Claude installations contain byte-identical `intelligible-ticket` skill instructions.
8. Both machine-wide skill installations pass the skill validator.
9. `AGENTS.md` requires the `intelligible-ticket` skill when an agent writes or edits a ticket.
10. `AGENTS.md` requires a bounded current-ticket scan after a ticket is created or materially rewritten.
11. A repository tool identifies which active tickets are eligible for downstream-first review.
12. The scheduler withholds a shared prerequisite until every active direct dependent has a committed review.
13. Every ticket that is open or in progress at final validation follows the intelligible-ticket structure.
14. Every reviewed ticket records the current-ticket scan's retained candidates and their intended meanings.
15. Every reviewed ticket names the real ticket that owns each delegated outcome.
16. No closed ticket is rewritten by this campaign.
17. `cff-y4nx` distinguishes fixed-parameter oracle inference from fitting model parameters with the observations exposed to the agent by the declared experiment.
18. `cff-y4nx` retains `cff-twzc` as a prerequisite.
19. The final active-ticket dependency graph contains no cycle.
20. The audit validator succeeds.
21. The JSON Schema validator succeeds.
22. The scheduler's automated checks succeed.
23. The bounded slice and scan smoke commands succeed.
24. `git diff --check` succeeds.

## Methodological Suggestions

- Keep the historical audit pinned to the `cff-y4nx` branch corpus at commit `c855af6` so the two audit versions remain comparable.
- Treat the applicable architecture change path as the documentation and ticket process. Runtime, compiler, serialization, host/JIT, message scheduling, and checkpoint execution surfaces do not change.
- Implement the scheduler as a stateless command in `tools/`. Derive the active graph from `tk`, determine completed reviews from committed ticket content, and accept active assignments from the orchestrator.
- Start with active terminal tickets, which have no active dependents, and move toward their prerequisites.
- Use three persistent GPT-5.6 Sol workers. Prefer giving a worker an eligible prerequisite of the ticket it most recently completed so the worker retains useful dependent context.
- Keep a worker's context only while it improves the work. After one unacceptable draft, give that worker one specific correction; after a second unacceptable draft, retire the context and assign the ticket to a fresh GPT-5.6 Sol worker.
- Start a replacement worker from the assigned ticket and only the dependency tickets needed to reconstruct its context. Do not repeat the failed conversation history.
- Give each worker one ticket at a time. Permit it to read dependencies and closed historical tickets, but permit it to edit only the assigned ticket.
- Have each worker obtain the assigned ticket's bounded historical audit slice before rewriting it, then run the current-ticket scan after rewriting it.
- Require each worker to report the file it edited, the wording it clarified, the structural changes it made, the sources it inspected, the commands it ran, and the retained scan candidates with their intended meanings.
- Have the orchestrator review and commit one completed ticket before assigning that worker another ticket.
- At a shared prerequisite, give the selected worker compact reports from the other dependent branches instead of requiring it to reread every rewritten dependent.
- Treat two workers editing one ticket and a shared prerequisite released before all active dependents are committed as scheduler failures.
- Treat missing owners, unnamed comparison sets, mixed cold and warm performance claims, oracle-assisted validation presented as observed-only fitting, and a title broader than its acceptance evidence as wording failures.
- Derive the final validation commands from the artifacts changed. Ticket-only commits do not require runtime pytest.

## Future work

- `cff-y4nx` owns implementation and validation of the recurrent switching state-space model tier. This ticket owns only the clarity of that ticket's claims.
- Each other rewritten active ticket owns the engineering work in its outcomes. This ticket owns only the review and rewrite of those outcomes.
- No additional ticket-writing work is deferred beyond the active tickets covered by this campaign.

## Definitions

- **Active ticket:** A ticket whose `tk` status is `open` or `in_progress`.
- **Current-ticket scan:** A scan of one ticket's present wording against the frozen version 2 candidate lexicon.
- **Direct dependent:** An active ticket that names another ticket as an immediate prerequisite.
- **Fan-in barrier:** The rule that a shared prerequisite cannot be assigned until every active direct dependent has a committed review.
- **Historical audit slice:** The bounded audit records associated with one ticket in the pinned corpus.
- **Intelligible-ticket structure:** The required background, outcomes, methodological suggestions, future work, and definitions sections supplied by the machine-wide skill.
- **Oracle inference:** Inference evaluated with parameters or latent values supplied by the data-generating process rather than learned from the agent's observations.
- **Terminal ticket:** An active ticket with no active direct dependent.

## Notes

**2026-08-07T15:22:31Z**

Handoff 2026-08-07. Draft PR #57 on agent/cff-twzc-ticket-semantics, based on agent/cff-4apw-pg-learning. V1 and v2 lexical/context audits are committed at ba672d7 and compare the identical cff-y4nx corpus snapshot c855af6: v1 65 candidates/3,651 occurrences/28 ambiguous/238 unlisted/14 false positives; v2 87 candidates/4,300 occurrences/7 ambiguous/5 unlisted/10 false positives. The seven survivors are structural rather than missing vocabulary. Next: add the mandatory ownership/claim-layer/validation/N/A schema to AGENTS.md, review all open/in-progress tickets, and rewrite cff-y4nx before rebasing PR #56. Preserve PR #56 implementation and experiment evidence; do not restart from scratch. Gotcha: the audit corpus is intentionally pinned to the child commit c855af6 even though this process PR is upstream, so do not regenerate reports from the upstream working tree. Existing unrelated local AGENTS.md edits remain unstaged.

**2026-08-07T16:03:30Z**

Navigation increment complete. Added the v2 JSON Schema, bounded slice/scan/summary/validate commands, the downstream-first assignment protocol with a fan-in barrier, and the mandatory post-authoring scan rule. Dogfood scan before this note found 69 retained occurrences across 33 candidates. Intended-meaning groups: metalinguistic claim labels (end_to_end, generic, n_a, implement, deliver, validate, learn, performance); artifact and status claims (fixed=constant_parameter, preserve=semantic_invariant, same=object_identity, provide=public_api, complete=exhaustive, remaining=residual_set or preserved_state, close=work_item_completion, current/existing=repository_at_reference, later=ordering_constraint); evidence claims (check=automated_check, evidence=provenance_artifact, exact=byte_or_value_identity, oracle=generating_parameter_oracle, pass=algorithmic_pass or non_test_check_succeeds, reproduce=rerunnable_artifact, fit=estimate_parameters, infer=state_or_regime_posterior); and explicitly bounded ownership or quantifiers (explicit_owner=current_ticket_owner, all/every=named_class, only=ticket_scope_boundary, intended=semantic_purpose). Work and match are unambiguous Git-working-tree and regex-match senses; test is the pytest command name. The refinement pass changed the fan-in sentence so a shared prerequisite waits for every open direct dependent and receives their aggregated downstream requirements. Validation: draft-2020-12 schema validation passed; the internal validator reconciled 87 candidates, 4,300 occurrences, and 1,907 candidate-ticket pairs; slice and current-scan smoke commands, py_compile, uv lock --check, tk dep cycle, and git diff --check passed. Per Cooper, runtime pytest is excluded for this standalone tool; the already-started suite was interrupted after 961 passes and is not claimed as final validation. The broader ownership-ledger ticket refinement remains next.

**2026-08-08T03:14:16Z**

Installed the Obsidian ticket-writing guidance as the machine-wide intelligible-ticket skill for Codex and Claude. Both copies validate and are byte-identical. AGENTS.md now requires the skill with one sentence. Validation: tk dep cycle and git diff --check passed; runtime pytest is N/A for this documentation-only increment.

**2026-08-08T03:37:28Z**

Cooper reconciled the legacy Plain English rule with the intelligible-ticket skill by removing the obsolete required heading from the general tk rule. Validation: tk dep cycle and git diff --check passed; runtime pytest is N/A for this documentation-only increment.

**2026-08-08T03:56:54Z**

Controller rewrite prepared; the campaign remains in progress. The bounded current scan found 148 occurrences across 40 candidates, including historical Notes. Retained meanings in the rewritten policy sections: fixed=constant_parameter; expose=public_api; implement=planned_work or non_code_delivery; support=internal_component_integration; complete=execution_finished, exhaustive, or ticket_scope_complete; remaining=preserved_state; close=work_item_completion; material=scope_impact; check=automated_check or acceptance_gate; evidence=provenance_artifact; match=literal_regex_match; oracle=generating_parameter_oracle or oracle_assisted_workflow; pass=non_test_check_succeeds; reproduce=rerunnable_artifact; test=automated_execution; validate=artifact_validation or acceptance_validation; fit=estimate_parameters; infer=state_or_regime_posterior; learn=parameter_update_primitive; performance=mixed_metrics_or_phases; can=demonstrated_capability or permission; intended=semantic_purpose; explicit_owner=current_ticket_owner or named_ticket_owner; named_ticket_handoff=concrete_future_owner; all=named_class_exhaustive; every=named_class_distributive; only=ticket_scope_boundary or minimization; current=repository_at_reference; deferred=scope_exclusion; later=historical_sequence; read=future_requirement. Validation: bounded slice and scan commands, tk dep cycle, and git diff --check passed. Runtime pytest is N/A for this ticket-only rewrite.

**2026-08-08T04:05:28Z**

Dependency scheduler increment complete. tools/ticket_rewrite_scheduler.py derives the active graph from tk, reads only committed ticket content when recognizing completed reviews, excludes active assignments, prefers a direct prerequisite of the worker last ticket, and withholds shared prerequisites behind the fan-in barrier. Focused validation: 6 tests passed. Final short suite after the final executable change: 1,288 passed, 38 deselected, zero failures; dashboard and runtime history refreshed. The live frontier contains 14 eligible tickets among 28 campaign tickets. tk dep cycle and git diff --check passed.
