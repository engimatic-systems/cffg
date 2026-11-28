---
id: cff-umag
status: closed
deps: []
links: []
created: 2026-09-09T03:57:33Z
type: task
priority: 2
assignee: coopwilliams
tags: [tooling, context]
---
# Reduce wiggle-audit output and token use

## Background in Plain English — WHY

The ticket wording audit repeats candidate definitions and occurrence details,
producing far more text than an agent needs to review a ticket. Cooper asked
us to stop requiring it and optimize its token use before reconsidering the
workflow. The tool remains available for this improvement work.

## Outcomes / Acceptance Criteria — WHAT

1. The default review output groups repeated matches and omits repeated
   definitions while retaining ticket IDs, line references, and enough context
   to assess each potentially ambiguous claim.
2. Full occurrence details are available through an explicit detail option.
3. A comparison on a short ticket and a long ticket records the command,
   tokenizer/version, and output-token count before and after the change;
   the default output uses at least 75% fewer tokens on both fixtures.
4. Compact output states when it omits or truncates results and explains how
   to retrieve those details. Existing machine-readable modes retain their
   documented contract or receive an explicitly documented version change.
5. Repository instructions no longer require running the audit. Optimizing
   the tool does not reinstate that requirement.

## Methodological Suggestions — HOW

These suggestions are optional; another method may satisfy the outcomes.

- Start with the scan renderer in
  tools/wiggle_words/query_ticket_wiggle_audit.py.
- Use line-grouped matches, one summary, and opt-in candidate definitions.
- Select representative fixtures before measuring output reduction.

## Future work

No further work is identified. Reintroducing a mandatory audit would require
a separate decision by Cooper.

## Definitions

- **Token:** A unit of text consumed by a language model.
- **Tokenizer:** The algorithm used to count those units.
- **Candidate:** A word or phrase the tool flags for contextual review.

## Notes

**2026-09-09T03:57:48Z**

Removed the mandatory scan/slice/rerun instructions from AGENTS.md at Cooper’s request. Historical review evidence and the tool remain intact. Output optimization is still open; no audit commands ran for this change. Validation: tk show, tk dep cycle, and git diff --check.

**2026-11-28T04:02:39Z**

Before implementation: the renderer and schema are in this repository under tools/wiggle_words/. The generated contextual audit is an optional ignored local input. No prior implementation or paired tokenizer counts were found. The current scan and the historical slice default to YAML; explicit JSON and YAML, summary and validate remain compatibility surfaces. The architecture change-path index has no runtime row for an offline ticket formatter. Host/JIT, flood/scheduled, single/batch and checkpoint paths are N/A.

Preselected scan fixtures at a774a54: cff-hvz2 (57 lines, a short real workflow ticket) and cff-du6m (215 lines, a long research ticket). Compare complete stdout including newlines before and after with tiktoken 0.13.0 o200k_base, reporting exact counts for this encoding. No tokenizer is prescribed in the ticket or installed audit; this fixed, reproducible encoding is a measurement choice. Check grouped output, access to full detail and exact machine-output bytes separately.

**2026-11-28T04:07:14Z**

The scan now prints each matched source line once with its ticket line number and a count per repeated candidate. The historical slice groups archived classifications on their recorded line, with section, rationale, observed meaning when present, and the pinned corpus revision. Both commands identify what they omit and how to retrieve it. Every matched line is shown in full. Append `--detail` for the former default YAML, or use `--format yaml` / `--format json` before the command. Summary and validate retain their defaults. Running this audit remains optional.

Measured complete UTF-8 stdout including the final newline. The reference was the unmodified default `scan` at a774a54 (tool blob fbbe05ff); the candidate ran on identical lexicon and ticket bytes. PyYAML 6.0.3, tiktoken 0.13.0, encoding o200k_base with encode_ordinary:
- cff-hvz2 (57 lines, git blob 8f367871): 92 hits on 18 lines; 16,978 → 1,298 tokens (92.35% reduction).
- cff-du6m (215 lines, git blob c742f165): 204 hits on 104 lines; 18,734 → 4,195 tokens (77.61% reduction).
For each fixture use `uv run --no-project --with pyyaml==6.0.3 python tools/wiggle_words/query_ticket_wiggle_audit.py scan TICKET > output.txt` at both revisions. Count with `uv run --no-project --with tiktoken==0.13.0 python -c "import pathlib,tiktoken; print(len(tiktoken.get_encoding('o200k_base').encode_ordinary(pathlib.Path('output.txt').read_text())))"`. At these ticket revisions the new --detail output also matched the old default byte for byte.

Verification: the baseline JSON independently accounted for every displayed line, complete line text, candidate ID and multiplicity. On the historical v2 audit from Git blob 8203c0eb, slice cff-y4nx, its --problems-only filter, summary and validate matched old JSON and YAML stdout byte for byte. Corrupted audit validation retained exit 1 and identical JSON; missing ticket retained exit 2 and identical stderr. The YAML schema changed only in its x-navigation usage annotation. `uv run --no-project --with pyyaml==6.0.3 python tools/wiggle_words/query_ticket_wiggle_audit_check.py` passed both focused CLI regressions. tk dep cycle and git diff --check passed. This change qualifies as isolated tooling: no numerical, model, serialization, dependency, suite-discovery, gate interpretation or workflow-instruction changes. No full suite was run.
