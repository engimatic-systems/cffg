---
id: c1-dn1y
status: closed
deps: []
links: []
created: 2026-08-20T04:20:25Z
type: task
priority: 2
assignee: coopwilliams
tags: [documentation, qa, paper-fidelity]
---
# Answer the 2026-08-20 CFFG architecture questions in Obsidian

## Background in Plain English — WHY

Cooper added three questions to the CFFG Obsidian Q&A for August 20. The answers need to distinguish the current implementation from proposed extensions and long-range research. They also need to explain the smallest practical value of the CFFG vision and why a CFFG runtime remains useful when an LLM can call Bayesian tools. The local question file lagged behind Obsidian Sync, so the latest synced version is the source text for this task.

## Outcomes / Acceptance Criteria — WHAT

1. The question about all 14 proposed features has one linked answer that states what exists, what each feature needs, and whether it requires a new first-class abstraction or an extension to an existing layer.
2. The question about the simplest distinctive CFFG construction has one linked answer that separates literal impossibility from the value of one reusable formalism.
3. The question comparing CFFG with an LLM and Bayesian tool libraries has one linked answer that states defensible advantages, complementary uses, and current limits.
4. Every answer is grounded in the current repository, the canonical CFFG papers, or both, and labels material uncertainty.
5. The August 20 question file links to each answer according to the Q&A folder convention.
6. The Obsidian artifacts are checked for complete backlinks, agent authorship metadata, and accidental edits outside the three answers and their links.

## Methodological Suggestions — HOW

These suggestions are optional. An implementer may choose another sound method that meets the outcomes.

- Audit the code and project-authoritative documents for implementation claims.
- Delegate bounded paper and code questions when that keeps unrelated source material out of the main context.
- Use the two papers to explain the constraint-language and generalized-free-energy distinction.
- Keep proposed items clearly separate from behavior exercised in this checkout.
- Treat the `docs/ARCHITECTURE.md` change-path index as not applicable: this task changes no repository code, schema, public API, execution path, or canonical project document. Host/JIT, flooding/scheduled, and single-instance/Colony pairs are all not applicable.

## Future work

No future work is identified. Architecture proposals discovered while answering stay explanatory unless Cooper asks for roadmap changes.

## Definitions

- **CFFG:** Constrained Forney-style Factor Graph.
- **First-class abstraction:** A named, serialized concept with its own input checks and execution contract, rather than an informal convention or one-off helper.
- **LAIF:** Lagrangian Active Inference.
- **Obsidian Q&A convention:** Each question keeps a compact backlink to a separate answer file in the linked-answers folder.

## Notes

**2026-08-20T04:27:48Z**

Research provenance, 2026-08-19.

Question: Which of proposed features 1–7 already have code foundations, and which need new abstractions?
Answer: Typed interfaces, scheduling/status, selector switching, and Template-based modules have substantial foundations. Pluggable local rules, neural surrogate semantics, general manifold domains, and generic gated fragments require named extensions at their respective layers.
Sources: repository checkout 800af9e; src/cffg/core/{ir,validate,compose,plates}.py, src/cffg/{kernels,schedule,engine}.py, docs/ARCHITECTURE.md, docs/LIMITATIONS.md.
Confidence: high.

Question: What layers own proposed features 8–14?
Answer: Collections need a modeling/lowering IR; structure learning and model discrimination belong in status-aware outer loops; higher-order operators, reified execution, learned compute policy, and reflection require a staged meta-language/control plane above the numerical engine.
Sources: checkout 800af9e; core edit/composition code, engine diagnostics, cff-cynx, cff-du6m, cff-gul8, cff-1o72, and docs/READING.md.
Confidence: high on current boundaries; moderate on exact future APIs.

Question: What is the smallest distinctive construction, and why is CFFG useful beside an LLM tool orchestrator?
Answer: One P-substitution bead on a future goal-observation composite is the smallest executable distinction; it adds epistemic value that ordinary factor multiplication does not express. Other systems can hand-code the same computation. CFFG’s value is the durable joint contract over model, constraints, messages, ledger, status, and state; an LLM is complementary as author and orchestrator.
Sources: Part I sections 3, 4, 5.5, 7, and 8; Part II sections 3.3.2, 4, 5.2, 6.3, and 9; tests/test_gfe.py, tests/test_tmaze.py, README.md, ARCHITECTURE.md, and LIMITATIONS.md.
Confidence: very high on the paper/repository mechanism and high on the bounded comparison.

**2026-08-20T04:28:04Z**

Completion handoff, 2026-08-19. Added three agent-authored Obsidian answers: 2026-08-20.1-Agent assesses all 14 architecture proposals against checkout 800af9e; .2 identifies the one-bead P-substitution epistemic chooser and limits the uniqueness claim; .3 explains the CFFG/LLM division of labor and current limitations. Added one backlink under each source question. Obsidian reports one backlink per answer, zero unresolved vault links, valid author: agent properties, and synced remote versions for all four edited/created notes. No repository code or canonical document changed, so the short pytest suite was not required. The ticket wiggle-word scan was reviewed in context; tk dependency-cycle and git diff-hygiene checks passed.
