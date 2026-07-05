# AGENTS.md — Ground Rules for the CFFG Project

This repository implements **Constrained Forney-style Factor Graphs (CFFGs)** and
**Lagrangian Active Inference (LAIF)** in JAX, following:

- *Realising Synthetic Active Inference Agents, Part I: Epistemic Objectives and
  Graphical Specification Language* (Koudahl, van de Laar, de Vries — arXiv:2306.08014)
- *Part II: Variational Message Updates* (van de Laar, Koudahl, de Vries)

Both papers live in `notes/` (PDFs plus extracted text). The project brief is in the
Obsidian vault at `agents/CFFG/0.0.md`. Reference implementation for parity testing:
https://github.com/biaslab/LAIF (Julia/RxInfer).

## Rules

1. **Package management: `uv` only.** Use `uv init`, `uv add`, `uv run`, `uv sync`.
   Never call `pip` directly, never hand-edit lockfiles. The venv is managed by uv.

2. **Atomic git commits.** One logical change per commit — a commit should build and
   pass tests on its own. Commit early and often; keep refactors separate from
   behavior changes. Reference ticket IDs in commit messages (e.g. `tk-42: ...`).

3. **Tickets via `tk`.** All work is tracked in the `tk` ticket system (`tk --help`
   for usage). Break work into small tickets with dependency links, use epics for the
   major milestones, and close tickets in the same session as the commit that
   resolves them. Before starting work, check `tk` for ready tickets.

4. **Act autonomously.** The agent has broad access to this machine and should use
   it: run tests, install deps, create tickets, commit, and keep moving without
   asking for permission on routine steps. Ask only for genuinely irreversible or
   scope-changing decisions.

5. **Side research goes to sub-agents.** When background reading is needed
   (downloading another paper, checking how PGMax allocates tensors, clarifying an
   active-inference derivation), spawn a sub-agent with the *specific questions* to
   answer and have it report back a short summary. Do not read whole papers into the
   main context window mid-task.

## Session protocol (context hygiene)

This project will span many context windows. The rule that makes that survivable:
**tickets are the shared memory; the repo carries no session state.**

- **Start of session**: read this file, run `tk ready`, pick ONE ticket (or a few
  trivially small ones). Don't re-derive project state from scratch — the ticket's
  description, design notes, and comments are the handoff.
- **During**: atomic commits referencing the ticket ID. Discovered work becomes a
  *new ticket* (linked with a dependency), not scope drift in the current one.
- **End of session**: run `uv run python tools/dashboard.py` to refresh the
  mobile test dashboard (~/agent-share/cffg-tests.html; add new test files to
  EPIC_FILES). Leave `main` green and the tree clean. Update the ticket with
  a handoff comment — what's done, what's next, any gotcha — then close it or leave
  it `in_progress` with that comment as the resume point. A session that ends with
  uncommitted work or an un-updated ticket has failed its landing.
- **One ticket ≈ one context window.** If a ticket can't fit, split it before
  starting, not after running out of context.

### Sub-agent research protocol

- Spawn research sub-agents with **specific questions**, not topics ("What update
  rule does PGMax use for factor-to-variable messages and how are its arenas
  indexed?" — not "read about PGMax").
- The sub-agent's report gets **distilled into the ticket that needed it** as a
  comment (question → answer → source → confidence). Durable engine-relevant
  conclusions go into docstrings next to the code they justify.
- Never paste a paper or long raw report into the main context; never let a
  sub-agent's full output become a new markdown file.

### Document policy

Exactly four canonical documents: `AGENTS.md` (rules), `docs/ARCHITECTURE.md`
(design), `docs/LIMITATIONS.md` (scope honesty), `README.md` (once there's
something to run). **Update in place; do not create new markdown files.** Everything
else lives in tickets (state, decisions-in-progress, research findings), code
docstrings (durable knowledge), or tests (executable claims). If a decision changes
the architecture, edit ARCHITECTURE.md in the same commit as the change.

## Technical invariants

- **True Bethe/Generalised free energy.** No shortcuts in the objective: free-energy
  evaluation must follow the papers' definitions (node-local BFE terms, edge-entropy
  overcounting corrections, P-substitution for GFE nodes). FE values are the
  convergence signal and the policy-scoring quantity — treat them as first-class
  outputs, and test them against analytically known cases.
- **Messages in log-space** wherever feasible; normalize aggressively; explicit JAX
  PRNG keys for anything stochastic (e.g. importance-sampled messages) so runs are
  reproducible.
- **Serialization is the source of truth.** Model structure lives in the template
  format (no large blobs inline — blobs are file paths); the compiled JAX arena is a
  derived artifact and must always be reconstructible from the template + data.
