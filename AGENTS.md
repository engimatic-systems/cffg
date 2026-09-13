# AGENTS.md — Ground Rules for the CFFG Project

This project is a collaboration with a user named Cooper.

This repository implements **Constrained Forney-style Factor Graphs (CFFGs)** and
**Lagrangian Active Inference (LAIF)** in JAX, following:

- *Realising Synthetic Active Inference Agents, Part I: Epistemic Objectives and
  Graphical Specification Language* (Koudahl, van de Laar, de Vries — arXiv:2306.08014)
- *Part II: Variational Message Updates* (van de Laar, Koudahl, de Vries)

Both papers live in `notes/` (PDFs plus extracted text). The project brief is in the
Obsidian vault at `agents/CFFG/0.0.md`.

## Session bootstrap

Before creating a work branch, update its intended base so upstream changes and
conflicts surface before implementation instead of as a late rebase surprise:

```sh
BASE_BRANCH=main  # replace when the ticket names another base
git switch "$BASE_BRANCH"
git pull --ff-only origin "$BASE_BRANCH"
```

## Rules

1. **Package management: `uv` only.** Use `uv add`, `uv run`, `uv sync`.
   Never call `pip` directly, never hand-edit lockfiles. The venv is managed by uv.

2. **Atomic git commits.** One logical change per commit — a commit should build and
   pass tests on its own. Commit early and often; keep refactors separate from
   behavior changes. Reference ticket IDs in commit messages (e.g. `cff-woop: ...`).

3. **Tickets via `tk`.** All work is tracked in the `tk` ticket system (`tk --help`
   for usage). Break work into small tickets with dependency links, use epics for the
   major milestones, and close tickets in the same session as the commit that
   resolves them. Write tickets using the `intelligible-ticket` skill. Before
   starting work, check `tk` for ready tickets.

4. **Act autonomously.** The agent has broad access to this machine and should use
   it: run tests, install deps, create tickets, commit, and keep moving without
   asking for permission on routine steps. Ask only for genuinely irreversible or
   scope-changing decisions.

5. **GitHub access: use authenticated `gh` for this private repository.** The
   official GitHub skill's app connector currently returns `404 Not Found` for
   `engimatic-systems/cffg` on both reads and writes, including PR lookup and
   creation. When a skill requires connector-first access, make that call once;
   after this known repository-specific 404, do not retry it or infer that the
   repository is absent. Fall back immediately to `gh`. `gh auth status` reports
   not-logged-in on this machine, but a PAT (user coopwilliams) lives in
   `~/.git-credentials`. Prefix `gh` commands with, in the same shell call
   (exports don't persist between calls):

   ```sh
   export GH_TOKEN=$(printf "protocol=https\nhost=github.com\n" | git credential fill | sed -n 's/^password=//p')
   ```

6. **Side research goes to sub-agents.** Delegate background reading under
   **Sub-agent research protocol** below. Do not read whole papers into the
   main context window mid-task.

7. **De-weight development costs.** When making technical decisions, do not give much weight to development cost and give more weight to quality, simplicity, correctness, scalability, and long-term maintainability.

## Coding style
- The code should be idiomatic
- Code should be self-explanatory. prefer code readability over comments
- Comments should be short, to the point
- Comments shouldn't include ticket numbers
- Comments shouldn't include made up words and phrases not in the english dictionary, math, or science
- New files shouldn't extend over 1000 lines of code (excluding comments and whitespaces)
- Think twice before adding code to files where the number of lines in a file already surpass 700
- Unit tests are nice, but don't write tests just for the sake of writing tests
- If you need to stuff multiple features or related changes into a PR, organize them into separate commits


## Session protocol (context hygiene)

This project will span many context windows. The rule that makes that survivable:
**tickets are the shared memory; the repo carries no session state.**

- **Start of session**: don't re-read `AGENTS.md` text if already supplied in
  context. Run `tk ready`, pick ONE ticket (or a few
  trivially small ones), then follow **Session bootstrap** before branching. Don't
  re-derive project state from scratch — the ticket's description, design notes,
  and comments are the handoff.
- **During**: atomic commits referencing the ticket ID. Discovered work becomes a
  *new ticket* (linked with a dependency), not scope drift in the current one.
- **Before implementation**: the ticket handoff names the applicable row in the
  `docs/ARCHITECTURE.md` change-path index and lists each duplicated execution
  surface affected; checked pairs that do not apply are marked N/A.
- **During implementation**, run focused test files as often as useful for the
  changed component. Plain `uv run pytest` runs the developer selection,
  `not slow and not pr_head`. Stable expensive regressions carry
  `@pytest.mark.pr_head` and run at every PR head. When changing their behavior,
  include them in focused iteration with
  `uv run pytest tests/test_component.py -m 'not slow'`. A developer or focused
  result does not satisfy the PR gate.
- **Test coverage and cost**: each test and parameter axis earns its place by
  exercising a distinct invariant, failure mechanism, or execution surface.
  Keep cheap useful combinations; replace redundant expensive Cartesian
  products with explicit representative cases. Host/JIT, flooding/scheduled,
  single-instance/`vmap`, and checkpoint-version checks remain distinct where
  they exercise independent implementations. Keep representative numerical
  contract cases in the developer suite: sparse support, improper messages,
  float32, and uncomfortable scale. Defer expensive end-to-end repetitions to
  `pr_head` without removing their direct numerical coverage.
  Reuse an expensive deterministic setup when several assertions need the
  same solved model, with immutable results or fresh copies so tests cannot
  affect each other. Added seeds, sizes, horizons, and parameter combinations
  need an additional claim; more test instances alone are not more coverage.
  Measure expensive additions and the largest contributors using durations
  from the required runs, record the coverage gained and cost in the ticket,
  and review growth before adding more. Do not add a separate suite run just
  to measure cost or impose arbitrary universal time limits.
- **Final validation for code outside the isolated-tooling exception below**
  requires one passing PR-gate run after the final code change at each
  independent proposed PR head. One passing gate at that head is sufficient
  to open or update its PR; another independent PR head requires its own gate.
  Before launching or waiting on it, inspect
  only `latest.wall_seconds` and `suggested_wait_seconds` in
  `tools/test-suite-runtimes.json` for measured wall time and a wait budget
  with headroom. If the focused tests passed before, then don't poll the full
  test suite. Instead just wait for the `suggested_wait_seconds` time.
  **Test-suite waits are an explicit exception to general guidance to send
  frequent progress updates, including once-per-minute check-ins.** Announce
  the suite launch and wait budget once, then wait quietly for the full
  `suggested_wait_seconds`: no periodic commentary and no polling the process,
  logs, or dashboard. Run
  `uv run python tools/dashboard.py --run-suite` to execute the PR-gate
  `not slow` pytest selection, including `pr_head`, exactly once
  and refresh both that runtime history and `~/agent-share/cffg-tests.html`
  (add new test files to `EPIC_FILES`). If the PR gate was run directly,
  preserve and reuse its
  result: `uv run pytest -q -m 'not slow' --junitxml=/tmp/cffg-tests.xml`, then
  `uv run python tools/dashboard.py --junitxml /tmp/cffg-tests.xml`. Opening a
  PR, requesting review, or refreshing the dashboard never calls for a second
  unchanged PR-gate run. The dashboard accepts only reports carrying complete
  PR-selection provenance; developer, focused, and legacy reports without
  that provenance cannot certify it. Runtime history records only this gate,
  so developer runs do not change its wait budget. A later change requiring
  this gate supersedes the previous result and gets one fresh final run.
- **Isolated tooling changes** use focused tests and changed-artifact checks
  instead of the PR gate. Examples include runtime-history storage,
  developer reports, and formatting helpers. Record the isolation rationale,
  commands, and results in the ticket and PR. Being under `tools/` is not enough:
  changes affecting engine/model/numerical/serialization behavior, shared
  dependencies, test discovery/selection/execution, or pass/fail interpretation
  still require the full gate. If isolation is uncertain, use the full gate.
  Validate after the final relevant edit; never publish focused results as if
  they were a fresh full-suite result.
- **Slow tests are opt-in.** Mark long behavioral, multi-trial, and globalization
  regressions with `@pytest.mark.slow`; pytest excludes them by default. Run
  them explicitly with `uv run pytest -m slow` only when the changed behavior
  needs that validation. They are not part of the PR gate. Use `pr_head` for
  stable expensive regressions required at every PR head; do not combine the
  two markers on one case.
- **Ticket- and documentation-only changes** do not require the PR gate.
  Validate the artifacts they change, including `tk dep cycle` and
  `git diff --check` for ticket-only work.
- **End of session**: leave `main` green and the tree clean. Update the ticket
  with a handoff comment — what's done, what's next, any gotcha — then close it
  or leave it `in_progress` with that comment as the resume point. A session
  that ends with uncommitted work or an un-updated ticket has failed its
  landing.
- **One ticket ≈ one context window.** If a ticket can't fit, split it before
  starting, not after running out of context.

### Bounded reads

- Reuse unchanged repository text and ticket content already available in
  context. Refresh when relevant content changes through an edit, branch switch,
  or pull, or when the needed text is missing after context compaction. If
  freshness is uncertain, check a diff or file hash before reloading it.
- Find filenames, headings, or symbols first; then read the needed section or
  function. Use the architecture change-path index before opening large modules.
  For PRs, inspect the changed-path list before requesting patches.
  Re-read to resolve missing evidence, not merely because a new turn starts.
- Return relevant matches, counts, and failures from tools. Keep long diagnostic
  detail in artifacts and return their paths instead of repeatedly printing it.
- Track prior reads in working context only; do not create repository read logs.

### Ticket and roadmap authoring

- A roadmap-refinement change includes a compact ASCII dependency tree in its
  ticket note or PR description. State the arrow direction and show the ready
  roots, parallel work, immediate blockers, and downstream roadmap
  feature/epic. For example:

  ```text
  Legend: prerequisite --> dependent

  [x] foundation
        |
        +--> [ ] interface (ready)
                  |
                  +--> [ ] feature
                            |
                            +--> [ ] roadmap epic
  ```

  Build the explanation from `tk dep tree --full`, `tk ready`, and `tk blocked`,
  then run `tk dep cycle`. The `tk` dependency records remain the source of
  truth; the ASCII tree is their human-readable companion.

### Sub-agent research protocol

- Spawn research sub-agents with **specific questions**, not topics ("What update
  rule does PGMax use for factor-to-variable messages and how are its arenas
  indexed?" — not "read about PGMax").
- Supply only the needed facts and source paths. Include full conversation
  history only when the research question needs it.
- The sub-agent's report gets **distilled into the ticket that needed it** as a
  comment (question → answer → source → confidence). Durable engine-relevant
  conclusions go into docstrings next to the code they justify.
- Never paste a paper or long raw report into the main context; never let a
  sub-agent's full output become a new markdown file.

### Document policy

Six canonical documents: `AGENTS.md` (rules), `docs/ARCHITECTURE.md`
(design), `docs/LIMITATIONS.md` (scope honesty), `docs/DERIVATIONS.md`
(formal message/FE derivations beyond the papers, each with a plain-English
companion and executable-test anchors — added 2026-07-09 at Cooper's request
for cff-fk2u), `docs/READING.md` (curated external reading list — added
2026-07-12 at Cooper's request; keep it cream-of-the-crop short, verified
links only), `README.md` (the user guide).
**Update in place; do not create new markdown files** — with one sanctioned
exception: `docs/todo-notes/` holds one note per *remaining* under-explained
work thread (per Cooper, 2026-07-05). Each note names its `tk` ticket; when
the work lands, **delete the note in the same commit** — todo-notes describe
the future only, never the present (the present belongs to README/
ARCHITECTURE/docstrings). Everything else lives in tickets (state,
decisions-in-progress, research findings), code docstrings (durable
knowledge), or tests (executable claims). If a decision changes the
architecture, edit ARCHITECTURE.md in the same commit as the change.

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

## Numerical robustness contract

Adopted 2026-07-09 after a divergent solver was reported as converged and four
stacked defects went undetected. Their shared root: rules were only ever
exercised inside the test fixtures' comfort zone. Every message rule, solver,
and energy term — including all future factor families on the roadmap — ships
against this contract.

- **Hostile-inputs checklist at planning time.** A rule ticket states, before
  implementation, what happens at: sparse/zero support (exact log-zeros),
  improper or vacuous incoming messages, float32 (the test suite forces x64;
  consumers get JAX's default), and 10–100× the fixture's domain size.
  "Provably N/A" is an acceptable answer; not asking is not. Answers become
  acceptance criteria.
- **Guards scale with the dtype.** No absolute numeric literals as floors,
  ridges, or clamps: 1e-300 is exactly zero in f32, and a 1e-9 ridge sits
  below f32's machine epsilon (numerically singular). Derive guards from
  `jnp.finfo(dtype)`, and comment which invariant each one protects — a guard
  whose purpose isn't stated can't be reviewed.
- **Convergence claims are falsifiable.** Any iterative loop distinguishes
  converged / budget-exhausted / diverged, and a non-finite delta or FE is
  divergence — loud, never a returned state that looks like success. The
  existing "FE is the convergence signal" invariant is enforced in code, not
  prose.
- **Inner solvers get globalization by default.** A fixed iteration count
  tuned on a small fixture is a scale trap. Newton-type solves take
  backtracking/damping unless the ticket argues the domain is provably safe;
  gauge freedoms (shift-invariant residuals) are projected out explicitly,
  not left to a ridge.
- **Test the contract, not just goldens.** Each rule family carries at least
  one sparse-support case, one float32 case, and one case at uncomfortable
  scale. Golden parity on small models proves correctness; these prove
  survival. When a defect is found, the regression test reproduces the
  *mechanism* (the hostile input), not the incident.
