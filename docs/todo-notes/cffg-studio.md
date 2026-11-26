# CFFG Studio

**Stack proposal for Cooper · 26 November 2026 · Delivery epic `cff-xikb`**

Build a keyboard-first graph editor that opens from agent-share. Give the browser and coding agents the same revisioned model API. Keep authoritative graphs in a separate private Git library on the Mac. Run the existing Python compiler and JAX engine next to that library. The remote computer handles drawing, selection and input locally.

The first release should prove one useful round trip. Open an rSLDS, rename a latent variable, ask an agent to critique the emission noise, inspect its proposed change, then run both revisions against the same observations. The two results should tell you exactly which model, simulator, seed and execution settings produced the difference.

## Stack

| Part | Choice | Responsibility |
| --- | --- | --- |
| Editor | Browser-native JavaScript ES modules, semantic HTML, CSS Grid and interactive SVG | Draw the library’s Forney notation, manage focus and commands, update the visible canvas locally. Ship assets inside the Python wheel through the existing uv build backend. |
| Model service | Python, FastAPI, Pydantic and Uvicorn | Versioned HTTP/JSON commands, OpenAPI discovery, session authentication and resumable server-sent events. Browser, CLI and Python clients use one contract. |
| Model library | A separate private Git repository called `cffg-models`, with Git LFS for large array files | Retain human-readable graph structure, content-addressed parameters and attributable history. Keep portable exchange through the current CFFG bundle format. |
| Search | SQLite FTS5, a background filesystem walk and Python watchfiles | Index the library and other directories on the Mac. Search model names, paths, factor families, tags and notes. Rebuild the index from its sources. |
| Experiments | Separate Python worker processes managed with uv, using CFFG, JAX and NumPy | Isolate compile jobs, run scheduled inference, measure resources and stream progress. Pin experiment inputs by revision and digest. |
| Simulator authoring | A server-side OpenAI Responses API adapter with structured output, plus a pinned Python/NumPy runtime in an isolated Linux container | Produce inspectable simulator source and a manifest. Check generated observations before binding them to a run. |
| Delivery and tests | macOS launchd, Tailscale Serve, pytest and Python Playwright | Restart the service at login, expose it privately and verify two-client editing and keyboard-only workflows in a real browser. |

The graph already has a layout engine. Native SVG lets Studio keep its variable wires, factor boxes, constraint beads, plate boundaries and named terminals. The browser owns hit testing, editing gestures and keyboard navigation. Browser modules ship with the wheel and need no JavaScript package manager or external asset server. [CFFG scene pipeline](../../src/cffg/viz/__init__.py) · [uv package data](https://docs.astral.sh/uv/concepts/build-backend/#file-selection) · [FastAPI and OpenAPI](https://fastapi.tiangolo.com/features/) · [Playwright for Python](https://playwright.dev/python/docs/intro)

## At the workstation

Use a compact charcoal workspace with resizable, dockable panels, a single accent for selection, a visible focus ring and explicit text for warnings. Put the model library and searchable graph outline on the left, document tabs and the canvas in the center, and the selected element’s properties on the right. Give the bottom panel tabs for Problems, History, Critiques and Runs. Save panel layouts per user. Opening a run alongside its graph should keep both selections when switching tabs.

**New** opens a blank graph or a catalog recipe. Ship a small linear Gaussian model and a recurrent switching linear dynamical system (rSLDS) alongside the blank starter. Show the recipe’s horizon, latent dimension, observation dimension, number of regimes and seed before creation, where applicable. **Open** searches the library and host files. **Save** accepts a compiled change under the active model. **Save As** forks a new model with its source revision recorded. Export writes a portable `.cffgz`. Studio annotations and layout travel with the library model or an explicit companion file when exporting.

Every menu item, inspector action and canvas operation gets a stable command ID. One searchable palette displays its current binding, arguments and availability. Proposed defaults include `⌘/Ctrl K` for commands, `F2` for rename, `F6` to cycle panels, and `Esc` to return to the graph. Arrow keys traverse an ordered outline or follow a wire between factors. **Connect ports** lets the user search for a source, see compatible target roles and complete the connection without dragging. Multi-object operations collect arguments in a private draft. Users can rebind keys and chords per context, detect collisions and browser-reserved keys, reset defaults, and import or export a profile. Text fields keep their editing keys. [Keyboard interaction patterns](https://www.w3.org/WAI/ARIA/apg/patterns/treeview/)

For example, select the latent wire `x1`, press `F2`, type `position_1` and inspect the affected factor roles. Change its dimension in the inspector. Studio previews every dependent array, observation binding and learned parameter, including ties across time. Choose how to map existing values or initialize replacement dimensions. If a selected model family cannot admit that shape, the inspector identifies the blocking factor and keeps the last accepted revision on screen.

For a factor’s parameters, show only compatible presets. A categorical transition can offer uniform, sticky or seeded random columns. A Gaussian linear transition can offer an identity map with a chosen process-noise covariance, or a seeded matrix and proper noise covariance. These are recipes across the factor’s parameter roles. Show the effective values and whether they come from a supplied array, a stored file or an initializer. Applying a new preset must replace an overriding value in the same transaction. Changing the size of a tied rSLDS parameter previews every affected slice and any prior or posterior that would need replacement. [Initializer definitions](../../src/cffg/inits.py) · [rSLDS builder](../../src/cffg/models/rslds.py)

## What a successful save means

1. The client sends a base revision, an idempotency key and an ordered group of typed commands. The document can have incomplete form fields or a half-drawn wire while a user is editing. Other clients continue to see the last accepted graph.
2. A worker applies the group to a private candidate, resolves arrays, learning state and plugin capabilities, and validates it against the declared execution profile. For a model change, compile the exact staged template and data, reload the portable bundle, and verify that the reloaded contents compile under the same profile. Attach diagnostics to the command, element, role and expected shape. A layout or note edit can reuse the admission result for identical model contents and the same engine profile.
3. Persist the new Git objects and referenced large files first. Advance the library head with a conditional Git ref update. A stale edit receives the current revision, affected elements and a diff. A retry of an accepted request returns its original receipt. A concurrent edit to a different model can be carried forward after checking that this model’s base is still current.
4. Emit a revision event after publication. A restarted service reconstructs accepted revisions from Git and replays missed events from an acknowledged cursor. Undo creates a checked inverse edit against the current revision. Where later edits conflict, show a preview or offer to fork the historical revision.

The current library already copies and structurally validates batches of **14 edit operations**. Compilation also checks numerical shapes, resolves data and admits supported compositions. New core actions are needed for general rename, resize and existing initializer replacement. The existing structure hash deliberately ignores initializer values and blob paths. Give every model a stable ID, every revision a full content identity, and every annotatable element a stable ID scoped to its plate or fragment. Store anchor mappings, positions, notes and critiques alongside the canonical model. Preserve old anchors in history on delete. [Edit actions and codec](../../src/cffg/core/edits.py) · [Compiler](../../src/cffg/compiler.py) · [Bundle IO](../../src/cffg/bundle.py) · [Conditional Git updates](https://git-scm.com/docs/git-update-ref)

At proposal time, the branch compiled an empty template and round-tripped a current-format, three-step rSLDS with 14 factors, 19 variables and four learning owners. This is a usable starter path. The rSLDS builder explicitly creates each time slice. A horizon change therefore has to update factors, DATA ports and learning sources together. A custom edited graph runs through the generic engine. Offer specialized rSLDS fitting only when the declared graph matches the fitter’s supported topology.

## One catalog for humans and agents

Use a host checkout such as `~/projects/cffg-models`, separate from the code repository and the served directory. Each `models/<model-id>/` contains canonical `model.json`, referenced arrays, the portable manifest and learning members, plus Studio metadata. Use the existing bundle reader and writer to assemble and check exchange archives. A fresh checkout with its large-file objects must restore the graph and declared learning state. Store model experiments by model, revision and run ID. Retain data tapes and measurements by digest so two revisions can reuse exactly the same input. Search and event indexes can be rebuilt. The host library is the initial central repository. A private Git remote can mirror accepted history independently of interactive saves.

| Client operation | Proposed HTTP interface |
| --- | --- |
| Find and inspect a model | `GET /v1/models?q=…`, `GET /v1/models/{id}/revisions/{rev}`, `GET /v1/capabilities` |
| Preview or apply changes | `POST /v1/models/{id}/transactions/preview`, `POST /v1/models/{id}/transactions` |
| Discuss or review | `POST /v1/models/{id}/critiques`, `POST /v1/models/{id}/proposals`, then accept, decline, reply or resolve by ID |
| Run and watch | `POST /v1/runs`, `GET /v1/runs/{id}/events`, `GET /v1/models/{id}/events` with reconnect cursors |
| Request a simulator | `POST /v1/models/{id}/simulators`, including the model revision, observation contract and prompt |

An agent can post “The emission covariance on this observation is too small for the simulator noise,” attach the port, revision and run evidence, and supply a proposed parameter edit. The human can reply, preview the numerical diff, accept the change, or retain the critique without changing the graph. Agents with edit permission can also commit directly using the same validator. Both clients see author, base revision and accepted change. Deliver an `uv run cffg-studio` CLI with JSON input/output so shell-based agents can use those operations without driving the GUI.

**Find on this Mac** searches explicitly selected roots, initially the model library and Cooper’s projects. Add other readable directories from settings. Identify `.cffgz` archives and loose CFFG template JSON by their structure, including recognized content with a `.cffg` name. Display path, format, last change and import diagnostics. Bound archive inspection and resolve relative blob paths inside the authorized roots. Skip caches and repeated symlink targets. A watcher updates the index and a background rescan repairs missed changes. Opening a discovered file imports a versioned copy. Explicit save-back checks for external edits and replaces a supported source atomically. A loose multifile model can be exported into a fresh directory to keep its files consistent. [SQLite full-text search](https://sqlite.org/fts5.html)

## Experiments while you edit

Keep the inspector tied to the accepted revision. A structural change immediately invalidates older measurements in the current view. Label previous results with their revision and offer them for comparison.

| Available when | Display | Interpretation |
| --- | --- | --- |
| Reading the template | Factors, variable domains, port counts, declared shapes and dtypes, logical plate sizes, parameter bytes or bounds where dimensions are known | Static analysis. Mark unknown dimensions and estimate assumptions explicitly. |
| Compiling the model | Actual persistent state and parameter bytes by leaf, separately reported temporary executable buffers, FLOPs and memory traffic where the backend provides them | Compiler and backend figures. State size alone cannot predict peak process RAM. |
| Measuring in a fresh worker | Process startup and imports, template-to-arena time, JAX lowering and compilation, first completed execution, peak process RSS and device metrics where available | Cold measurements. Record hardware, dtype, schedule, engine revision and cache state, including persistent compilation caches. |
| Repeating on the same worker | Synchronized warm latency distribution, sample count, throughput and workload size | Measure the same schedule and inputs. Report instrumented step playback separately from the throughput benchmark. |

A **Random data** run builds seeded observations matching DATA port domains and shapes. A saved observation tape reproduces inputs exactly. The **Step** control names its unit. Depending on the run, it advances an inference iteration, a schedule stage with a supported hook, or an environment tick with plan and filter phases. Pause or cancel at a defined boundary. Stream free energy when defined, residuals, actual iterations, elapsed time and solver diagnostics. End with converged, budget exhausted or diverged as reported by the engine. Preserve the final state and objective of the selected schedule when tracing. The existing `Engine.step` performs a flooding sweep. Stepping a scheduled run needs its own progress path. Bound event frequency, keep terminal statuses lossless, and let clients reconnect and fetch the full recorded trace.

The existing estimator needs `cff-j7mm` for scheduled executable analysis and reliable calibration. Its current persistent state-byte counts apply after arena compilation. Compact-plate resource counts before compilation also touch `cff-6s31`. The UI should say **unavailable** with the backend’s reason where a figure cannot be obtained. Cold and warm times become available after a benchmark for that model and execution profile. [Estimator](../../src/cffg/estimate.py) · [Existing cold/warm benchmark](../../tools/benchmark_ordinary_iteration.py) · [JAX timing requirements](https://docs.jax.dev/en/latest/benchmarking.html)

## The simulator prompt

The Simulator panel accepts a description such as “Generate a switching trajectory with two regimes and intermittent noisy position observations.” Give the configured language model the contract for the selected revision’s DATA ports, units if supplied, and a seedable Python/NumPy interface. Send only the model context selected for that request. The response contains a manifest, source and parameter defaults. Store the prompt, provider/model identifier, dependency image and source digest, and show a code preview and diff for each revision. [Structured model outputs](https://platform.openai.com/docs/guides/structured-outputs)

The generated entry point is `simulate(seed, steps, parameters)`. Its result separates named observations from optional latent ground truth. An adapter maps series to explicit unrolled DATA ports or successive agent ticks as the selected model declares. The writer emits a JSON manifest and non-pickled numeric arrays. The isolated worker gets a read-only runtime, bounded CPU, RAM and time, an empty writable output directory and no network or host credentials. The host validates names, masks, lengths, shapes, dtypes, finite values and domain support, then verifies binding through the actual engine. Record the accepted tape’s digest. A stochastic simulator and seeded random test data remain separately selectable sources.

## Remote access and release criteria

Place the launcher at `~/agent-share/cffg-studio/index.html`. The existing file server handles GET and HEAD. The launcher opens a private HTTPS Tailscale Serve route to a loopback Uvicorn service managed by launchd. Serve both Studio assets and its API from that origin. Authenticate the user from the trusted private ingress and issue scoped, revocable agent tokens. Keep credentials, model blobs and run state outside the shared directory. The remote browser can reconnect after sleep, recover its selected revision and catch up using its event cursor. [Tailscale Serve](https://tailscale.com/kb/1312/serve)

Measure the following proposed acceptance targets on a recorded host, browser and remote connection.

- With 1,000 visible elements from a 10,000-element graph, local pan and zoom should meet a 16.7 ms frame budget at the 95th percentile. Keyboard selection and palette results should appear within 50 ms at the 95th percentile. Cull offscreen geometry, collapse repeated slices and keep the logical outline searchable.
- Warm search over 50,000 indexed file entries should return its first page within 150 ms on the host. Measure network round-trip and browser rendering separately.
- A running compile, model search or experiment must leave local selection and typing responsive. Two connected clients should see a committed change or a progress event within 250 ms plus network latency under the reference load. Measure publication separately from time to compile.
- A keyboard-only browser test and an independent CLI client must complete the opening rSLDS walkthrough. Include a stale agent proposal, an invalid resize, an interrupted save, a dropped event connection and replay of the identical observation tape.

## Delivery order

The first independently usable slice is a remote blank-or-rSLDS editor with a real save, reopen and agent read. Deliver the library API and edit vocabulary first, then build the keyboard editor, remote entry and two-client review. Runs, profiling and simulator authoring consume the same immutable revisions. Each stage should demonstrate its normal browser or CLI entry point.

| Ticket | Reviewable outcome |
| --- | --- |
| `cff-x6c0` | Central library, compile-checked revisions, portable round trips and baseline API |
| `cff-7piw` | Reference-safe rename, dimensional changes and parameter initialization |
| `cff-cz82` | Forney canvas, graph outline, inspector, command palette and editable keymap |
| `cff-7noc` | Agent CLI, anchored critique threads and reviewable proposals |
| `cff-w551` | Searchable host files and authenticated remote entry from agent-share |
| `cff-djod` | Seeded input, run workers and resumable schedule-aware traces |
| `cff-5mf2` | Structural, compiled and measured resource reports, using `cff-j7mm` |
| `cff-fw2v` | Prompt-to-simulator generation, isolated execution and replayable observation tapes |

Retire this note when `cff-xikb` lands. Put the shipped commands and workflows in the README and the durable service and persistence contracts in ARCHITECTURE.
