# CFFG

CFFG is a JAX library for building and running constrained Forney-style factor
graphs and Lagrangian active-inference models. It is useful when an experiment
has an explicit state ontology and auditable local factors: tabular agents,
linear or nonlinear state-space models, conjugate exponential-family models,
hard discrete constraints, and declared hybrid approximations.

It is not a general probabilistic-programming system, a structure learner, or a
raw-sensor representation learner. Exactness depends on the graph, factor
family, constraint, and objective; convergence never turns an approximate
model into an exact one. Start with the [capability map](#capability-map), and
use [Known Limitations](docs/LIMITATIONS.md) to qualify experiment claims.

CFFG is currently version 0.1 and requires Python 3.12 or newer.

## CFFG Studio

Open the **CFFG Studio** launcher in agent-share from a computer signed into the
owner’s Tailscale account. Start with **Switching dynamics** to inspect an rSLDS,
**Linear Gaussian tracker** for a Kalman model, or **New model → Blank graph**.
Factors are boxes and variables are wires. Mint terminals mark observations,
constraint beads carry their mathematical symbols, and selection has a separate
outline. The inspector lists the roles and dimensions behind each mark.
Hover a wire to highlight all occurrences of that quantity, including equality
branches. Unrolled recurrences with identifiable time steps read left to right.
Choose **Open selected plate** to enter a repeated fragment. Edits there apply
to every copy, and annotations follow their variables through renames.

Press **⌘K** on macOS or **Ctrl+K** elsewhere to find any command. **F6** moves
between panels. From the outline, press **F2** to rename a variable or **Shift+R**
to resize it. The preview lists affected arrays, shapes, learning owners and
the factors sharing them. **Commit change** publishes a revision. Shift-drag a
factor or use **Position selected factor** to change its grid offset. **Keyboard
shortcuts** edits bindings, checks conflicts, supports chords, and imports or
exports a profile. Panel sizes persist in the browser.

Hold **Option** (**Alt** on other keyboards) and hover over a control or value
outside the canvas to read its explanation. Try the number beside **Persistent
state** in **Profile**. You can also Tab to a control and hold Option.
Click **Hover help** in the bottom bar to keep explanations on as you move
or Tab between controls. Click it again or press **Esc** to turn it off.

**Open** searches saved models and indexed directories on the host. The same
dialog accepts file uploads and changes search roots. **Hide incompatible files**
remembers your preference; turn it off to inspect schema diagnostics. **Export** writes a portable
`.cffgz` with graph data and learning state. **Export graph with Studio annotations**
writes a `.cffgstudio` project including layout, notes and review threads.
**Save as a new model** forks a named revision. Use **History** to inspect changes
or restore an earlier graph as a new revision.

**Run** binds seeded observations or a saved tape. **Step through inference**
plays progressively larger solver budgets from the same initial state and can
pause after each sample. **Resource profile** distinguishes initialized-state
storage from measured peak RSS, first execution and synchronized warm latency.
Experiment history retains solver statuses and inputs by revision. **Simulator**
can generate editable Python/NumPy source from a prompt or run supplied source.
The isolated process returns one observation per named DATA port and optionally
its hidden ground truth. Reuse the resulting tape to compare model revisions.

### Guided tours

To have a coding agent guide you through a model, open the model, choose
**Tours → Copy agent prompt**, and paste it into the agent. Add what you want
explained, such as “trace one observation back to the hidden state.” The prompt
includes the model ID, its link, and instructions for publishing. A remote
agent needs the Studio endpoint and its own edit token (see **Coding agents**
below). Open the playback link the agent returns, or select its tour under
**Tours**.

Use **←** and **→** to page through stops and **Escape** to leave. With
**Animations** on, the camera pulls back to show both stops, pans across the
graph, then zooms in on the next highlighted group. Crossing into or out of a
plate takes you through its box on the parent graph. Turn **Animations** off to
jump directly to each stop. Studio remembers your choice in this browser.
Animations start on unless your system requests reduced motion.
**Copy link** shares the tour. For a walkthrough
of the editor itself, choose **Tours → Getting started with CFFG Studio**.

For example, an agent can save this command array as `tour.json` for the
canonical rSLDS.

```json
[{
  "action": "set_tour",
  "id": "switching-dynamics",
  "title": "Follow one switching transition",
  "steps": [
    {"title": "The switching model", "text": "Follow how a discrete regime changes the continuous state and its next observation."},
    {"title": "Initial regime", "ref": "factor:z0_prior", "text": "This categorical prior supplies the starting regime."},
    {"title": "One transition", "refs": ["factor:dynamics1", "var:x0_dynamics", "factor:switch1"], "text": "Trace the previous state and the new regime into the first continuous transition."},
    {"title": "Next observation", "ref": "factor:emission1", "text": "The emission factor maps x1 to observation y1."}
  ]
}]
```

Run `uv run --extra studio cffg-studio show MODEL_ID` to get the current
`revision` and valid reference names under `metadata.anchors`.
Run `uv run --extra studio cffg-studio preview MODEL_ID --base REVISION --commands tour.json`,
then substitute `apply` for `preview` with the same base revision to publish. Reuse the tour ID to revise
its steps or send `{"action":"remove_tour","id":"switching-dynamics"}` in a
command array to remove it. For an overview, omit both `ref` and `refs`.
A group accepts up to 64 distinct references. The inspector opens on the first
surviving member. Set `"plate":"name"` to frame that plate's contents and use
reference keys from `GET /v1/models/MODEL_ID?plate=name`. Published stops follow
renamed elements. If a member disappears, playback identifies it and still
frames the surviving marks. Studio
project exports include tours, and `/v1/capabilities` exposes the authoring schema.

### Coding agents

The application exposes OpenAPI at `/openapi.json`. The CLI reads the local owner
credential on the Studio host. Issue a separate token for each remote agent and
supply its private HTTPS endpoint in `CFFG_STUDIO_URL` and its token in
`CFFG_STUDIO_TOKEN`. Treat issued token output as a secret.

```sh
uv run --extra studio cffg-studio models
uv run --extra studio cffg-studio capabilities
uv run --extra studio cffg-studio show MODEL_ID
uv run --extra studio cffg-studio preview MODEL_ID --base REVISION --commands edits.json
uv run --extra studio cffg-studio apply MODEL_ID --base REVISION --commands edits.json
uv run --extra studio cffg-studio critique MODEL_ID --base REVISION --ref var:x1 --text "Check process noise against the observed trajectory."
uv run --extra studio cffg-studio events MODEL_ID
```

`edits.json` contains an ordered array such as
`[{"action":"rename_variable","name":"x1","to":"position_1"}]`.
Reuse a transaction's `--request-id` when retrying. A concurrent edit returns
HTTP 409 with the current revision. `propose` accepts a comment and a command
file for human review. `token issue NAME --scope read`, `token list`, and
`token revoke ID` manage access on the host.

### Host installation

`uv run --extra studio python tools/install_studio.py --login OWNER_LOGIN`
installs a wheel into a private release environment, starts a launchd agent,
routes private Tailscale HTTPS to its loopback listener, and writes
`~/agent-share/cffg-studio/index.html`. The launcher opens HTTPS port **8098**,
within this host’s 8080–8100 Tailscale access rule for the owner's devices.
On a different host, use `--https-port` to select a free port that its
Tailscale policy permits from the intended clients. The central model repository is
`~/projects/cffg-models`. Clone it to inspect ordinary Git history.
Studio uses Git LFS when a model contains an array at least 1 MiB in size.
Back up the repository including its LFS object store and the experiment data in
`~/.local/share/cffg-studio/runs`. Publishing an edit goes through the Studio API.

The simulator uses a configured OpenAI API key or the signed-in host Codex CLI.
Executing generated source requires Docker with the `colima` context and the
Studio NumPy image. The pinned arm64 image can be built from the repository root:

```sh
cd tools/studio-simulator
uv pip install --target packages --python-version 3.12 --python-platform aarch64-manylinux_2_28 --only-binary :all: --no-deps --require-hashes --link-mode copy -r requirements.txt
docker --context colima build --network=none -t cffg-studio-simulator:py312-numpy2.5.1-arm64 .
```

## Install from a sibling experiment

The repository is not published as a package yet. From an experiment project
next to a checkout named `cffg`, add the complete editable dependency with uv:

```sh
mkdir my-cffg-experiment
cd my-cffg-experiment
uv init --python 3.12
uv add --editable ../cffg
uv run python -c 'import cffg; print(cffg.Engine, len(cffg.__all__))'
```

That command records both the `cffg` dependency and its local source in the
experiment's `pyproject.toml`; no hand-written `[tool.uv.sources]` fragment is
needed.

JAX uses its normal platform defaults. Ordinary use does not require x64. When
an experiment needs 64-bit numerical comparisons, configure it before Python
imports JAX:

```sh
JAX_ENABLE_X64=1 uv run python experiment.py
```

Or call `jax.config.update("jax_enable_x64", True)` before constructing arrays,
compiling a template or initializing a state. Enabling x64 later cannot recover
precision already lost in float32 inputs or messages.

## Five-minute workflow

Every workflow has the same spine: build a serializable `Template`, compile its
static structure, initialize and bind a runtime state, run inference, reject an
unfinished status, then use the readout whose semantics match the model.

<!-- executable-quickstart:start -->
```python
from cffg import Engine, build_gridworld, compile_template

template, arrays = build_gridworld(past=1, steps=1, k=2, acc=1.0)
engine = Engine(compile_template(template, arrays=arrays))
state = engine.bind_data(engine.init_state(), {"act1": 1, "obs1": 0})

result = engine.run(state, 20, tol=1e-6)
state = result.require_converged()

posterior = engine.marginals(state)["zm1"]
energy = engine.bfe(state)
print(posterior, energy, engine.objective_contract)
```
<!-- executable-quickstart:end -->

The complete version also saves a bundle, runs a rolling agent, and proves
checkpoint continuation: [examples/discrete_agent.py](examples/discrete_agent.py).

## Capability map

“Ready” means the row is usable inside its stated envelope, not that every
graph assembled from those parts is exact. “Bounded” means the approximation
or fit claim is part of the API contract. “Pre-1.0” marks an intentionally
unstable extension surface. Every row links to its explanation or a runnable
example.

| Task | Import or entry point | Model and mode | Primary result | Objective or approximation | Runnable example | Main limit |
|---|---|---|---|---|---|---|
| [Build the graph IR, compose, plate, and edit](#build-and-compose-models) | `cffg.core`; `cffg.core.compose`; `cffg.core.edits` | Typed variables, factors, ports, constraints, compact plates, transactions | Validated `Template` plus external arrays | Declaration only; compilation selects executable rules | [discrete_agent.py](examples/discrete_agent.py) | **Ready:** only registered factor/constraint combinations execute; expand compact plates before composition |
| [Build a corridor agent](#discrete-planning-and-rolling-agents) | `cffg.build_gridworld`; `CorridorEnv` | Tabular discrete BP/GFE | State/action marginals, BFE/GFE, actions | Exact on eligible ordinary trees; declared GFE for P-substituted planning | [discrete_agent.py](examples/discrete_agent.py) | **Ready:** finite horizon, designed state space, fixed factorization |
| [Build a general discrete rollout](#discrete-planning-and-rolling-agents) | `cffg.models.build_rollout`; `build_tmaze` | Finite-horizon POMDP/LAIF, fixed or learned A/C/B | `marginals`, `node_beliefs`, goal/parameter readouts | Bethe or declared GFE by construction | [discrete_rollout.py](examples/discrete_rollout.py) | **Ready/bounded:** functional mu3 and transition novelty require their diagnostics |
| [Build a factored rollout](#discrete-planning-and-rolling-agents) | `build_factored_rollout`; `StateBlockSpec`; `CouplingSpec` | Mean-field state blocks with declared couplings | Per-block and joint-boundary marginals | Same BP/GFE contracts as the selected rollout | [discrete_rollout.py](examples/discrete_rollout.py) | **Ready:** couplings and joint sensors can still scale as Cartesian products |
| [Compile, run, and swap parameters](#compile-run-and-read-results) | `compile_template`; `Engine`; `set_params`; `RunStatus` | Exact two-pass trees or iterative scheduled graphs | Status, traces, family readouts, energy | Depends on the objective contract below | [discrete_agent.py](examples/discrete_agent.py) | **Ready:** static shapes compile separately; loopy convergence is not an accuracy certificate |
| [Use GFE and parameter epistemics](#discrete-planning-and-rolling-agents) | `ConstraintKind.P_SUBSTITUTION`; `prepare_mu3`; `score_mu3_epochs`; builder flags | Observation epistemics, learned goals, finite-sample mu3(A), transition novelty | Policy posterior, ESS/support facts, specialized diagnostics | GFE; transition novelty is an optimistic bound | [discrete_rollout.py](examples/discrete_rollout.py) | **Bounded:** explicit PRNG keys; open validation defects are listed in `LIMITATIONS.md` |
| [Run a receding-horizon agent](#discrete-planning-and-rolling-agents) | `RollingAgent` | Plan, act, observe, learn, roll a fixed-memory boundary | Action, beliefs, learning state, checkpoint | Replans a finite-horizon factorized policy each tick | [discrete_agent.py](examples/discrete_agent.py) | **Ready:** environment state is external; deep plans need status/trace checks |
| [Run the costly-cue behavioral oracle](#discrete-planning-and-rolling-agents) | `cffg.experiments.behavioral_epistemics.run_benchmark` | Fixed active-inference versus BFE-ablation task | `BehavioralReport` | Bounded scientific regression, not a general inference objective | [API pattern](#discrete-planning-and-rolling-agents) | **Research-only:** fixed task and no supported package facade or reuse guarantee |
| [Learn declared parameters](#learn-and-swap-parameters) | `cffg.learn.Learnables`; `cffg.conj`; `LearnableSpec` | Dirichlet, NIW, MNIW, component-MNIW, Pólya–Gamma | `expected`, `rollover`, persistent posterior state | Between-trial conjugate or empirical-Bayes updates | [learning_bundle.py](examples/learning_bundle.py) | **Ready:** `accumulate` and `replace` have different trial semantics; not full joint VB |
| [Persist and continue](#persist-and-resume) | `save_bundle`; `load_bundle`; agent checkpoints | Template, arrays, declared learning, or rolling-agent runtime | `LoadedBundle`, restored agent | Serialization preserves declared identity; no inference objective | [learning_bundle.py](examples/learning_bundle.py) | **Ready/current schema:** a bundle is not a mid-run snapshot; stable historical migration is open |
| [Run linear-Gaussian inference](#continuous-exponential-family-and-hybrid-models) | `cffg.models.lgssm` | Gaussian and linear-Gaussian BP with missing observations | `gaussian_marginals`, node statistics, BFE | Exact Kalman/RTS marginals and evidence identity on a tree | [linear_gaussian.py](examples/linear_gaussian.py) | **Ready:** positive-definite covariance contract; scaled readout is still host-loop bound |
| [Run scalar exponential-family inference](#continuous-exponential-family-and-hybrid-models) | `gamma`, `beta`, `poisson_obs`, `bernoulli_obs`; `ef_marginals` | Conjugate Gamma–Poisson and Beta–Bernoulli BP | Standard parameters, mean, `proper`, BFE | Exact registered tree messages and ledger | [exponential_family.py](examples/exponential_family.py) | **Ready:** conditioned observations only; predictive outputs leave the shipped family |
| [Run hybrid mixture and recurrence inference](#continuous-exponential-family-and-hybrid-models) | `gaussian_mixture(_linear)`; `recurrence_link`; `gaussian_prec` | Assumed-density mixtures, structured recurrence, Gamma precision | Mixture predictives, pair beliefs, sufficient statistics | Mixture moment collapse and recurrence JJ bound are explicit approximations | [hybrid_rslds.py](examples/hybrid_rslds.py) | **Bounded:** no exact switching trajectory posterior; partial goals can lose backward influence |
| [Build or fit an rSLDS](#continuous-exponential-family-and-hybrid-models) | `rslds`; `fit_rslds`; `PCAARHMMInitializer`; rSLDS learning classes | Recurrent switching LDS inference, learning, observed-only fitting | Fit status, attempts, marginals, FE/residual traces | Empirical-Bayes variational fit with declared hybrid/JJ approximations | [hybrid_rslds.py](examples/hybrid_rslds.py) | **Bounded:** budget exhaustion is preserved; no broad recovery-superiority claim |
| [Use nonlinear Gaussian factors](#nonlinear-and-neural-gaussian-models) | `FunctionSpec`; `register_mean_function`; `moment_matching` | Taylor or fixed sigma-point positive-noise Gaussian factors | Gaussian marginals, `objective_metadata`, `objective_contract` | Frozen-Gaussian surrogate; adaptive consistency is not an evidence bound | [Sensor](examples/nonlinear_gaussian.py), [reactor planning](examples/cstr_planning.py) | **Bounded:** declared reactor inputs; zero noise, nonlinear GFE, and continuous-to-discrete means are unsupported |
| [Use neural means and fixed-point gradients](#nonlinear-and-neural-gaussian-models) | `neural_mean`; `initialize_neural_parameters`; `FixedPointDifferentiator`; `fixed_point_gradient` | Dense MLP mean inside a positive-noise nonlinear Gaussian factor | Gradient status, adjoint/root residuals, parameter gradient | Local derivative of the certified adaptive root and surrogate ledger | [API pattern](#nonlinear-and-neural-gaussian-models) | **Bounded:** local isolated-root assumption; no weight uncertainty or stochastic mu3 |
| [Train or amortize neural inference](#nonlinear-and-neural-gaussian-models) | `NeuralTrainer`; `TrainingConfig`; `GaussianAmortizer`; `PolynomialFeatures` | Deterministic full-batch Adam and Gaussian recognition initialization | Training/continuation state and proposed Gaussian messages | Optimizes the same approximate adaptive objective | [API pattern](#nonlinear-and-neural-gaussian-models) | **Bounded:** no minibatching, discrete point decisions, or universal speedup claim |
| [Use hard constraints or MAP](#hard-constraints-and-map) | Constraint factors; `compile_template(..., semiring="max")`; `cffg.csp` | Sum-product CSP marginals or discrete max-product | Marginals/BFE, or `decode` and `objective` | Exact tree partition/MAP identities; loopy max-product is bounded | [hard_constraint.py](examples/hard_constraint.py) | **Ready:** max mode is discrete-only; `alldifferent` is capped at `n, K <= 10` |
| [Batch matching structures](#batch-extend-render-and-estimate) | `cffg.colony.Colony` | Vmapped iterative states; per-instance exact schedules | One `RunResult` per instance | Each instance keeps its normal engine objective/status | [API pattern](#batch-extend-render-and-estimate) | **Ready/partial:** functional mu3 rejected; exact schedules are per-instance; deterministic mixed GFE offers compiled batching |
| [Register an ordinary custom factor](#batch-extend-render-and-estimate) | `FactorKernel`; binding classes; `register_kernel` | Single-arena discrete BP/VMP, Gaussian, or scalar EF | Generic marginals, energy, optional `kernel_readouts` | Extension author supplies mathematically valid messages and ledger | [API pattern](#batch-extend-render-and-estimate) | **Pre-1.0 unstable:** specialized/multi-arena variational objects need dedicated engine paths |
| [Render a model](#batch-extend-render-and-estimate) | `cffg.viz` | Deterministic scene, SVG, or PNG from a template | `Scene`, SVG text, image file | Visualization only; no compile or inference certificate | [API pattern](#batch-extend-render-and-estimate) | **Ready:** PNG needs the `viz` extra; legacy moment syntax is rejected |
| [Estimate resources](#batch-extend-render-and-estimate) | `cffg.estimate.estimate` | Static state bytes, optional cost analysis and calibration | `Estimate`, summary, predicted seconds | Measurement/planning aid, not a probabilistic objective | [API pattern](#batch-extend-render-and-estimate) | Exact persistent bytes, backend-reported executable costs, and empirical schedule timing |

## The mental model

A CFFG experiment has four layers:

1. `Template` is the serialized source of truth. It names variables, factors,
   external array references, ports, constraints, plates, learnables, and
   optional agent boundaries.
2. `compile_template(template, arrays=...)` validates that declaration and
   lowers its fixed shapes into JAX arenas, schedules, message rules, parameter
   addresses, and ledger terms.
3. `Engine.init_state()` creates dynamic messages, parameters, observations,
   masks, and specialized solver state. Bind observations and update named
   parameters here; do not rewrite the template for each same-shaped trial.
4. `Engine.run(...)` returns a `RunResult`. Its status and diagnostics decide
   whether the state is usable; the engine's readout and objective contract
   decide what the numbers mean.

Variables are edges in the factor graph. Factors own local probability or
energy terms. Ports identify runtime data, goals, priors, and graph boundaries.
Constraints select a variational interpretation such as mean field,
P-substitution, or endpoint moment matching. Array blobs stay outside the
template so the same serialized structure can be reconstructed with its data.

## Supported facades

These four package facades are the current newcomer import surfaces and are
checked against the implementation. The project is still pre-1.0, so they are
not yet a permanent cross-version compatibility promise. Focused modules such
as `cffg.learn`, `cffg.conj`, `cffg.colony`, `cffg.csp`, and `cffg.estimate`
expose the specialized workflows named in the capability map.

<details>
<summary><code>cffg</code> — daily workflow and extension primitives (38)</summary>

<!-- facade-cffg:start -->
`CorridorEnv`, `DiscreteBinding`, `DiscreteMode`, `EFBinding`, `Engine`,
`FactorKernel`, `FailureDiagnostic`, `FixedPointDifferentiator`, `FixedPointGradient`, `FunctionSpec`,
`GaussianAmortizer`, `GaussianBinding`, `GaussianConditioning`, `GradientStatus`,
`HostileInputs`, `LoadedBundle`, `MeanImplementation`, `NeuralTrainer`,
`ParameterSpec`, `PersistentArray`, `PersistentParameter`,
`PersistentStateDescriptor`, `PersistentTarget`, `PolynomialFeatures`,
`RollingAgent`, `RunStatus`, `TrainingConfig`, `build_gridworld`,
`compile_template`, `fixed_point_gradient`, `initialize_neural_parameters`,
`load_bundle`, `neural_mean`, `save_bundle`, `register_kernel`,
`register_mean_function`, `resolve_persistent_state`, `set_params`
<!-- facade-cffg:end -->

</details>

<details>
<summary><code>cffg.core</code> — serialized graph language (33)</summary>

<!-- facade-core:start -->
`FACTOR_TYPES`, `AgentSpec`, `BeliefBinding`, `BlobRef`, `BoundaryFactorization`,
`Broadcast`, `Chain`, `ConstraintKind`, `ConstraintSpec`, `Expose`, `FactorSpec`,
`FactorType`, `FanOut`, `ObserveInterior`, `FunctionSpec`, `LearnableSpec`, `ModelError`,
`MomentMatchingEndpoint`, `MomentProjection`, `PlateSpec`, `PortRole`, `PortSpec`,
`Template`, `VariableSpec`, `expand_plates`, `fan_out`, `observe_interior`, `plate_observations`,
`migrate_moment_matching`, `moment_matching`, `moment_matching_endpoint`,
`register_factor_type`, `validate`
<!-- facade-core:end -->

</details>

<details>
<summary><code>cffg.models</code> — supported model catalogue (22)</summary>

<!-- facade-models:start -->
`CorridorEnv`, `CouplingSpec`, `ObservationSpec`, `PCAARHMMInitializer`,
`RSLDSFitPriors`, `RSLDSFitPhase`, `RSLDSFitResult`, `RSLDSInitialization`,
`RSLDSInferenceAttempt`, `RSLDSInitializer`, `RSLDSLearning`, `StateBlockSpec`,
`build_factored_corridor`, `build_factored_rollout`, `build_gridworld`,
`build_rollout`, `build_tmaze`, `fit_rslds`, `lgssm`, `rslds`,
`vague_rslds_fit_priors`, `vague_rslds_learning`
<!-- facade-models:end -->

</details>

<details>
<summary><code>cffg.viz</code> — diagram scene and rendering API (22)</summary>

<!-- facade-viz:start -->
`BeadGroup`, `BeadShape`, `CompositeView`, `Diagram`, `FULL_COMPRESSION`,
`LayoutStyle`, `NodeEnd`, `NodeView`, `Presentation`, `Scene`, `SvgStyle`,
`TerminalEnd`, `TerminalKind`, `WireView`, `build_scene`, `derive`, `layout`,
`present`, `render_svg`, `save_png`, `save_svg`, `to_svg`
<!-- facade-viz:end -->

</details>

## Build and compose models

Use `cffg.models` when a supported builder matches the experiment. Use
`cffg.core` when the graph itself is the experiment. Builders return
`(template, arrays)`; they do not hide a compiled engine or mutable global
state.

The shipped factor vocabulary is:

| Family | Built-in factors | What is represented |
|---|---|---|
| Discrete | `categorical`, `matrix_cat`, `transition_mixture`, `dirichlet`, `equality` | Priors, transitions, controlled transitions, parameter edges, copies |
| Hard constraint | `alldifferent`, `exactly_k`, `at_most_k`, `table` | Exact indicator potentials with combinatorial message rules |
| Gaussian | `gaussian`, `gaussian_linear`, `gaussian_nonlinear` | Priors, affine conditionals, registered nonlinear means |
| Scalar exponential family | `gamma`, `beta`, `poisson_obs`, `bernoulli_obs` | Conjugate positive-rate and probability models with conditioned observations |
| Hybrid | `gaussian_mixture`, `gaussian_mixture_linear`, `recurrence_link`, `gaussian_prec` | Switching Gaussian factors, recurrent logits, uncertain precision |

The four serialized constraint kinds are `delta`, `mean_field`,
`p_substitution`, and `moment_matching`. Endpoint moment matching must name the
factor role and `MomentProjection.GAUSSIAN`; factor-only legacy payloads can be
loaded for explicit migration but cannot compile or render.

For reusable structure:

- `cffg.core.compose` scopes, merges, and chains templates.
- `PlateSpec`, `expand_plates`, and `plate_observations` declare repeated
  fragments. Plate fragments cannot nest, and compact plates must be expanded
  before scoping or merging. For a hub shared by 49 copies, the outer
  `Broadcast` has at most six equality factors from hub to copy and at most
  twelve between copies. Factors inside each copy add to the distance that
  evidence must travel under synchronous flooding. On an Apple M2 Max CPU,
  the full 784-pixel classifier converged in 43–44 sweeps and processed
  36.2 images/s singly or 47.7 images/s in batches of 128. For an acyclic
  ordinary BP model, the exact two-pass schedule is also available. It
  processed 263.9 and 273.1 images/s on this classifier.
- `cffg.core.fan_out(template, edge, branch_count)` returns a validated
  template and ordered GLUE ports for attaching new factors. Its endpoints
  keep their roles. `observe_interior(template, edge)` returns a template and
  a DATA port for conditioning a supported latent. For a Gaussian latent
  shared by ordinary priors and affine Gaussian factors, bind a vector to
  this port to observe the latent exactly. Its posterior mean equals the
  vector and its covariance is zero. Bind `None` to clear the observation.
  An ordinary emission DATA edge continues to report its cavity prediction.
- `cffg.core.edits` applies validated all-or-nothing graph edits rather than
  leaving a half-mutated template after a failure.
- `validate(template)` reports structural problems before compilation, while
  compilation additionally checks executable factor/constraint ownership and
  numeric arrays.

## Filter a slow regime and a fast state

Use `cffg.hierarchy.Hierarchy` when a hidden regime selects the fast dynamics
for a known number of observations. For example, the regime below determines
whether the fast state tends to stay put or switch. The sensor sometimes
misreads the fast state.

```python
from cffg.hierarchy import Hierarchy

model = Hierarchy.from_tables(
    parent_prior=[0.5, 0.5],
    parent_transition=[[0.9, 0.1], [0.1, 0.9]],
    child_prior=[0.7, 0.3],
    child_transition=[
        [[0.95, 0.05], [0.05, 0.95]],  # persistent regime
        [[0.10, 0.90], [0.90, 0.10]],  # switching regime
    ],
    emission=[[0.85, 0.15], [0.15, 0.85]],
    thickness=3,
)
for reading in [0, None, 1]:
    result = model.step(reading).require_converged()
    print(result.parent, result.child, result.free_energy)
```

Columns of each transition table index the previous state. Each call advances
one fast transition. A missing reading (`None`) still advances time. On the
third call, `result.joint` holds the posterior just before the slow transition.
`model.current_joint` holds the prediction for the next period and retains
regime–state correlations. `preview(reading)` computes a candidate update
without advancing either clock. Free energy is in nats and covers all the
observations since the start of the current period. Use the last value from
each completed period when summing evidence across periods.

To resume between fast ticks, use `model.save_bundle(path)` and
`Hierarchy.load_bundle(path)`. The hierarchy archive contains both model
templates, their parameter arrays, and the checkpoint for both clocks. With a
learned-goal child planning window, `plan_goals(...)` stages its inferred
Dirichlet counts. Call `finish_episode()` at the real episode boundary to
apply them once. Staged counts survive a save and reload.

To reproduce the registered cue-and-commit comparison, run
`JAX_ENABLE_X64=1 uv run --frozen python tools/hierarchy/drift.py --output drift.json`.
The JSON includes the cue, true regime, posterior and action for each trial.

## Compile, run, and read results

Compile once for a fixed template shape. Observations and same-shaped parameter
values are runtime inputs; a changed horizon, cardinality, dimension, dtype, or
schedule selects another JAX executable. `set_params` performs validated named
writes into `state.params` and preserves serialized parameter tying.

For a tick with many new readings, pass a dictionary to one
`engine.bind_data(state, observations)` call. Omitted ports retain their
bindings. An explicit `None` removes that port's evidence. The engine checks the
supplied readings before writing their values and masks together.
After a solve, request the variables your controller actually consumes with
`engine.marginals(state, edges=["action"])` or
`engine.gaussian_marginals(state, edges=["position"])`, using edge names from
your template. Omitting `edges` returns all variables of that family. The
same selection works on a Colony instance.

Never consume `result.state` as if the iteration budget implied success. Use
`result.require_converged()` or branch explicitly on `result.status`:

| Status | Meaning |
|---|---|
| `CONVERGED` | The selected engine certificate passed for the returned state |
| `BUDGET_EXHAUSTED` | The state may be useful diagnostically, but the requested certificate did not pass |
| `DIVERGED` | A non-finite or failed numerical path invalidated the solve |
| `INFEASIBLE` | The declared support or constraints admit no valid solution under the solver contract |

On an unsuccessful run, inspect `result.failure`: an immutable
`FailureDiagnostic(reason, factor, phase, iteration)`. For example,
`reason="improper_cavity"` identifies an invalid local Gaussian input;
`factor` is its serialized model-local name, not a compiled group name.
If only a global check found the problem, `factor=None`: reasons include
`nonfinite_state`, `empty_support`, `nonfinite_objective`,
`undefined_objective`, `iteration_budget`, and `exact_certificate`.
Local codes also distinguish `local_budget`, `local_divergence`,
`nonfinite_output`, `invalid_noise`, `nonfinite_function`, `improper_tilt`,
`unrepresentable_sigma_points`, and `nonfinite_shared_goal`.
Functional parameter updates use `functional_budget` or `functional_divergence`.

Ordinary sum-mode Gaussian/hybrid initialization can need several sweeps before every belief
has a finite energy. Those sweeps consume the same budget; exhausting it before
any finite energy returns `DIVERGED` with `undefined_objective`, never success.
Ordinary iteration does not silently restart or choose damping. `damp=.5` is
an explicit experiment setting, not a general convergence guarantee.

The first failing update wins; simultaneous failures choose the
alphabetically first factor name. Initialization and priming are iteration
0; message-update attempts start at 1 across soft/hard phases. Exact
propagation and certification both belong to iteration 1. A functional
parameter failure can precede its numbered message-update attempt.
`phase` distinguishes these stages from the `ledger` check.
Adaptive outer solves retain the terminal map's local cause, or a generic
`outer_*` reason when none is known. A local attempt is numbered after completed
controller updates, not by map calls (which also include derivative probes).
Reconstruction has its own `reconstruction` phase. Rejected trials and shadow
proposals that recover do not become run failures; a collapsed trust region
names a failed trial only when every trial at that step was invalid.

Converged results have `failure=None`. Diagnostics are run-local and are not
saved in agent checkpoints: a restored cached unsuccessful result can also
have `failure=None`. Always use `status` to decide whether inference succeeded.

Readouts are family-specific on purpose:

| Model family | Readouts to inspect |
|---|---|
| Discrete | `marginals`, `node_beliefs`, `dirichlet_marginals`, `goal_marginals`, `goal_prior_updates` |
| Gaussian | `gaussian_marginals`, `gaussian_node_stats` |
| Exponential family | `ef_marginals`; check every marginal's `proper` field before using its mean |
| Hybrid recurrence/mixture | `recurrence_pair_beliefs`, `recurrence_weight_stats`, `gaussian_mixture_predictives`, `hybrid_gfe_diagnostics` |
| Registered extension | `kernel_readouts` when the binding declares one |
| Discrete max-product | `decode` plus `objective`; BFE is deliberately undefined |

The common `engine.bfe(state)` accessor reports the graph's declared variational
ledger; the name does not make every value an exact Bethe evidence identity.
Interpret it with `engine.objective_contract` and per-factor
`engine.objective_metadata`:

| Situation | Meaning of the reported objective |
|---|---|
| Ordinary registered sum-product tree | Exact BFE and `-log p(observed data)` |
| Loopy ordinary graph | Bethe free energy at a BP fixed point; marginals and evidence are approximate |
| P-substituted discrete graph | GFE for the supported LAIF composite; no generic evidence identity |
| Gaussian mixture | Assumed-density collapse; the reported BFE lower-bounds the mixture-Bethe value at the held moments |
| `recurrence_link` | Variational objective containing the Jaakkola–Jordan transition bound |
| Nonlinear/neural Gaussian | Frozen-Gaussian surrogate with conditional stationarity metadata; no proven original-model evidence bound |
| Discrete max-product | `objective(state)` is decoded assignment energy; `bfe(state)` raises |

Also inspect `proper` flags, support diagnostics, effective sample size, and
specialized residuals when those fields exist. A finite scalar energy cannot
override a failed status or improper readout.

## Discrete planning and rolling agents

`build_rollout` and `build_tmaze` expose the general tabular path;
`build_gridworld` is the compact supported corridor path. Their switches select
fixed or learned likelihoods and goals, functional mu3(A), and transition
epistemics. These are different declared objectives, not interchangeable tuning
flags. `build_factored_rollout` keeps named hidden-state blocks separate until a
declared coupling or observation needs their product.

`RollingAgent` reconstructs finite planning and filtering windows around a
fixed-size boundary. The usual loop is `act`, environment transition,
`observe`, and optional declared learning; `step` combines those operations for
a callback-style environment. `checkpoint` and `restore` preserve agent
beliefs, learning state, cadence, RNG, and functional-mu3 state when configured.
They intentionally do not serialize the environment.

Functional mu3 uses explicit JAX keys and fixed particles within one inference
attempt. Record ESS and support facts, and use `score_mu3_epochs` only when the
experiment deliberately averages refreshed particle epochs. Transition
epistemics price latent next-state information optimistically; consult
[Known Limitations](docs/LIMITATIONS.md#soft-limitations) before interpreting
it as realizable information gain.

Two current input hazards need caller-side checks: fixed GFE likelihoods and
goals must be finite, nonnegative, and normalized along their documented
probability axes; caller-supplied fixed mu3 particles must be finite,
nonnegative probability tables. Some low-level paths do not yet reject every
violation before it can corrupt the reported objective or moment.

`cffg.experiments.behavioral_epistemics.run_benchmark` is the bounded costly-cue
scientific oracle used to compare epistemic planning with a BFE ablation. It is
a fixed research workload, not a general environment API. Other modules under
`cffg.experiments` and `cffg.benchmarks` are derivation, diagnostic, or
performance artifacts rather than supported newcomer facades.

### Commit to an action after each possible reading

For one or two future categorical readings, solve a policy once and execute its
branches by lookup. This example has a 90%-accurate cue costing one utility
unit and a correct/incorrect guess worth +4/−4.

```python
from cffg.contingent import ContingentPlanner
from cffg.experiments import behavioral_epistemics as cue
from cffg.models.contingent import build_contingent_rollout

task = cue.CueTask(cue_quality=.9, cue_cost=1., reward_utility=4.)
template, arrays = build_contingent_rollout(
    "paid cue", depth=1, d=cue.initial_state(), Bs=cue.transition_stack(),
    A=cue.observation_matrix(task), c=cue.goal_statistic(task),
)
policy = ContingentPlanner(template, arrays).solve()
policy.action()                  # 2: buy the cue
policy.action([cue.CUE_LEFT])    # 0: choose left
policy.action([cue.CUE_RIGHT])   # 1: choose right
```

The commitment earns 2.2 expected utility. Passing `open_loop=True` ties
both responses to the same action and gives an optimum of zero. Each reachable
history has its physical probability and conditional action distribution in
`policy.branches`. `policy.free_energy` scores the whole policy distribution
and `policy.committed_free_energy` scores the selected commitment. Both are
native tree Bethe energies in nats. `policy.expected_loss` sums its observed
negative log preferences. For direct history tables use `build_contingent`;
both builders produce a template and arrays suitable for `save_bundle`.

This planner is opt-in. Existing agents retain their priors and default
horizons. In the example above, increasing to depth two generated 57 histories
and 73,392 CPT entries across factor instances. Shared tables occupied 95,424
uncompressed source bytes and the saved bundle occupied 37,907 bytes. On an
Apple M2 Max CPU in float32, the first depth-two solve took **24.4 s** including
JAX compilation and process peak memory was **1.64 GB**. Native preparation
took another 0.29 s. Once warm, solving again took a median of 14.57 ms.
Executing an already chosen branch is a lookup. See D17 in
[the derivations](docs/DERIVATIONS.md#d17-committing-to-a-categorical-action-after-each-reading).

## Continuous, exponential-family, and hybrid models

`lgssm` constructs a linear-Gaussian state-space model with DATA ports for
conditioning and missing observations. On its tree, Gaussian marginals match
Kalman/RTS inference and the ledger has the exact negative-log-evidence
identity. See [examples/linear_gaussian.py](examples/linear_gaussian.py).

An exact-tree certificate checks message consistency, not posterior accuracy.
Choose float64 for tight moment or evidence comparisons, especially with widely
different variances. The retained PCGym first-order controller certifies in both
precisions, but float32 still loses accuracy in its information-form messages
and readouts. Its tests separately check posterior-scaled moments and evidence;
the bounded precision contract is in [Gaussian numerics](docs/LIMITATIONS.md#soft-limitations).

Gamma–Poisson and Beta–Bernoulli graphs use scalar natural-parameter messages.
Observation factors are conditioned leaves; unobserved Poisson or Bernoulli
predictives leave the shipped conjugate family and are rejected. See
[examples/exponential_family.py](examples/exponential_family.py).

Hybrid mixture factors collapse Gaussian mixtures by moments so inference stays
tractable. `recurrence_link` uses a structured source/destination categorical
belief and a Pólya–Gamma/Jaakkola–Jordan coordinate. `rslds` assembles this
production path; `fit_rslds` adds an observed-only empirical-Bayes fitting loop
with truthful inner and outer statuses. These are named approximations, not
exact switching-trajectory inference. The deferred D12 research path described
in `DERIVATIONS.md` is not a production capability. See
[examples/hybrid_rslds.py](examples/hybrid_rslds.py).

For hybrid control, test that a distant goal actually changes the first action.
Rank-deficient backward mixture messages currently become vacuous. A goal on
position alone in a position/velocity model can therefore produce a converged
result whose first action ignores later goals. Convergence certifies the
declared approximate updates, not useful lookahead; see LIMITATIONS S1d.

## Nonlinear and neural Gaussian models

A `gaussian_nonlinear` factor references a serialized `FunctionSpec`.
`register_mean_function` binds a trusted local implementation by name and
version; choose Taylor linearization or fixed sigma points and declare endpoint
moment matching. Noise covariance must be positive definite. The runtime checks
the complete adaptive residual and energy change, while
`objective_metadata` labels the surrogate's narrower meaning. See
[examples/nonlinear_gaussian.py](examples/nonlinear_gaussian.py).

For a CPU Taylor chain with a Gaussian prior and a linear DATA emission at
each time, the default engine can solve both message directions in one pass
each. Pass a positive iteration budget and a tolerance. Inspect the method
that actually certified the result before comparing timings.

```python
result = engine.run(bound_state, n_iters=100, tol=1e-6)
state = result.require_converged()
mean, covariance = engine.gaussian_marginal(state, action_edge)
method = result.termination_reason.value  # "directed_chain_certified" on this route
```

The directed solver checks every message against an ordinary engine step
and evaluates the frozen Bethe objective before and after that step. The first
solve compiles the step and objective. Later solves with the same shapes
reuse those compiled functions. Missing observations and partial goals are
supported. For a matched coupled-solver reference on the same graph, pass
`outer_solver="trust_region"`. The single-edge readout also works with that
reference. It reads the named edge in the model's own units.

Fresh neutral adaptive sites can wait for partial input information during
bounded startup; those sweeps count toward `n_iters`. Invalid models and
established sites remain strict. Unresolved startup reports budget exhaustion
with the pending factor, not convergence or a fabricated Gaussian belief.

Gaussian-only adaptive plans use compiled, fully adaptive proposals before
trust inference. Each proposal refreshes the nonlinear factors; only the full
root, energy and domain certificate can accept it. Unsuccessful proposals use
the remaining budget for trust inference. Same-shape observations reuse kernels
with fresh, solve-fixed chart metrics. This is not a universal convergence or
speed guarantee; discrete/EF/functional plans retain the trust route.

To reproduce the recent wide-prior P4/H15 CSTR runtime workload, run
`uv run python tools/benchmark_adaptive_runtime.py --output /tmp/cstr-runtime.json`.
It checks the full objective and all state marginals, reports cold/repeated-base/
changed-observation timings and executable counts, and never warm-starts a plan.
Use independent worktrees with `uv sync --frozen` for comparisons. This workload
is distinct from the smaller example below; sigma-point startup still fails.
On the matched Apple M2 Max/locked-JAX fixture, repeated-base plans fell from
36.93 s to 0.106 s (349×), and changed-observation plans from 120.68 s to
0.108 s (1122×), with unchanged 1e-6 certificates and no warm recompilation.
Cold first-plan time was 130.72 s versus 26.48 s; see `LIMITATIONS.md` for scope.

For physical-model planning, run
`uv run python examples/cstr_planning.py --method sigma_point --dtype float32`.
The registered RK4 reactor mean is evaluated by CFFG; no caller-side
linearization is needed. This declared two-step example starts from a narrow
prior and observes the current concentration/temperature. It reports status,
adaptive residual, approximate energy, failure attribution and planned coolant
moments. Both methods, precisions and schedules have cold/warm-start checks;
the temperature-goal noise grid is specified in `LIMITATIONS.md`. Other
priors/horizons need validation. Evaluation clipping is not a hard actuator
constraint. Use `--save model.cffgz` to save sources and arrays, then install the
trusted callback with `install_mean()` before `load_bundle()`; supply DATA
bindings separately.

For a **directed CFFG reactor plan with actuator limits**, declare the constrained
coordinates on the source template. In a four-history, two-command window the
coolant is coordinate 3 of each transition input. Bounds use the graph's
normalized units, where −1 and +1 correspond to 295 and 302 K.

```python
from cffg.bounded_chain import BoundedChainRegion, BoundedCoordinate
from cffg.bounded_actions import BoundStatus

region = BoundedChainRegion(template, (
    BoundedCoordinate("s4c", 3, -1., 1.),
    BoundedCoordinate("s5c", 3, -1., 1.),
))
engine = region.compile(arrays=arrays)
# Bind the usual window DATA to this engine's state before running it.
result = engine.run(bound_state, 100, tol=1e-6)
decision = region.evaluate(engine, result)
if decision.bounded.status == BoundStatus.SUCCESS:
    coolant_kelvin = 298.5 + 3.5 * float(decision.bounded.command)
```

The evaluator derives the trajectory covariance and normalized objective from
the certified directed result's live parameters, observations and transition
cavities. It raises on an unsupported declaration, foreign or uncertified
result, or failed certificate check. The declaration travels as a JSON-compatible
`region.to_dict()` document containing the source template, variable names,
coordinates and bounds. Restore it with `BoundedChainRegion.from_dict`;
`region.scoped(prefix)` renames the whole model and its selected edges together.
Supply the referenced blob arrays or root at compilation, and install the
registered mean implementation before loading the declaration.

Run `uv run python tools/cstr_bounded_comparison.py` for the actual P4/H2
four-substep RK4 reactor comparison. Both readouts share the same directed
solve and available observations, including the missing history. The example
reports each command, normalized objective and physical one-step tracking error.
Each arm times its first and repeated decision readouts after the common solve.
The clipped arm reads only the two selected edges. Both clocks stop after the
proposals, command and objective are available on the host.
A connected chain can constrain one or two action coordinates.

For a separately supplied **frozen Gaussian plan with actuator limits**, declare the hard box and
supply the joint Gaussian in action-then-state order. The first action is the
conditional mean under the entire box. For example, a two-command horizon is

```python
from cffg.bounded_actions import BoundedGaussianPlan, BoundStatus

box = BoundedGaussianPlan((2,), (-1., -1.), (1., 1.))
result = box.solve_joint(mean, covariance, gaussian_free_energy)
if result.status == BoundStatus.SUCCESS:
    command = result.command
```

Supply the complete free energy of the unbounded frozen region in nats.
`result.free_energy` includes its box correction once. When the model provides
structurally independent Gaussian action marginals, use `box.solve` with
one covariance matrix per declared width and optional affine conditional
state parameters. Multiple independent blocks can share a downstream state.
The declaration round-trips through `to_dict` and `from_dict`. Run
`uv run python examples/bounded_coolant.py` for a matched two-command
surrogate in 295–302 K coolant units.

`neural_mean` creates a portable dense MLP specification and
`initialize_neural_parameters` creates its arrays. `FixedPointDifferentiator`
or `fixed_point_gradient` differentiates the certified adaptive root and
reported approximate ledger; check `GradientStatus` and both root and adjoint
residuals. `NeuralTrainer` with `TrainingConfig` provides deterministic
full-batch Adam continuation. `GaussianAmortizer` learns a Gaussian message
initializer over declared `PolynomialFeatures`; the normal engine still makes
the final convergence and properness decision.

These neural APIs do not provide zero-noise equality factors, minibatching,
weight uncertainty, stochastic mu3 differentiation, or discrete point-decision
gradients.

## Learn and swap parameters

`LearnableSpec` declares which serialized factors own persistent posterior
state. Construct `cffg.learn.Learnables` with immutable priors, write its current
expected parameters into a fresh engine state, run and status-gate inference,
then choose the cadence deliberately:

- `accumulate(engine, result)` adds statistics from a new independent trial to
  the current posterior.
- `replace(engine, result)` recomputes one sequence coordinate from the
  immutable initial prior; repeated use is an outer variational/EM coordinate,
  not another independent trial.
- `expected(name)` returns point parameters for the next solve.
- `rollover(name)` returns the family-specific state carried across a declared
  trial boundary.

Dirichlet, NIW, MNIW, component-MNIW, and recurrence Pólya–Gamma families have
versioned persistent descriptors. Their expected point parameters feed the next
state; parameter uncertainty generally does not enter all state messages, so
the continuous/hybrid cadence is empirical Bayes rather than full joint
variational Bayes. See
[examples/learning_bundle.py](examples/learning_bundle.py).

For fixed-structure parameter sweeps that are not declared learning,
`set_params` updates named runtime parameters without recompiling. Shapes,
factor roles, probability normalization, covariance validity, and tied blobs
are validated.

## Persist and resume

Choose the artifact by what must continue:

| Artifact | Stores | Does not store |
|---|---|---|
| Single-template `.cffgz` bundle | Template, source arrays, structure identity, complete declared-learning priors and posteriors | Inference messages, solver progress, agent cadence/RNG, optimizer cursor |
| `Hierarchy.save_bundle` archive | Both templates and arrays, joint boundary, level clocks and statistics, optional child goal window and pending counts | Environment state |
| `RollingAgent` `.npz` checkpoint | Structure-bound rolling beliefs, learning state, cadence, selection RNG, configured functional-mu3 lifecycle | Environment state or a portable replacement for the model bundle |
| Neural training continuation | Versioned parameters and deterministic optimizer/training state for its bound initialization and data | General model/agent state or minibatch cursor |

`save_bundle` and `load_bundle` are the source-of-truth model exchange path.
Recompiling the loaded template and arrays must reconstruct the engine. The
current alpha loader supports the current schema and refuses older or
provenance-poor learning archives rather than inventing resumability. A stable
historical migration contract is future work.

## Hard constraints and MAP

Hard factors can be combined with ordinary discrete probability factors:

- `alldifferent` uses exact combinatorial messages and is capped at
  `n, K <= 10`.
- `exactly_k` and `at_most_k` constrain Boolean-style counts.
- `table` admits explicitly allowed tuples.

In sum-product mode, tree marginals and the hard-factor contribution to BFE are
exact. Compile with `semiring="max"` for discrete max-product, status-gate the
run, call `decode(state)`, and score it with `objective(state)`. On a tree this
is the exact MAP/Viterbi identity. On loops it retains the usual max-product
caveats; `cffg.csp.decimate` is the explicit feasibility-oriented helper. See
[examples/hard_constraint.py](examples/hard_constraint.py).

## Batch, extend, render, and estimate

`cffg.colony.Colony` caches one engine per structure hash and vmaps matching
ordinary iterative state layouts. Each member has its own stopping condition
inside a compiled sweep loop and returns its own state, status and traces in a
`RunResult`. A member that stops early keeps its final state while peers continue.
Exact schedules use the same per-instance propagation and certification as
`Engine.run`; they do not fall back to flooding just because they are in a colony.
Different parameters do not create different structures. Functional mu3 is
rejected because Colony does not yet own per-instance PRNG streams.

Mixed GFE uses the host trust controller by default. For repeated solves or a
batch of deterministic GFE members, select the compiled controller explicitly:

```python
result = engine.run(state, 80, tol=1e-8, outer_solver="compiled_trust_region")
by_id = colony.run(80, tol=1e-8, outer_solver="compiled_trust_region")
```

Members sharing a support face can run together while retaining separate
radii, stopping decisions and root certificates. First use can take longer
and use more memory. Adaptive Gaussian and stochastic functional updates
use their existing solvers.

### Independent CPU solves with host threads

For a batch of ordinary iterative solves, you can share one warmed `Engine`
across Python threads. Prepare an immutable `State` per member on the calling
thread. Workers share model parameters and return separate `RunResult` objects.
Keep any changes to Colony membership or stored states on the calling thread.

```python
from concurrent.futures import ThreadPoolExecutor

import jax
from cffg import Engine, build_gridworld, compile_template

template, arrays = build_gridworld(past=1, steps=1, k=4, gfe=False)
engine = Engine(compile_template(template, arrays=arrays))
base = engine.init_state()
states = [engine.bind_data(base, {"act1": 1, "obs1": cell}) for cell in range(4)]

def solve(state):
    result = engine.run(state, 120, damp=0.5, tol=1e-5)
    jax.block_until_ready(result.state)
    return result

solve(states[0])  # Warm the executable before starting workers.
with ThreadPoolExecutor(max_workers=4) as pool:
    results = list(pool.map(solve, states))  # Same order as the inputs.
posteriors = [engine.marginals(result.require_converged())["zm1"] for result in results]
```

On an Apple M2 Max CPU with JAX 0.10.2 in float64, 32 independent Hallway
priors shared one engine for 60 damped sweeps. That benchmark used `tol=None`,
so all 32 results reported `budget_exhausted` as expected. The medians below
cover three warm batches per worker count.

| Host workers | Wall time for all 32 | Solves/s | Member p95, excluding queue | Process peak RSS before output validation |
|---|---:|---:|---:|---:|
| Serial | 2.490 s | 12.9 | 79 ms | 439 MiB |
| 4 | 0.877 s | 36.5 | 112 ms | 513 MiB |
| 8 | 0.748 s | 42.8 | 194 ms | 512 MiB |

Eight workers finished the batch 3.33× faster than serial. Going from four
to eight gained 17% throughput while increasing time in service at p95 by
74%. Every member's state, status and traces matched serial execution exactly
on this workload. Time your own batch with the CPU capacity available to it.
Host threading is opt-in. The separate 8× throughput target for `Colony.run`
is tracked in [chr-d4dl](.tickets/chr-d4dl.md).

### Custom factors, rendering, and resource estimates

The custom factor API registers one ordinary single-arena factor at a time:

- `DiscreteBinding` covers discrete BP/VMP with a declared potential,
  messages, energy, assignment scoring, and optional readout.
- `GaussianBinding` covers a Gaussian conditioning rule with its parameter
  schema and optional iterative behavior.
- `EFBinding` covers scalar exponential-family natural-parameter rules.
- `HostileInputs` records the extension's zero-support, improper/vacuous,
  float32, and scale contract.

`FactorKernel` and `register_kernel` connect those pieces. Registration does not
derive mathematical correctness, a learning family, or a specialized
multi-arena variational object. This surface is explicitly internal-unstable
before 1.0 even though serialized implementation identity and loud missing-
implementation failure are supported.

`cffg.viz` derives a deterministic scene and renders SVG directly; PNG uses the
optional `viz` dependencies. Rendering explains declared structure and
constraints but does not compile or validate numerical inference.
External wires follow deterministic routes around factor boxes. At crowded
factors, ports and parameter terminals use the free sides before sharing one.
A variable name sits beside its wire. A parameter name sits at its square terminal.
Terminals that share a side extend different distances so each caption has room.
Disconnected components are laid out independently and stacked without overlap.
Use `render_svg` for a template, or `to_svg` after constructing a scene:

```python
from cffg.viz import build_scene, render_svg, to_svg

svg_text = render_svg(template)       # Template -> SVG string
scene = build_scene(template)        # Template -> positioned Scene
assert to_svg(scene) == svg_text     # Scene -> SVG string
for label in scene.labels():
    print(label.ref, label.text, label.bounds)
```

Each caption gives its template reference, baseline in pixels, horizontal anchor,
font role, and bounding box as `(left, top, right, bottom)`. Bounds use a
deterministic estimate that works without installed fonts. `render_svg` lays out
captions using the requested SVG font sizes.

`cffg.estimate.estimate(engine)` reports exact bytes per persistent arena and parameter
leaf. Its named XLA executables expose backend-reported FLOPs, byte traffic and
planned buffers where supported. For example, `report.executables["flooding_step"]["cost_diagnostic"]`
gives the backend, exception type and message when cost analysis is unavailable.
Each compiled unit has separate lowering, compilation and first-execution times.
Calibration also times the first end-to-end run, including any lazy controller
compilation, and at least five synchronized warm runs. `report.prediction_unit`
identifies the active schedule used by `predict_seconds`. Exact two-pass solves have a
one-run timing. Studio's fresh-process profile also measures startup, compilation
events and peak resident memory.

## Experiment checklist

For a result that another agent can compare or reproduce, record:

- repository commit, Python/JAX versions, backend, dtype, and whether x64 was
  enabled before import;
- serialized template/bundle identity, model dimensions, schedule, damping,
  iteration budgets, tolerance, and explicit PRNG keys;
- `RunStatus`, iteration count, residual and energy traces, plus every relevant
  `proper`, support, ESS, bound, root, or adjoint diagnostic;
- `objective_contract` and applicable `objective_metadata`, not just the scalar
  returned by `bfe`;
- compilation/first-call time separately from warm execution time;
- whether learning used `accumulate`, `replace`, direct `set_params`, or a fit
  helper, and which artifact is required to resume it.

Use small analytic trees to establish exact identities before moving to loopy,
hybrid, nonlinear, or long-horizon experiments. Increase dimensions under both
the default JAX dtype and x64 when numerical robustness matters.

## Runnable examples

The examples are standalone programs with their own fixtures. Each rejects an
unfinished result and asserts the quantity an experiment should inspect.

| File | Demonstrates | Claim boundary |
|---|---|---|
| [discrete_agent.py](examples/discrete_agent.py) | Bundle, compile/bind/run/read, rolling agent, checkpoint restore | Exact conditioned tree plus registered fixed-A planning |
| [discrete_rollout.py](examples/discrete_rollout.py) | Generic finite-horizon discrete GFE planning | Supported one-step P-substituted objective |
| [linear_gaussian.py](examples/linear_gaussian.py) | LGSSM with a missing observation | Exact tree Gaussian marginals and evidence identity |
| [exponential_family.py](examples/exponential_family.py) | Gamma–Poisson conjugate inference | Exact registered tree posterior and BFE |
| [hard_constraint.py](examples/hard_constraint.py) | All-different marginals and max-product MAP | Exact bounded tree CSP identities |
| [nonlinear_gaussian.py](examples/nonlinear_gaussian.py) | Sigma-point nonlinear Gaussian inference | Declared approximate objective metadata |
| [cstr_planning.py](examples/cstr_planning.py) | Registered reactor physics, partial goals, status-bearing plans, portable source bundle | Declared short-horizon Gaussian approximation; no hard actuator bounds |
| [learning_bundle.py](examples/learning_bundle.py) | Declared learning, save/load, later-trial continuation | Current-schema immutable prior plus posterior state |
| [hybrid_rslds.py](examples/hybrid_rslds.py) | Small recurrent switching LDS | Declared mixture/JJ approximation |

Run one from the repository checkout with, for example:

```sh
uv run python examples/linear_gaussian.py
```

## Reference map

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — owners, execution paths, and
  extension change paths.
- [docs/LIMITATIONS.md](docs/LIMITATIONS.md) — exactness, approximation,
  numerical, scaling, and known-defect boundaries.
- [docs/DERIVATIONS.md](docs/DERIVATIONS.md) — project derivations and executable
  test anchors beyond the papers.
- [docs/READING.md](docs/READING.md) — short external reading list.
- [notes/cffg-pt-1.pdf](notes/cffg-pt-1.pdf) and
  [notes/cffg-pt-2.pdf](notes/cffg-pt-2.pdf) — the two CFFG/LAIF papers.
- `src/cffg/core/` — serialized graph language and validation.
- `src/cffg/models/` — supported builders and rSLDS workflow.
- `src/cffg/readouts/` — family-owned result interpretation.
- `tests/` — executable contracts and numerical oracles.

Contributing changes to the library? Read [AGENTS.md](AGENTS.md) for the uv,
ticket, commit, numerical-robustness, and validation protocols.
