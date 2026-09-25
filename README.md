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
| [Run hybrid mixture and recurrence inference](#continuous-exponential-family-and-hybrid-models) | `gaussian_mixture(_linear)`; `recurrence_link`; `gaussian_prec` | Assumed-density mixtures, structured recurrence, Gamma precision | Mixture predictives, pair beliefs, sufficient statistics | Mixture moment collapse and recurrence JJ bound are explicit approximations | [hybrid_rslds.py](examples/hybrid_rslds.py) | **Bounded:** no exact switching trajectory posterior or general evidence bound |
| [Build or fit an rSLDS](#continuous-exponential-family-and-hybrid-models) | `rslds`; `fit_rslds`; `PCAARHMMInitializer`; rSLDS learning classes | Recurrent switching LDS inference, learning, observed-only fitting | Fit status, attempts, marginals, FE/residual traces | Empirical-Bayes variational fit with declared hybrid/JJ approximations | [hybrid_rslds.py](examples/hybrid_rslds.py) | **Bounded:** budget exhaustion is preserved; no broad recovery-superiority claim |
| [Use nonlinear Gaussian factors](#nonlinear-and-neural-gaussian-models) | `FunctionSpec`; `register_mean_function`; `moment_matching` | Taylor or fixed sigma-point positive-noise Gaussian factors | Gaussian marginals, `objective_metadata`, `objective_contract` | Frozen-Gaussian surrogate; adaptive consistency is not an evidence bound | [nonlinear_gaussian.py](examples/nonlinear_gaussian.py) | **Bounded:** zero noise, nonlinear GFE, and continuous-to-discrete means are unsupported |
| [Use neural means and fixed-point gradients](#nonlinear-and-neural-gaussian-models) | `neural_mean`; `initialize_neural_parameters`; `FixedPointDifferentiator`; `fixed_point_gradient` | Dense MLP mean inside a positive-noise nonlinear Gaussian factor | Gradient status, adjoint/root residuals, parameter gradient | Local derivative of the certified adaptive root and surrogate ledger | [API pattern](#nonlinear-and-neural-gaussian-models) | **Bounded:** local isolated-root assumption; no weight uncertainty or stochastic mu3 |
| [Train or amortize neural inference](#nonlinear-and-neural-gaussian-models) | `NeuralTrainer`; `TrainingConfig`; `GaussianAmortizer`; `PolynomialFeatures` | Deterministic full-batch Adam and Gaussian recognition initialization | Training/continuation state and proposed Gaussian messages | Optimizes the same approximate adaptive objective | [API pattern](#nonlinear-and-neural-gaussian-models) | **Bounded:** no minibatching, discrete point decisions, or universal speedup claim |
| [Use hard constraints or MAP](#hard-constraints-and-map) | Constraint factors; `compile_template(..., semiring="max")`; `cffg.csp` | Sum-product CSP marginals or discrete max-product | Marginals/BFE, or `decode` and `objective` | Exact tree partition/MAP identities; loopy max-product is bounded | [hard_constraint.py](examples/hard_constraint.py) | **Ready:** max mode is discrete-only; `alldifferent` is capped at `n, K <= 10` |
| [Batch matching structures](#batch-extend-render-and-estimate) | `cffg.colony.Colony` | Vmapped iterative states; per-instance exact schedules | One `RunResult` per instance | Each instance keeps its normal engine objective/status | [API pattern](#batch-extend-render-and-estimate) | **Ready/partial:** functional mu3 rejected; exact and mixed controllers remain per-instance |
| [Register an ordinary custom factor](#batch-extend-render-and-estimate) | `FactorKernel`; binding classes; `register_kernel` | Single-arena discrete BP/VMP, Gaussian, or scalar EF | Generic marginals, energy, optional `kernel_readouts` | Extension author supplies mathematically valid messages and ledger | [API pattern](#batch-extend-render-and-estimate) | **Pre-1.0 unstable:** specialized/multi-arena variational objects need dedicated engine paths |
| [Render a model](#batch-extend-render-and-estimate) | `cffg.viz` | Deterministic scene, SVG, or PNG from a template | `Scene`, SVG text, image file | Visualization only; no compile or inference certificate | [API pattern](#batch-extend-render-and-estimate) | **Ready:** PNG needs the `viz` extra; legacy moment syntax is rejected |
| [Estimate resources](#batch-extend-render-and-estimate) | `cffg.estimate.estimate` | Static state bytes, optional cost analysis and calibration | `Estimate`, summary, predicted seconds | Measurement/planning aid, not a probabilistic objective | [API pattern](#batch-extend-render-and-estimate) | **Partial:** persistent arena bytes are useful; current XLA FLOP/traffic analysis is defective |

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
<summary><code>cffg.core</code> — serialized graph language (29)</summary>

<!-- facade-core:start -->
`FACTOR_TYPES`, `AgentSpec`, `BeliefBinding`, `BlobRef`, `BoundaryFactorization`,
`Broadcast`, `Chain`, `ConstraintKind`, `ConstraintSpec`, `Expose`, `FactorSpec`,
`FactorType`, `FunctionSpec`, `LearnableSpec`, `ModelError`,
`MomentMatchingEndpoint`, `MomentProjection`, `PlateSpec`, `PortRole`, `PortSpec`,
`Template`, `VariableSpec`, `expand_plates`, `plate_observations`,
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
  before scoping or merging.
- `cffg.core.edits` applies validated all-or-nothing graph edits rather than
  leaving a half-mutated template after a failure.
- `validate(template)` reports structural problems before compilation, while
  compilation additionally checks executable factor/constraint ownership and
  numeric arrays.

## Compile, run, and read results

Compile once for a fixed template shape. Observations and same-shaped parameter
values are runtime inputs; a changed horizon, cardinality, dimension, dtype, or
schedule selects another JAX executable. `set_params` performs validated named
writes into `state.params` and preserves serialized parameter tying.

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

The first failing update wins; simultaneous failures choose the
alphabetically first factor name. Initialization and priming are iteration
0; message-update attempts start at 1 across soft/hard phases. Exact
propagation and certification both belong to iteration 1. A functional
parameter failure can precede its numbered message-update attempt.
`phase` distinguishes these stages from the `ledger` check.
Adaptive outer solves currently report generic `outer_*` terminal reasons;
rejected trials that recover do not become run failures.

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

## Continuous, exponential-family, and hybrid models

`lgssm` constructs a linear-Gaussian state-space model with DATA ports for
conditioning and missing observations. On its tree, Gaussian marginals match
Kalman/RTS inference and the ledger has the exact negative-log-evidence
identity. See [examples/linear_gaussian.py](examples/linear_gaussian.py).

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

## Nonlinear and neural Gaussian models

A `gaussian_nonlinear` factor references a serialized `FunctionSpec`.
`register_mean_function` binds a trusted local implementation by name and
version; choose Taylor linearization or fixed sigma points and declare endpoint
moment matching. Noise covariance must be positive definite. The runtime checks
the complete adaptive residual and energy change, while
`objective_metadata` labels the surrogate's narrower meaning. See
[examples/nonlinear_gaussian.py](examples/nonlinear_gaussian.py).

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
| `.cffgz` bundle | Template, source arrays, structure identity, complete declared-learning priors and posteriors | Inference messages, solver progress, agent cadence/RNG, optimizer cursor |
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
ordinary iterative state layouts while returning a separate `RunResult` per instance.
Exact schedules use the same per-instance propagation and certification as
`Engine.run`; they do not fall back to flooding just because they are in a colony.
Different parameters do not create different structures. Functional mu3 is
rejected because Colony does not yet own per-instance PRNG streams, and mixed
adaptive/GFE controllers still run per instance.

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

`cffg.estimate.estimate(engine)` reports persistent arena bytes, a parameter
total, optional backend cost analysis, and a short calibration extrapolation.
Use the persistent byte inventory as a planning aid only. The current XLA
analysis calls an obsolete execution signature, so FLOP and traffic fields may
be absent; the summary is not peak RSS, compile time, or a reliable cross-model
performance forecast.

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
