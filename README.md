# cffg

Constrained Forney-style Factor Graphs (CFFG) and Lagrangian Active Inference
(LAIF) in JAX — an engine for building **white-box probabilistic agents**:
models you can read, edit, checkpoint, and ship, whose exploration comes from
their objective (Generalised Free Energy) rather than bolted-on noise.

Implements Koudahl, van de Laar & de Vries, *Realising Synthetic Active
Inference Agents* Parts I & II (arXiv:2306.08014), validated to 1.7e-6
against the authors' Julia reference on the T-maze benchmark.

```sh
uv sync          # install
uv run pytest tests/test_engine_tree.py  # focused iteration
uv run python tools/dashboard.py --run-suite  # one final full pass + dashboard
```

Run focused tests as often as useful while changing code. One passing full
suite after the final code change is sufficient to open or update a PR; the
dashboard command above runs pytest exactly once and renders that result.
Before starting or waiting on the full pass, inspect
`tools/test-suite-runtimes.json`: it records recent measured wall times and a
conservative `suggested_wait_seconds`.
Ticket- and documentation-only changes use their relevant validation instead
of the full suite.

## The mental model

A model is a **Template**: a factor graph (square factors, edges = variables)
plus **constraint beads** that pick the variational objective, plus typed
**ports** where the world plugs in.

| Piece | Meaning |
|---|---|
| `VariableSpec` | a discrete random variable (an edge; connects ≤ 2 factors) |
| `FactorSpec` | a distribution node: `categorical`, `matrix_cat`, `transition_mixture`, `equality`, `dirichlet` |
| `ConstraintSpec` | a bead: `P_SUBSTITUTION` (adds the epistemic/GFE drive), `DELTA` (MAP readout), `MEAN_FIELD` (VMP) |
| `PortSpec` | a dangling edge with a role: `data` (sensor), `goal`, `prior`, `glue` (composition) |
| `AgentSpec` | serialized action, sensor, actuator, and rolling-boundary bindings for `RollingAgent` |

The load-bearing invariant: every variable is either fully connected or
exposed through exactly one port — so "no dangling edges" and "models
compose" are the same rule.

**The one bead that matters most:** `P_SUBSTITUTION` on an observation factor
facing a goal factor turns that pair into a GFE composite — the agent then
*values information* about that observation. Without it you get a KL-control
agent that never explores (we test this: same graph, one bead, 90% vs 0% on
the T-maze).

## Build a model

For the common case — a discrete POMDP over a time window — one call:

```python
import numpy as np
from cffg.models.rollout import build_rollout

template, arrays = build_rollout(
    "my_world",
    steps=3,          # slices in the window
    past=0,           # slices 1..past are perception (data ports act{k}/obs{k})
    d=d,              # prior over the initial state          [K]
    A=A,              # observation model, column-stochastic   [M, K]
    c=c,              # goal prior over observations           [M]
    e=e,              # control prior                          [U]
    Bs=Bs,            # per-control transitions, col-stochastic [U, K, K]
    gfe=True,         # P-beads on future observations (curiosity on)
)
```

`cffg.models.tmaze` and `cffg.models.gridworld` are two worked instances of
this. When a joint state would make every controlled transition quadratic in
the full Cartesian cardinality, declare named blocks and only the products
the model actually uses:

```python
from cffg.models import (
    CouplingSpec, ObservationSpec, StateBlockSpec, build_factored_rollout,
)

template, arrays = build_factored_rollout(
    "position-and-mode",
    steps=3,
    blocks=(
        StateBlockSpec(
            "position", d_position, B_position_by_mode_and_action,
            selector="mode_parent",
            transition_learnable="position_transition",
        ),  # B shape [K_mode, U, K_pos, K_pos]
        StateBlockSpec(
            "mode", d_mode, B_mode,
            transition_learnable="mode_transition",
        ),  # B shape [U, K_mode, K_mode]
    ),
    couplings=(
        CouplingSpec("mode_parent", ("mode",)),
        CouplingSpec("joint_sensor", ("position", "mode")),
    ),
    actions=e,
    observation=ObservationSpec(
        "sensor", "joint_sensor", A_joint, c,
    ),
)
```

Component order is explicit and uses NumPy/C flattening (rightmost component
fastest). The builder serializes one mean-field `BeliefBinding` per block,
ties each declared parameter blob across time, and lowers products/fan-out to
existing deterministic `transition_mixture`/equality nodes. It does not add a
flat joint transition table. `build_factored_corridor` is the ready-made
position × mode example; pass `observation_matrix=` and
`observation_source="position" | "mode" | "state"` for identifying or
aliased sensors.

For other custom topologies, build the `Template` by hand (see
`tests/conftest.py` for a hand-built HMM slice) and grow big models from
small ones:

```python
from cffg.core.compose import unroll, compose
chain = unroll(slice_template, 10, {"tail": "head"})     # glue-port chaining
model = compose(prior_template, chain, {"attach": "t0/head"}, name="hmm10")
```

An uncertain categorical table can also live inside the graph. Bind the
likelihood's parameter role through `param_edges`, face that edge with a
`dirichlet` prior, and put a `MEAN_FIELD` bead on the likelihood. The carrier
cardinality is the flattened table size; the prior blob retains its canonical
shape:

```python
from cffg.core import ConstraintKind, ConstraintSpec, FactorSpec, VariableSpec

variables["A_edge"] = VariableSpec("A_edge", M * K)
factors["A_prior"] = FactorSpec(
    "A_prior", "dirichlet",
    edges={"out": "A_edge"},
    param_blobs={"counts": "alpha_A"},       # alpha_A.shape == (M, K), all > 0
)
factors["obs"] = FactorSpec(
    "obs", "matrix_cat",
    edges={"out": "x", "in": "z"},
    param_edges={"matrix": "A_edge"},
)
constraints.append(ConstraintSpec(ConstraintKind.MEAN_FIELD, "obs"))
```

The likelihood sends fractional state counts into the posterior and consumes
`E[log A]` in return. After inference,
`engine.dirichlet_marginals(state)["A_edge"]` reports `alpha`, `mean`,
`expected_log`, and `proper`. An unconstrained latent parameter edge is
rejected because exact sum-product is not Dirichlet-closed.

## Run inference

```python
from cffg.compiler import compile_template
from cffg.engine import Engine

engine = Engine(compile_template(template, arrays=arrays))
state  = engine.init_state()
state  = engine.bind_data(state, {"obs1": 4, "obs2": None})   # clamp / missing
result = engine.run(state, 200, tol=1e-12, damp=0.5)            # flood to stationarity
state  = result.require_converged()                             # raises on every other status

engine.marginals(state)["u1"]   # posteriors by variable name
engine.bfe(state)               # exact Bethe/Generalised free energy
```

To let future planning slices send the nonconjugate Part-II `mu3(A)` message
back toward a learned observation model, opt in when building the rollout and
prepare one reproducible particle set after installing the current proper
Dirichlet posterior:

```python
import jax

template, arrays = build_rollout(
    "planning_with_parameter_epistemics",
    steps=3, d=d, A=A, c=c, e=e, Bs=Bs,
    gfe=True, learned_a=True, mu3=True,
)
arrays["Ac"] = alpha_A                       # perception/VMP posterior, all > 0
engine = Engine(
    compile_template(template, arrays=arrays),
    mu3_samples=4096,
    mu3_min_ess_fraction=0.1,
)
state, next_key = engine.prepare_mu3(
    engine.init_state(), jax.random.PRNGKey(7)
)
result = engine.run(state, 200, tol=1e-8, damp=0.5)
result.require_converged()
engine.mu3_diagnostics(result.state)         # weighted moments, log Z, KL, ESS
```

Particles remain fixed during that run. Reusing the same key is bitwise
reproducible; use only `next_key` for a refresh. A changed `alpha_A` requires a
fresh `prepare_mu3` call. `Engine.score_mu3_epochs` refreshes particles between
policy-scoring epochs and averages only converged post-burn-in scores; burn-in
does not turn an unfinished or collapsed estimate into convergence.

A learned goal uses the conjugate Fig. 5 `mu1(c)` path. Future slices sharing
`Cc` add their predictive observation mass into one Dirichlet posterior:

```python
template, arrays = build_rollout(
    "learned_goal",
    steps=3, d=d, A=A, c=c, e=e, Bs=Bs,
    gfe=True, learned_c=True, c_counts=alpha_c,
)
engine = Engine(compile_template(template, arrays=arrays))
trial = engine.run(engine.init_state(), 200, tol=1e-8, damp=0.5)
trial_state = trial.require_converged()
next_prior = engine.goal_prior_updates(trial_state)  # {"Cc": alpha_c + Σ Abar zbar}

# Rollover is accepted only on a fresh state: sweeps are not new trials.
next_state = engine.with_goal_priors(engine.init_state(), next_prior)
```

`E[log c]` drives the learned-goal GFE coordinate. This generic predicted-mass
rollover is explicit; ordinary observed categorical learning remains the D6a
parameter-edge path. `RollingAgent(..., learn_goal=True,
initial_goal_counts=alpha_c)` stages the latest converged planning posterior
and commits it once at `reset_episode`; checkpoints preserve a pending
uncommitted trial exactly.

Rules of thumb: `damp=0.5` whenever the model has ≥ 2 GFE composites (their
outer coordination can limit-cycle undamped). `result.status` is one of
`converged`, `budget_exhausted`, `diverged`, or `infeasible`; its
`free_energies`, `message_deltas`, `iterations`, and `residual` explain that
classification. FE stationarity is the primary sum/GFE stopping signal and
message delta guards coherence. On tree graphs
`bfe` equals −log evidence exactly; with no data it is 0 for a normalized
model — both make good sanity checks. Parameters are *inputs* (`state.params`),
so you can swap them per run without recompiling.

`Engine.run` returned `(state, deltas)` before cff-j4mu. That tuple API is
intentionally not iterable now: use `result.state` when a fixed sweep budget is
deliberate, or `result.require_converged()` before acting on beliefs.

Schedules: the engine picks one automatically (cff-d56d). On an acyclic BP
model over discrete, Gaussian, or EF edges — chains, trees, state-space
models — `run` executes one exact inward/outward pass in O(E), certifies
the fixed point with a fused verification sweep, and returns `converged`
(even with `tol=None`; there is no sweep budget to exhaust). Loopy pure-BP
models iterate colored Gauss–Seidel sweeps; models with iterative
structures (mean-field beads, GFE composites, hard constraints, hybrid
nodes) run block-phased: each sweep updates those structures, then
refreshes the BP parts — tree-shaped parts exactly, loopy parts by one
colored sweep — before convergence is judged. Only δ form
constraints (and `damp > 0` on otherwise-exact models) still use plain
flooding; `engine.schedule.kind` and `.reason` say what you got. Force
flooding with `Engine(arena, schedule="flood")`.

## Run an agent

```python
import functools
from cffg.agent import RollingAgent
from cffg.models.gridworld import CorridorEnv, build_gridworld

env   = CorridorEnv(k=6, seed=1)
agent = RollingAgent(functools.partial(build_gridworld, k=6), horizon=3)
for _ in range(10):
    action, observation = agent.step(env.step)   # plan → act → filter belief

agent.checkpoint("agent.npz"); agent.restore("agent.npz")
```

Two compiled engines cover any episode length: a planning window from the
current beliefs and a one-slice filter that re-anchors them. `build_rollout`
serializes the action readout, DATA ports, prior target, posterior readout, and
episode learnable in `template.agent`; custom builders must declare the same
interface with their own model-local names. A one-block model retains the
`agent.belief` array. Multi-block models use `agent.beliefs[name]`, and their
joint versus mean-field rollover is explicit in each `BeliefBinding` rather
than inferred by the runtime.

Pass
`learn_a=True, initial_counts=...` for continual Dirichlet learning of the
observation model (see `cffg/experiments/tmaze_learning.py` for the full
perception–action–learning cycle reproducing Part II Fig. 9).

For behavioral calibration beyond the T-maze golden, the costly-cue experiment
reports the quantities needed to audit an epistemic decision directly:

```python
from cffg.experiments.behavioral_epistemics import run_benchmark

gfe = run_benchmark(gfe=True)
bfe = run_benchmark(gfe=False)
gfe.candidate_posterior_mi             # MI for left/right/cue/distractor
gfe.realized_information_gain_by_seed  # posterior KL after the chosen action
gfe.task_return_by_seed, gfe.regret_by_seed
gfe.convergence_statuses, gfe.free_energies
```

The cue is reward-relevant, the distractor has the same uniform observation
marginal but no context information, and the second action is replanned after
the first observation. This is exact observation epistemics; dynamics
curiosity's latent-next-state optimistic bound is a separate contract in
`docs/LIMITATIONS.md` S1c.

Many agents on one machine:

```python
from cffg.colony import Colony
colony = Colony()
ids = [colony.spawn(t, arrays_i) for arrays_i in per_instance_params]
colony.run(100, damp=0.5)      # one vmapped step-loop per unique structure
colony.marginals(ids[0])
```

Engines are cached by structure hash — N instances of one template cost one
compilation, and instances couple only through their ports.

## Edit, ship, budget

```python
from cffg.core.edits import apply_edits, check_edits, bundle_from_json, AddFactor
new_t   = apply_edits(t, actions)     # all-or-nothing; original untouched
problems = check_edits(t, actions)    # dry-run: [] means applicable
# actions round-trip through JSON (bundle_to_json / bundle_from_json)

from cffg.bundle import save_bundle, load_bundle
save_bundle("model.cffgz", template, arrays)   # zip: model + blobs + manifest
template, arrays, manifest = load_bundle("model.cffgz")  # hash-verified

from cffg.estimate import estimate
print(estimate(engine).summary())
# state 9.1 KiB (messages 2.8, params 6.4) · 2.76e+04 flops/sweep · 0.203 ms/sweep
```

## Register a custom factor

EF-arena factor types are executable registrations, not engine edits
(`cffg.kernels`, ARCHITECTURE 2e). Declare the structure, math, and
robustness answers in one `FactorKernel`; the compiler, VM, ledger, and
serialization dispatch from the registry:

```python
from cffg.core import FactorType
from cffg.kernels import FactorKernel, HostileInputs, register_kernel

register_kernel(FactorKernel(
    factor_type=FactorType("exponential_obs", edge_roles=("out", "rate"), family="ef"),
    impl_id="mypkg.exponential_obs", version="1.0.0",   # namespaced; serialized
    arena="ef", family="gamma", message_roles=("rate",),
    edge_families={"out": "exponential", "rate": "gamma"},
    message=my_message, energy=my_energy,               # jnp, trace-safe
    observed_out="exponential", validate_observation=my_validator,
    hostile=HostileInputs(...),                         # required, or registration is refused
))
```

Templates using the type record `(impl, version)` in their JSON/.cffgz;
loading without the kernel registered fails loudly and names what to
install. `tests/test_kernels.py` is the worked example, including the
independent evidence oracle every kernel is expected to ship with.

## Consume the engine from an experiment harness

Evaluation harnesses live *outside* this repo (sibling folders under
`../experiments/`, one uv project per experiment, results never committed
here). The patterns that matter, learned the measured way (2026-07-09):

```toml
# pyproject.toml of the harness
[tool.uv.sources]
cffg = { path = "../../cffg", editable = true }
```

- **Enable x64 once, first** (`jax.config.update("jax_enable_x64", True)`)
  — the ledger identities the tests pin assume it; f32 works but your
  tolerances won't match the docs.
- **Compile once, swap parameters through state.** Parameters are *inputs*
  to the jitted step (the colony contract): for trial loops, build the
  Engine once for the graph *shape*, then write `state.params[...]` (see
  `member_of` for per-factor discrete writes) and rebind observations. A
  loop that recompiles per trial is ~50× slower for nothing.
- **Sliding windows are the streaming pattern**: fixed window shape,
  compile once, thread the previous posterior in as the next window's
  prior params. Measured: flat steady-state memory over 10⁵-step streams
  (drift ≈ 0 after JIT warmup) at hundreds of filtered steps/sec.
- **Convergence is diameter-bound, not FLOP-bound**, on wide fan-in models:
  a two-level 784-leaf plate compiles in ~1 s and each sweep is cheap, but
  a linear equality chain needs ~1 sweep per leaf (cff-9xmq tracks the
  balanced-tree fix). Budget `run()` sweeps against your graph's diameter,
  and always pass `tol=` so coherent models exit early.
- **Read beliefs, not messages**: `marginals`, `gaussian_marginals`,
  `ef_marginals`, `node_beliefs`, `gaussian_node_stats` are the supported
  readouts; learning loops mirror `cffg.agent`'s count pattern (infer →
  accumulate stats → conjugate update → roll expected params back). Improper
  Gaussian marginals are NaN-marked; node statistics also expose
  `proper=False`, which learning loops must check before accumulation.
- **Report framework gaps in framework terms.** When a harness needs a
  numpy workaround, that sentence is a ticket (see cff-9xmq, cff-lquu,
  cff-5edo, cff-mjku, cff-gul8 — all minted from one evaluation round).

## Draw a model

```python
from cffg.viz import render_svg, save_svg, save_png

svg = render_svg(template)            # the papers' visual algebra, compressed
save_svg(template, "model.svg", level=0)   # every bead explicit (Part I Fig 14)
save_png(template, "model.png")            # needs: uv sync --extra viz
```

`level=0..4` selects how much of Part I's compression recipe (p.19) is
applied: 0 shows every bead and internal edge of the variational spec; 4
(default) is the compact form the papers use for real models — beads pushed
to node borders, dashed boxes around GFE composites, `⋯` where glue ports
continue the rollout. Layout is deterministic and *edit-stable*: growing a
model appends rows/columns without moving what was already drawn. For GUIs,
`build_scene(template)` returns the typed scene graph (every primitive
carries the `factor:`/`var:`/`port:` ref it depicts) instead of SVG text.

## Where things are

| | |
|---|---|
| `cffg/core/` | IR, validation, serialization, composition, edit actions (JAX-free) |
| `cffg/compiler.py` | template → message arena (groups, index tables, GFE composites) |
| `cffg/rules.py` | BP / VMP / GFE message updates + their free-energy terms |
| `cffg/engine.py` | jitted flooding step, damping, δ-hardening, marginals, BFE |
| `cffg/agent.py` `cffg/colony.py` | receding-horizon runtime; many-instance execution |
| `cffg/models/` `cffg/experiments/` | rollout builder, T-maze, gridworld, learning loop |
| `cffg/viz/` | paper-notation diagrams: derive → compress → layout → scene → SVG/PNG |
| `docs/ARCHITECTURE.md` | design + decisions · `docs/LIMITATIONS.md` — what this will never do |
| `docs/todo-notes/` | remaining threads, one note each, tied to `tk` tickets |

Working on the code? Read `AGENTS.md` first — uv-only, atomic commits,
tickets via `tk`, and the canonical-document policy are enforced, one of them
by a test.
