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
uv run pytest    # 105 tests; the T-maze + gridworld suites are living examples
```

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
this. For custom topologies, build the `Template` by hand (see
`tests/conftest.py` for a hand-built HMM slice) and grow big models from
small ones:

```python
from cffg.core.compose import unroll, compose
chain = unroll(slice_template, 10, {"tail": "head"})     # glue-port chaining
model = compose(prior_template, chain, {"attach": "t0/head"}, name="hmm10")
```

## Run inference

```python
from cffg.compiler import compile_template
from cffg.engine import Engine

engine = Engine(compile_template(template, arrays=arrays))
state  = engine.init_state()
state  = engine.bind_data(state, {"obs1": 4, "obs2": None})   # clamp / missing
state, deltas = engine.run(state, 200, tol=1e-12, damp=0.5)   # flood to coherence

engine.marginals(state)["u1"]   # posteriors by variable name
engine.bfe(state)               # exact Bethe/Generalised free energy
```

Rules of thumb: `damp=0.5` whenever the model has ≥ 2 GFE composites (their
outer coordination can limit-cycle undamped); `deltas` is the coherence
signal — trees converge hard, so treat a plateau as a smell. On tree graphs
`bfe` equals −log evidence exactly; with no data it is 0 for a normalized
model — both make good sanity checks. Parameters are *inputs* (`state.params`),
so you can swap them per run without recompiling.

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
current belief and a one-slice filter that re-anchors it. Pass
`learn_a=True, initial_counts=...` for continual Dirichlet learning of the
observation model (see `cffg/experiments/tmaze_learning.py` for the full
perception–action–learning cycle reproducing Part II Fig. 9).

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
tickets via `tk`, and the four-canonical-docs rule are enforced, one of them
by a test.
