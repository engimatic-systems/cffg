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
uv run python tools/dashboard.py --run-suite  # short default suite + dashboard
uv run pytest -m slow  # opt-in long-running regressions
```

Run focused tests as often as useful while changing code. Pytest excludes
tests marked `slow` by default, so one passing short suite after the final
code change is sufficient to open or update a PR. The dashboard command above
runs pytest exactly once and renders that result. Before starting or waiting
on the short pass, inspect
`tools/test-suite-runtimes.json`: it records recent measured wall times and a
conservative `suggested_wait_seconds`.
Ticket- and documentation-only changes use their relevant validation instead
of the short suite. Run slow regressions explicitly when changing the behavior
they cover; they are not part of the default PR gate.

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

A recurrent switching model can make its regime transition depend on a
Gaussian drive without discretizing that drive. Add a `recurrence_link` whose
two categorical edges have the same `K >= 2` cardinality:

```python
factors["switch"] = FactorSpec(
    "switch", "recurrence_link",
    edges={"out": "z_next", "in": "z_prev", "drive": "x"},
    param_blobs={"weights": "switch_weights", "offsets": "switch_offsets"},
)
arrays["switch_weights"] = weights  # [K, K - 1, d], source/stick/drive
arrays["switch_offsets"] = offsets  # [K, K - 1]
```

For source regime `i`, stick `l` has logit
`weights[i,l] @ x + offsets[i,l]`; destination `j < K - 1` accepts stick `j`
after rejecting earlier sticks, and the last destination rejects them all.
Inference uses a structured source/destination pair belief. Current Gaussian
moments define a Pólya–Gamma/Jaakkola–Jordan expected-log transition table;
log-space sum-product sends both categorical messages, and the same pair gives
the Gaussian-drive message. On an acyclic ordinary categorical recurrence
component this is exact inference for the bounded table. A categorical loop
uses local Bethe beliefs and carries no exact-chain claim. The reported VFE is
a documented upper bound on the exact-transition VFE at the same Gaussian
belief. Exact regime zeros stay exact, while an improper Gaussian drive is
divergence. `engine.arena.recurrence_components` records the derived component
membership and `acyclic` classification.

Hybrid learning and forecasting use factor-local readouts rather than products
of endpoint marginals:

```python
stats = engine.gaussian_node_stats(state)
transition = stats["transition"]
transition["responsibility"]   # [K], soft regime belief
transition["Syy"]              # [K, dy, dy], already responsibility-weighted
transition["Syx"]              # [K, dy, dx + 1], offset column included
transition["Sxx"]              # [K, dx + 1, dx + 1]

forecast = engine.gaussian_mixture_predictives(state)["transition"]
forecast["component_mean"]       # [K, dy]
forecast["component_covariance"] # [K, dy, dy]
forecast["mean"], forecast["covariance"]
```

`gaussian_mixture_predictives` is a one-step factor-to-output cavity
predictive, not the posterior output marginal: it uses the selector and input
messages but excludes downstream evidence facing `out`. Prior-mixture
statistics name their arrays `weighted_mean` and `weighted_second` to make the
weighting explicit. `recurrence_pair_beliefs` exposes a `recurrence_link`'s
normalized `[source, destination]` pair, its two marginals, literal support,
direct Gaussian drive moments, and the at-risk/signed-success masses shared by
inference and learning. `gaussian_node_stats` includes that same record, with
`in_probability`/`out_probability` compatibility aliases. The pair is derived
from current factor cavities; it is not independently persisted.
Every hybrid record has `proper`; consumers must reject a false value and its
shape-preserving NaNs.

The between-trial recurrence update is deliberately short and remains outside
JIT, like NIW/MNIW learning:

```python
from cffg.conj import vague_pg_weights

posterior = vague_pg_weights(regimes=K, dimension=d)
stats = engine.recurrence_weight_stats(state)["switch"]
if not stats["proper"]:
    raise ValueError("improper recurrence belief")
posterior = posterior.update(stats["linear"], stats["quadratic"])
weights, offsets = posterior.rollover()  # next trial's parameter arrays
```

`linear` and `quadratic` use the same structured pair and coordinate-optimal PG
mean as the Gaussian drive message. Exact-zero pair exposure contributes exact
zero natural statistics, so an unexposed source/stick retains its prior. The
rollover is an empirical-Bayes point estimate; it does not claim to carry
weight uncertainty inside inference.

### Recurrent switching state-space models

The model helper assembles a complete rSLDS with state-dependent regime
transitions, per-regime linear dynamics, and shared linear-Gaussian emissions.
It returns an ordinary serializable `Template`; parameters remain arrays and
can be replaced between runs without recompilation:

```python
from cffg.compiler import compile_template
from cffg.engine import Engine
from cffg.models import rslds, vague_rslds_learning

template, arrays = rslds(
    n_steps=T,
    dynamics_matrix=A,                 # [K, d, d]
    dynamics_offset=b,                 # [K, d]
    dynamics_covariance=Q,             # [K, d, d]
    recurrence_weights=Rz,             # [K, K - 1, d]
    recurrence_offsets=rz,             # [K, K - 1]
    emission_matrix=H,                 # [dy, d]
    emission_covariance=V,             # [dy, dy]
    initial_state_mean=m0,             # [d]
    initial_state_covariance=P0,        # [d, d]
    initial_regime_probability=pi,     # [K]
)
engine = Engine(compile_template(template, arrays=arrays))
learning = vague_rslds_learning(K, d)

state = engine.init_state()  # first trial uses the arrays supplied above
state = engine.bind_data(
    state, {f"y{t}": observation for t, observation in enumerate(ys)}
)
result = engine.run(state, 200, tol=1e-7)
result.require_converged()
learning = learning.update(engine, result)

# A later trial reuses the compiled engine with learned point parameters.
next_state = learning.install(engine, engine.init_state())
```

The template declares all four fitted families: the initial-regime
Dirichlet, component-MNIW dynamics, PG recurrence block, and tied emission
MNIW. `RSLDSLearning` is the compatibility facade for the historical
distinct-trial API: `install` accepts a fresh state and writes the current
initial-regime, dynamics, and recurrence point parameters, while `update`
accumulates a status-bearing result into those three posteriors. Its generic
checkpoint is available through `state_dict(engine)` and
`load_state_dict(engine, data)`. The facade deliberately leaves emission
fixed; use `Learnables` directly when a distinct-trial application also
learns emission. The recurrence offsets
already represent the ordinary source-specific transition baseline; multiplying
a separate `matrix_cat` into the same transition would require a new normalized
factor and free-energy derivation.

Recurrence checkpoints include a scalar manifest naming
`cffg.recurrence.d9-structured-pair/1.0.0`, the contributing factor sources,
parameter blobs, hyperparameter dtypes, and shapes. A pre-manifest `v2:` or
slash-key checkpoint may contain D8 singleton-product statistics and is
therefore rejected for recurrence, even though its array keys and shapes still
fit. `vague_rslds_learning` and direct `RSLDSLearning(...)` construction define
fresh priors; `update` and `load_state_dict` return objects explicitly marked
with the D9 identity. To carry an old posterior forward as a prior, construct a
new prior deliberately and rerun inference; do not relabel the old checkpoint
as a D9 posterior. That is a new D9 run, not checkpoint resume; CFFG provides
no D8-posterior migration. Per-array digests bind the manifest to the exact
persisted values.

For one-sequence observed-only fitting, use the host-side coordinate loop. It
chooses latent coordinates from the observations, compiles one engine, and
estimates every rSLDS parameter family without accepting generating states,
regimes, or parameters:

```python
import jax

from cffg.models import (
    PCAARHMMInitializer,
    fit_rslds,
    vague_rslds_fit_priors,
)

fit = fit_rslds(
    observations,                         # [T, dy], floating and finite
    regimes=K,
    latent_dimension=d,
    initializer=PCAARHMMInitializer(),
    priors=vague_rslds_fit_priors(K, d, observations.shape[1]),
    max_outer_iterations=20,
    max_inner_iterations=500,
    inner_tolerance=1e-5,
    key=jax.random.key(0),
)

if fit.status.value == "diverged":
    raise RuntimeError(fit.error)

# The fitter certifies the state once after its terminal parameter write.
final = fit.last_inference.require_converged()
```

`PCAARHMMInitializer` uses PCA for a reproducible observation-derived latent
coordinate system, clusters consecutive latent pairs, and fits local
autoregressions. The explicit key selects cluster centers. Each outer
coordinate reruns hidden-variable inference and recomputes the initial-regime,
per-regime dynamics, recurrence, and shared-emission posteriors from the same
immutable priors. One `Learnables.replace` validates all four candidate
posteriors before committing them, and one atomic write installs every point
value. The shared emission is the ordinary tied `gaussian_linear` MNIW
learnable, so every time slice contributes once. The continuous
initial-state mean and covariance are fixed initializer outputs, not fitted
posteriors.

This replacement cadence is different from `RSLDSLearning.update`: `update`
accumulates evidence from a new independent trial, while `fit_rslds` replaces
one sequence's coordinates and never counts that sequence again as new data.
The fit result reports `converged`, `budget_exhausted`, or `diverged`, preserves
the terminal inner-engine status, and keeps the inner Bethe free-energy trace
separate from the scale-normalized outer fixed-point residual. That residual
compares the normalized initial-regime potential and every dynamics,
recurrence, and emission value installed into the next solve. The historical
posterior/reporting snapshot and a truth-free transition-function residual
remain available in `fit.phases` as diagnostics. After a finite
inner budget exhaustion, the default two retries restart from the same fresh
state and double the convex update fraction while preserving the fixed point;
`max_inner_retries=0` retains one fixed-damping attempt. `fit.phases` records
each damping, status, recurrence status, energy and message-delta tail,
Gaussian propriety check, and parameter scale. The default outer tolerance is
dtype-scaled and must hold twice consecutively. A finite outer iteration limit
may therefore return useful fitted parameters with `budget_exhausted`; it is
not silently relabeled as convergence.

Functional `mu3` engines are unsupported by this fitter and are rejected
before inference because the retry policy has no explicit key and fixed-sample
reuse contract. Fit coordination remains rSLDS-owned until another production
model proves that its retry, runtime-certificate, and terminal-solve policy has the same
interface; only the declared-learning transaction is currently shared.

The original canonical NASCAR check supplied the generating parameters and,
at `T=10,000`, converged in 248 sweeps with `0.001264` latent-state RMSE and
`99.95%` MAP regime accuracy. The preregistered observed-only comparison then
fit 8,000 of the same 10,000 generated observations under one disjoint seed:

| Fitted arm | Scalars | Held-out mean log likelihood | Aligned latent RMSE | Aligned regime accuracy | 10,000-step periodicity |
| --- | ---: | ---: | ---: | ---: | ---: |
| rSLDS, K=4 | 160 | 30.550824 | 0.001583 | 86.325% | 0.517638, bounded |
| SLDS, K=4 | 136 | 30.237528 | 0.001559 | 88.8875% | 0, unbounded |
| SLDS, K=5 | 154 | 31.762280 | 0.002027 | 99.6375% | 0, unbounded |

All arms preserved `budget_exhausted` after the registered two outer
coordinates, with converged inner and final inference. Exactly one registered
claim passed: rSLDS held-out prediction exceeded the same-`K` SLDS by
`0.313296`, above the `0.005` margin. The latent- and regime-recovery margins
missed, so no recovery advantage is claimed; no claim was registered against
the K=5 budget-matched arm. The bounded rSLDS rollout is reported as an
unregistered observation, not promoted to an advantage claim. Exact artifacts
live in `../experiments/cffg-nascar-validation` at experiment commit `f4e7b0f`
with results SHA-256 `99d02c92315357e1d778146c7b7ca74cd20aa2390f4ad165ed17515fba29548e`.

#### Two-family rSLDS evidence audit

The parity target has two independent parts. Probabilistic-architecture parity
asks whether CFFG's factor types and graph composition represent the classic
rSLDS distributions and dependencies. Task parity asks whether the fitted CFFG
model performs roughly as well as the independent paper-equation adapter on
the classic NASCAR task.

The architecture half is established for the source-homogeneous,
shared-emission NASCAR subclass. Executable cross-mapping oracles cover the
initial categorical regime, tied initial Gaussian, source-indexed
decision-list recurrence, destination-indexed linear-Gaussian dynamics, and
shared linear-Gaussian emission. Both fit interfaces receive observations and
declared priors, but not generating states, regimes, or parameters. This is a
probability-model correspondence; CFFG uses variational Pólya--Gamma/
Jaakkola--Jordan inference and the adapter uses Pólya--Gamma Gibbs sampling.

The immutable 3+3 final run did not establish either empirical claim:

| Evidence layer | Final result |
| --- | --- |
| Classic task parity | `not_ready`: all six CFFG tiers converged, but all six adapter tiers failed the registered mixing gate, so no paired effects or formal absolute decisions were issued. |
| Heterogeneous source recovery | `not_ready`: one seed failed the prefit source/gate outcome rule, another exhausted both CFFG outer budgets, and every fitted adapter tier failed mixing. |
| Joint claim | Not registered; the two family decisions are independent. |

The classic family used 10,000 training steps plus a separate 2,500-step
held-out extension. It is therefore paper-scale training, not an exact replay
of Figure 3. The local heterogeneous extension used 1,280 training and 320
held-out steps with different gates for every source regime; it is not part of
the original NASCAR demonstration.

Available CFFG-only values are descriptive. All six classic held-out
on-support transition errors were below their registered ceilings, but native
seed 2001 periodicity was `0.4058` against a `0.4110` floor. On the one
converged heterogeneous seed, shared/native worst-source transition errors were
`0.1946`/`0.3761` and source-contrast errors were `0.2498`/`0.3736`, all above
the `0.150` ceilings. These values do not override the frozen `not_ready`
decisions.

The complete local experiment artifact is commit `6c0437a`; `results.json` is
SHA-256 `eaf4bdccb6b2d9ecf35f59a91fa1012873bba697c8fe499781b28c1f787697c4`
and its manifest is
`b71c41ad46cd2e210a9e204743d8fd58f1a76a69305afae8601c7e307b64a56a`.
The result supports neither authors-code identity, inference-algorithm
identity, nor universal rSLDS-family performance. cff-l5ag, cff-q1au,
cff-qulu, cff-0oqo, and cff-3v7q own the observed evidence gaps.

Cold setup was a separate host-grounding concern rather than an rSLDS rule or
JAX-kernel defect. Linear incidence indexes removed the per-variable template
rescans. In three fresh CPU/x64 workers per path at `T=10,000`, median setup
fell from `741.361` to `24.060` seconds (`30.81×`), while median converged
inference changed from `27.342` to `27.020` seconds. Messages, exact support,
readouts, BFE, and statuses were identical at `T=100`, `1,000`, and `10,000`;
indexed peak RSS was `1.948` GiB. The reproducible command, every raw phase
sample, profiles, environment, and hashed arrays live in
`../experiments/cffg-nascar-validation` at commit `1b1b714`.

## Write named parameters and learn Gaussian factors

Runtime parameter arrays are grouped by factor family and shape. Use factor
names and serialized parameter roles instead of editing those stacked arrays by
position:

```python
from cffg import set_params

params = set_params(
    engine,
    engine.init_state().params,
    {
        "emission0": {
            "matrix": H,
            "offset": h,
            "cov": R,
        }
    },
)
```

`set_params` validates every requested value before changing its returned
mapping. Probability roles keep literal zero as the runtime log-zero sentinel;
covariances must be finite, symmetric, and positive definite and become the
runtime precision. A write to one blob/role reaches every factor tied to that
same serialized blob. Unequal requests through two tied factors are rejected.
Supported named-write factors are `categorical`, `matrix_cat`,
`transition_mixture`, `gaussian`, `gaussian_linear`, `gaussian_mixture`,
`gaussian_mixture_linear`, `recurrence_link`, and `gaussian_prec`.

Declared learning supports NIW for `gaussian`, MNIW for `gaussian_linear`, a
component MNIW for `gaussian_mixture_linear`, and Pólya–Gamma Gaussian weights
for `recurrence_link`. An observed Gaussian-linear emission uses the same MNIW
pairing after compilation to its conditioned runtime rule:

```python
from cffg import conj
from cffg.core import LearnableSpec
from cffg.learn import Learnables

template.learnables["emission"] = LearnableSpec(
    "emission",
    "mniw",
    ("emission_matrix", "emission_offset", "emission_covariance"),
)
template.learnables["dynamics"] = LearnableSpec(
    "dynamics",
    "component_mniw",
    ("dynamics_matrix", "dynamics_offset", "dynamics_covariance"),
)
template.learnables["recurrence"] = LearnableSpec(
    "recurrence",
    "pg_weights",
    ("recurrence_weights", "recurrence_offsets"),
)
engine = Engine(compile_template(template, arrays=arrays))
learnables = Learnables(
    engine,
    priors={
        "emission": conj.vague_mniw(observation_dim, state_dim),
        "dynamics": tuple(
            conj.vague_mniw(state_dim, state_dim) for _ in range(regimes)
        ),
        "recurrence": conj.vague_pg_weights(regimes, state_dim),
    },
)

result = engine.run(bound_state, 200, tol=1e-7)
learnables.replace(engine, result)       # same sequence: reset from fixed prior
next_params = learnables.write(engine, engine.init_state().params)
```

Pass the status-bearing `RunResult`; learning refuses an unfinished or
diverged run. `replace` is idempotent for unchanged one-sequence statistics.
Use `accumulate` instead when the result is a new independent trial whose
evidence should be added to the current posterior. Both operations pool all
tied source factors and commit all selected posteriors atomically. Reporting
uses `learnables.expected(name, role)`; inference rollover writes
`psi / nu`, the inverse expected precision, rather than the posterior-mean
covariance. `component_mniw` stores `m`, `lam`, `psi`, and `nu` with a leading
selector-component axis; exact zero responsibility adds exact zero statistics
and leaves that component at its prior. `pg_weights` stores `eta` and
`precision` with source-regime, stick, and augmented `[drive; offset]` axes,
then rolls posterior means into separate weights and offsets. Both families
remain host-side empirical-Bayes updates and reject improper or non-finite
statistics before mutation. Versioned checkpoints preserve every keyed
hyperparameter. Legacy multi-array checkpoints for unchanged families must
name a complete posterior. Recurrence is stricter: its current checkpoint must
also carry the structured-pair statistic manifest; unidentified D8/D9 state is
never guessed from compatible array shapes.

## Run inference

```python
from cffg.compiler import compile_template
from cffg.engine import Engine

engine = Engine(compile_template(template, arrays=arrays))
state  = engine.init_state()
state  = engine.bind_data(state, {"obs1": 4, "obs2": None})   # clamp / missing
result = engine.run(state, 200, tol=1e-12)                      # solve to stationarity
state  = result.require_converged()                             # raises on every other status

engine.marginals(state)["u1"]   # posteriors by variable name
engine.bfe(state)               # exact Bethe/Generalised free energy
```

For hybrid regime epistemics, place a `P_SUBSTITUTION` constraint with
`substituted_role="out"` on a `gaussian_mixture` or
`gaussian_mixture_linear` factor whose output faces a fixed `gaussian` goal.
The compiler derives one D5 composite; Gaussian state edges keep their D3
messages while the selector solves the regime-information coordinate:

```python
result = engine.run(engine.init_state(), 200, tol=1e-8)
state = result.require_converged()
terms = engine.hybrid_gfe_diagnostics(state)["emission"]
terms["rho"]                       # stationary regime belief
terms["mi_regime_surrogate"]       # scored collapse surrogate
terms["mi_regime_exact_lower"]     # diagnostic sandwich
terms["mi_regime_exact_upper"]     # includes the H(rho) cap
terms["bound_direction"]           # reported_gfe_lower_bounds_exact
```

The direct ledger term is the optimistic D5 `G-hat`, never its stationary
Lagrangian. Exact selector zeros remain exact; improper state, emission, or
goal covariances cannot return a converged result.

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
result = engine.run(state, 200, tol=1e-8)
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
trial = engine.run(engine.init_state(), 200, tol=1e-8)
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

Plans containing GFE composites use the fixed-support trust-region outer
solver automatically. Caller-tuned `damp` is rejected on that path; use
`outer_solver="picard"` only to reproduce or diagnose the historical
fixed-point iteration. `result.status` is one of `converged`,
`budget_exhausted`, `diverged`, or `infeasible`. Its `outer_residual`,
`outer_merit`, `outer_gradient_norm`, `trust_radius`, accepted/rejected step
counts, `outer_map_evaluations`, `trial_evaluations`, `linear_iterations`,
`solver_compilations`, and `termination_reason` explain a GFE classification;
the merit never enters `engine.bfe` or a policy score. Fixed chart/support,
dtype, and hardening phases reuse derivative kernels while observations and
parameters remain dynamic inputs. Convergence requires a coupled outer root,
successful local solves, coherent exact support, and a finite stationary FE
ledger. On tree graphs
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
nodes, recurrence links) run block-phased: each sweep updates those
structures, then refreshes the BP parts — tree-shaped parts exactly, loopy
parts by one colored sweep — before convergence is judged. Only δ form
constraints (and `damp > 0` on otherwise-exact models) still use plain
flooding; `engine.schedule.kind` and `.reason` say what you got. Force
flooding with `Engine(arena, schedule="flood")`.

Recurrence links remain time-local factors in both paths. Each recurrence
categorical pass is `O(T*K^2)` work with `O(T*K^2)` transient pair storage for
`T` equal-cardinality links, while factor messages and the BFE keep local pair
ownership. The current mixed scheduler does not fuse all recurrence links
into a monolithic chain region; fusion may reduce the number of outer passes
but cannot change the fixed point or counting numbers.

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
colony.run(100)                # grouped by structure
colony.marginals(ids[0])
```

Engines are cached by structure hash — N instances of one template cost one
compilation, and instances couple only through their ports. Ordinary iterative
plans use vmapped rule iterations with host status tracking. Mixed-GFE plans
reuse the compiled matrix-free kernels but currently run one trust controller
per instance so radii, active supports, and termination reasons remain
independent.

## Edit, ship, budget

```python
from cffg.core.edits import apply_edits, check_edits, bundle_from_json, AddFactor
new_t   = apply_edits(t, actions)     # all-or-nothing; original untouched
problems = check_edits(t, actions)    # dry-run: [] means applicable
# actions round-trip through JSON (bundle_to_json / bundle_from_json)

from cffg.bundle import save_bundle, load_bundle
save_bundle("model.cffgz", template, arrays)   # zip: model + blobs + manifest
template, arrays, manifest = load_bundle("model.cffgz")  # hash-verified

# BlobRef paths use .npy or .safetensors; .cffgz stores .npy members.
# RollingAgent checkpoints use .npz for many small runtime arrays.

from cffg.estimate import estimate
print(estimate(engine).summary())
# state 9.1 KiB (messages 2.8, params 6.4) · 2.76e+04 flops/sweep · 0.203 ms/sweep
```

A `.cffgz` bundle is model-only: it contains the template and point-parameter
blobs, not a learned posterior or resumable inference state. Existing bundles
therefore remain loadable and execute with the current documented recurrence
semantics. The recurrence statistic manifest belongs only to checkpoints that
can resume learning.

`.npz` is a ZIP container and does not support ordinary per-array memory
mapping, so it is reserved for agent checkpoints rather than parameter blobs.

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
  Engine once for the graph *shape*, then use `set_params` or
  `Learnables.write` and rebind observations. `Arena.member_of` is the
  derived role-bearing address map when lower-level inspection is needed. A
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
  `ef_marginals`, `node_beliefs`, `gaussian_node_stats`, and
  the specialized `gaussian_mixture_predictives` and
  `recurrence_weight_stats` are the supported readouts; learning loops mirror
  `cffg.agent`'s count pattern (infer → accumulate stats → conjugate update →
  roll expected params back). Improper Gaussian marginals are NaN-marked;
  node statistics and specialized readouts also expose `proper=False`, which
  consumers must check before accumulation or use.
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
