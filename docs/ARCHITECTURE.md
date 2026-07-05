# CFFG Engine — Architecture Plan

Status: proposal, 2026-07-05. Sources: Koudahl et al. Part I (arXiv:2306.08014),
van de Laar et al. Part II, project brief in Obsidian `agents/CFFG/0.0.md`.

## What the papers give us (and what they don't)

The papers provide a complete recipe for a specific, powerful thing:

- **CFFG notation**: a factor graph (nodes = factors, edges = variables) *plus
  constraint annotations* ("beads") that fully specify the variational optimization
  problem — factorization constraints, form constraints (δ/pointmass, etc.),
  moment-matching constraints, and **P-substitution** (replace `q(x)` with
  `p(x|z,θ)` in a node-local expectation, turning that node's local VFE into a local
  **Generalised Free Energy** with an epistemic mutual-information term).
- **Hybrid message passing**: constraints are node/edge-local, so different
  algorithms (BP, VMP, EP, GFE-messages) coexist on one graph. Stationary points of
  the constrained Bethe Lagrangian = fixed points of message passing.
- **Closed-form GFE messages for the discrete goal-observation submodel**
  (Cat/Dirichlet): Part II Fig. 5 gives the message table. Two are nontrivial:
  the backward state message is found by solving `z̄ = σ(ρ(z̄) + log d)` with
  Newton's method (the naive fixed-point update can diverge — Part II §4.5), and
  the message to the observation matrix A is a non-standard log-message evaluated
  by importance sampling (making GFE evaluation stochastic; average after burn-in).
- **A validation target**: T-maze results (policy posteriors in Part I Fig. 28,
  learning curves in Part II §7) plus the biaslab/LAIF Julia reference code.

What they do *not* give: GFE messages for arbitrary node types. Each new
factor-type × constraint combination needs a derived update rule. So the engine
must be organized around an **extensible rule registry**, not a universal solver.

## Layered design

```
┌─────────────────────────────────────────────────────────┐
│ 5. Agent runtime   perception-action loop, rolling      │
│                    horizon, continual learning, ckpts   │
├─────────────────────────────────────────────────────────┤
│ 4. Inference VM    schedules, damping, FE accounting,   │
│                    convergence monitoring               │
├─────────────────────────────────────────────────────────┤
│ 3. Rule library    per (factor-type, constraint-config) │
│                    message updates: BP / VMP / GFE      │
├─────────────────────────────────────────────────────────┤
│ 2. Compiler        template → grounded graph → arenas   │
│                    (PGMax-style) → jitted step function │
├─────────────────────────────────────────────────────────┤
│ 1. Core IR         typed CFFG graph + constraints +     │
│                    ports; serialization; edit actions   │
└─────────────────────────────────────────────────────────┘
```

### 1. Core IR and serialization (`cffg.core`)

A pure-Python (dataclass) representation, independent of JAX:

- `VariableSpec` (domain: categorical cardinality now, continuous later),
  `FactorSpec` (type + parameter refs), `ConstraintSpec` (bead types:
  normalization, marginalization, form/δ, moment-matching, P-substitution set 𝒫),
  `CompositeNode` (Markov-blanket grouping, e.g. the goal-observation pair).
- **Templates with ports.** A template is a CFFG fragment whose boundary edges are
  declared **ports** with roles: `data` (sensor clamp), `goal` (goal prior),
  `prior` (parameter prior), `glue` (connects to the next unrolled slice). A model
  is valid iff every port is terminated by data, a prior/goal, or a glue bond.
  This is the single mechanism behind unrolling, rollouts, "no dangling edges,"
  and composing templates — one invariant instead of four features.
- **Serialization**: versioned JSON (or msgpack) schema for structure and
  constraints; all array data (A matrices, Dirichlet counts, goal statistics)
  referenced as relative file paths to `safetensors`/`.npz` blobs. Content hash
  of the structural part keys the JIT cache.
- **Portable bundle**: the deployable unit is a ZIP (`.cffgz`) containing
  `model.json` (structure + constraints), `blobs/` (parameter arrays), and
  `manifest.json` (schema version, content hashes, provenance). Unzipping and
  pointing the engine at `model.json` is a full restore; checkpointing writes
  the same bundle. No third-party interchange format — this schema is canonical.
- **Edit actions**: the only public mutation API is transactional bundles
  (`add_slice`, `bind_data`, `set_constraint`, `compose(template, port_map)`) that
  validate the port invariant before committing. This is the GUI-ready surface —
  a GUI later is just a client of this API, and edit actions double as an
  incremental (append-only) serialization of model history.

### 2. Compiler (`cffg.compile`)

Two-phase, mirroring PGMax:

1. **Grounding**: template + horizon + data bindings → flat grounded graph.
   Unrolling = repeated self-composition along glue ports. Time-dependent
   constraints applied here: past observation edges get δ (data) constraints,
   future observation edges get P-substitution + goal prior — this is exactly how
   the papers unify perception and planning in one model.
2. **Arena layout**: group factors and edges by (type, shape) into dense arrays;
   messages live in one flat log-space tensor per direction with gather/scatter
   index maps. One MP sweep = a few vectorized segment-ops per factor group, and
   the whole iteration loop is a `jax.lax.scan` inside `jax.jit`. Structure edits
   trigger recompile; data/parameter edits are just new arena inputs (no
   recompile). Static shapes throughout.

### 3. Message rule library (`cffg.rules`)

A registry keyed by `(factor_type, constraint_config, out_edge_role)`:

- **Standard rules**: sum-product (BP) and VMP for `Cat`, `Dir`, `TransitionMixture`
  (the TM node from Part I Appendix C), equality nodes, δ.
- **GFE rules** for the P-substituted goal-observation composite (Part II Fig. 5):
  Dirichlet message to goal statistics, softmax/Newton backward state message
  (Corollary 1 form, 20-ish Newton steps as in the paper), importance-sampled
  log-message to A. All in log-space; every stochastic rule takes an explicit PRNG
  key.
- Each rule ships with a **local free-energy contribution** function (average
  energy + entropy terms with correct edge-entropy overcounting), so global
  BFE/GFE is always exactly the sum of registered local terms — the "true BFE, no
  shortcuts" requirement is enforced structurally.

### 4. Inference VM (`cffg.infer`)

- Schedules: flooding (default) and sequential; damping as a knob. The brief is
  right that scheduling shouldn't be overthought — flooding + iterate-to-coherence
  covers the discrete case — but "coherence" needs a number: we track per-node FE
  and message deltas each sweep and stop on tolerance or budget.
- GFE nodes are stochastic (importance sampling), so FE traces get burn-in
  averaging (paper uses ~10 iterations) before any decision is read off.
- Inference *modes* are just constraint presets on the same grounded model:
  filtering/smoothing (data on past), planning (goals on future, P-substitution
  on), what-if (clamp hypothetical δs), backward inference (goals only, no data).

### 5. Agent runtime (`cffg.agent`)

- **Receding-horizon loop**: at each tick, bind new observation (future edge's
  constraint flips from goal to data δ), glue one new slice at the horizon,
  re-infer, read the control marginal (or its MAP under a δ-constraint), emit
  action. This directly fixes the hard-horizon problem the papers flag with the
  original GFE formulation.
- **Continual learning**: conjugate posteriors roll into next-trial priors
  (`q_s(A) → p_{s+1}(A)` Dirichlet count accumulation, same for goal statistics
  c). Checkpoint = serialize current parameter blobs + template ref; can happen
  at any tick.
- Sensor/actuator adapters are plain Python callbacks mapping env observations to
  one-hot data bindings and control marginals to env actions.

### 5b. Colony execution (many small graphs, one machine)

A first-class use case: ~10²–10³ small CFFG instances running concurrently as
adaptive components ("learning goo") inside a larger deterministic host system.
This costs three design constraints, adopted from M1 onward, and one extra layer —
not an architecture change:

- **Stateless engine, state as pytree.** An instance is pure data:
  `(structure_hash, params, messages, rng_key)`. The compiled step is a pure
  function `state → state`. No singletons, no Python object holding arrays.
  (JAX pushes us here anyway; we make it a contract.)
- **Batch axis by construction.** The compiled step must be `vmap`-compatible
  (no data-dependent Python control flow; Newton loops as `lax` primitives with
  fixed iteration caps). Then N instances *sharing a structure* execute as one
  batched XLA kernel — a colony of clones with different parameters, data, and
  RNG keys is barely more expensive than one instance.
- **Batching by structure hash.** The JIT cache is already keyed on structure;
  the colony scheduler groups instances by that key and runs one vmapped call
  per group per tick. A hundred goo blobs = (few unique templates) compiled
  kernels, not a hundred compilations.
- **`Colony` layer** above the agent runtime: owns the instance table, groups by
  structure, ticks groups (sync round-robin first; JAX's async dispatch already
  overlaps host code and XLA), routes host-system signals to instance data/goal
  ports and actions back. Inter-instance coupling happens **through the host via
  ports** (the Part II buyer/seller pattern: nested Markov blankets, exchanging
  observations/actions), never by merging graphs at runtime.

Known cost: per-instance *structure* evolution fragments the batching (each
divergent structure = its own compile). The goo pattern should draw structures
from a small template library and evolve parameters freely; structure variants
get promoted to the library deliberately, not mutated ad hoc. Memory is a
non-issue at this scale (a T-maze-sized instance's arenas are kilobytes).

## Test suite (three tiers, per the brief)

1. **Undirected / CSP**: small Ising or Potts grids and a mini-Sudoku; check BP
   marginals against exhaustive enumeration, and PGMax parity where applicable.
2. **Sequential prediction**: HMM with known parameters; engine's smoothed
   marginals must match exact forward-backward to numerical tolerance; then
   Dirichlet learning of the transition matrix converging to the true one.
3. **Epistemic action**: the T-maze, three variants from the papers —
   (a) known A, direct policy inference: reproduce Part I Fig. 28 posteriors
   (≈[.20 .30 .30 .20] then [.25 .13 .35 .26]) and the cue-first behavior;
   (b) learning A over trials: win-rate curve shape vs Part II Fig. 9, GFE agent
   beats BFE-ablation agent (flip one constraint bead — great ablation test);
   (c) δ-constrained MAP policy variant.
   Golden numbers cross-checked against biaslab/LAIF via a sub-agent.

Property tests throughout: FE never counted twice (edge entropy correction),
messages normalized, VMP monotonically decreases BFE on tree graphs, BP exact on
trees.

## Resource estimator

Because shapes are static post-compile, memory is computable analytically from
arena sizes, and `jax.jit(...).lower(...).compile().cost_analysis()` gives
FLOPs/bytes per MP sweep. Estimator = analytic memory + cost-analysis FLOPs +
a small calibration run (fit sweep-time curve on 3–4 model sizes per backend).
Cheap to build once the compiler exists; don't build it before.

## Dependency policy (JAX ecosystem)

Adopt (well-established, matched to our actual needs):

- **`jax`** — the engine substrate.
- **`equinox`** — pytree containers for engine/instance state. `eqx.Module` gives
  typed dataclass-pytrees, and `filter_jit`/`filter_vmap` cleanly separate static
  fields (structure hash, shapes, schedule) from dynamic arrays (messages,
  parameters) — exactly our compile boundary. Used for *state containers and
  compiled callables*, not neural networks.
- **`jaxtyping`** — shape/dtype annotations on message-rule signatures, runtime
  checked in tests. Message-passing bugs are shape bugs; catch them at the rule
  boundary.
- **`chex`** — test assertions (pytree/shape equality, jit-vs-nojit variants).
- **`safetensors`** — parameter blobs in the `.cffgz` bundle.
- **`hypothesis`** — property tests (FE accounting invariants, normalization).

Explicitly not adopted (revisit only with a ticket making the case):

- **`flax`/NNX** — neural-network library; nothing here is a neural net until
  hybrid surrogate nodes (LIMITATIONS S5), and Equinox covers that day too.
- **`numpyro`/`blackjax`** — sampling-based inference; different paradigm, and
  leaning on it risks "shortcut" FE evaluation.
- **`distrax`/`tfp.substrates.jax`** — heavyweight for what Cat/Dirichlet need;
  `jax.scipy.special` (digamma, logsumexp) covers the math. Reconsider at the
  continuous milestone.
- **`pgmax`** — read for arena-layout ideas; BP-only scope makes it a dead-end
  dependency.
- **`inferactively-pymdp` (1.x JAX rewrite)** — investigated (ticket cff-9n7o):
  contains zero Bethe/CFFG/GFE machinery (fixed-form POMDP only), and drags in
  matplotlib/seaborn/mctx/networkx as hard deps. Policy: *read-only reference +
  selective vendoring* — small pure MIT functions (stable_xlogx-style numerics,
  dirichlet_kl_divergence, spm_wnorm exact form, info-gain decompositions) may
  be vendored into `cffg/vendor/` with attribution; its `calc_vfe`
  (return_decomposition=True) and chain-VMP serve as cross-validation oracles
  for our FE ledger in M2/M3 tests.
- **`optax`/`jaxopt`** — first-order optimizers; our Newton solves are a few
  lines of `jax.jacobian` + `lax.while_loop` on tiny systems.

One structural rule: **`cffg.core` (IR + serialization) imports neither JAX nor
Equinox.** Templates must be loadable, validatable, and editable on a machine
with no accelerator stack; JAX enters at the compiler boundary.

## Staging for hierarchical and sophisticated inference

- **Hierarchical models: cleanly additive, plan as M7.** Templates-with-ports
  already compose vertically: a higher level places priors on a lower level's
  parameters or goal statistics (Part II §7.2's goal hyperprior is the depth-2
  base case, and the buyer/seller simulation is two coupled models at different
  timescales). What M7 adds is message rules for inter-level couplings
  (Dirichlet hyperpriors exist; Cat-parameters-from-a-parent and temporal
  thickness à la de Vries & Friston 2017 need deriving) and nested-schedule
  support in the VM. No IR or compiler changes anticipated.
- **Sophisticated inference: shallow yes, deep no.** Contingent planning
  ("branch on what I might observe") can be expressed as a *bigger template* —
  observation-branch mixture nodes unrolled into a policy tree — so the engine
  runs it without architectural change; the cost is combinatorial in branch
  depth (practical to depth ~2–3). Deep SI is tree search, which is precisely
  what LAIF exists to avoid; it will not fall out of message passing
  (LIMITATIONS H5/S4). Default posture: receding-horizon replanning first,
  shallow branch templates where contingency demonstrably pays.

## Positions on open questions from the brief

- **Interchange format (brief item 8)**: decided — no ONNX. ONNX encodes directed
  computational graphs of NN operators and cannot represent factor-graph
  structure, variational constraints, or P-substitution. The canonical format is
  our own schema, and the deployable model is the self-contained `.cffgz` ZIP
  bundle described above.
- **Existing probabilistic libraries**: borrow PGMax's arena/tensor-layout ideas
  (by reading, not depending — its scope is BP-only and we need custom messages),
  and use `distrax`/`tfp.substrates.jax` only for distribution math utilities.
  The FE accounting and message rules must be ours — that's the whole point.
- **Scheduling**: agree it's not the hard part. The hard parts are (1) GFE message
  convergence (hence Newton/Corollary-1 form as default, damping available),
  (2) importance-sampling noise in parameter learning, (3) growing the rule
  registry beyond the discrete submodel (continuous/Gaussian GFE messages are
  research work, not engineering — schedule accordingly).

## Milestones

- **M0** — scaffolding: uv project, tk epics, CI (pytest + ruff), this doc.
- **M1** — core IR: graph + constraints + ports, serialization round-trip,
  edit-action API with port-invariant validation.
- **M2** — discrete BP/VMP engine on arenas: CSP + HMM tiers green.
- **M3** — GFE composite node: message table of Part II Fig. 5, Newton solver,
  importance-sampled A-message; unit tests against hand-computed values.
- **M4** — T-maze parity: all three variants, BFE ablation, golden tests vs LAIF.
- **M5** — agent runtime: receding horizon, continual learning, checkpointing;
  a live gridworld demo driven through the sensor/actuator adapters.
- **M6** — resource estimator; `.cffgz` bundle export/import round-trip; GUI-facing
  edit-action hardening.
- **M7** — hierarchical couplings: inter-level message rules, nested schedules,
  goal hyperpriors as the validation case; shallow contingent-planning template
  spike.
