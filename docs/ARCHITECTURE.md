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
- **Init templates (cff-wx28)**: a `BlobRef` may carry a canonical-initializer
  spec (`cffg.inits`: uniform/one-hot/sticky/banded/shift/seeded-random plus
  Gaussian eye/SPD), so warm-start parameters need no external file; any
  probability template takes a `strength` knob that turns it into Dirichlet
  counts (confidence-weighted priors for learn_a/learn_b). Resolution order:
  explicit arrays > existing blob file > init spec — checkpoints override
  warm starts on reload. Init specs, like paths, are erased from the
  structure hash (parameter data, not structure).
- **Portable bundle**: the deployable unit is a ZIP (`.cffgz`) containing
  `model.json` (structure + constraints), `blobs/` (parameter arrays), and
  `manifest.json` (schema version, content hashes, provenance). Unzipping and
  pointing the engine at `model.json` is a full restore; checkpointing writes
  the same bundle. Plate fragments remain nested specs in `model.json`; the
  manifest inventories every grounded tied/per-copy array, and per-copy blob
  paths use an `{index}` marker resolved during grounding. Array-key-derived
  subpaths prevent collisions without flattening the plate spec. Original flat
  schema-1 manifests remain loadable. No third-party interchange format — this
  schema is canonical.
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

### 2b. Domain families and the Gaussian arena (cff-4e5z)

Variables carry a domain family: discrete (`cardinality: k`, one-hot) or
Gaussian (`dim: d`, a point in R^d). The two families get *separate arenas*
in one State pytree. Persistent storage is shape-segmented: discrete messages
are one flat `sum_slot K` log array, while Gaussian information-form messages
use flat `eta` (`sum_slot d`) and flattened `Lambda` (`sum_slot d²`) arrays.
Compiler prefix-offset tables map logical slots into each array, so one outlier
domain cannot pad every other edge. Products still add and the vacuous
Gaussian message is exactly zeros (half-edges and initialization are free).
Factor groups key on their full cardinality/dimension signature (`equality` is
polymorphic and routes by family); the VM gathers transient dense blocks only
to that group's local maximum and scatters exact segments back. Thus rule calls
retain static shapes without persistent model-wide padding. Design decisions
worth remembering:

- **Never invert an incoming precision.** All linear-Gaussian rules are
  derived (docstring of `cffg.gauss`) in forms whose only inversions are
  `noise + incoming` combinations — improper/partially-informative messages
  propagate exactly rather than through a lossy moment conversion.
- **Observations condition, they don't clamp.** A continuous δ-clamp has
  −∞ differential entropy and diverges the Bethe ledger, so a Gaussian DATA
  port compiles the likelihood node into an *observed* group taking the
  value (and a missing-mask) as runtime inputs. This keeps the continuous
  tree identity exact: BFE = −log p(bound observations), the Kalman tier's
  pinned test. Missing data contributes −H[mu_in], mirroring the discrete
  missing-data cancellation.
- **One damp knob.** Linear damping on (eta, Lambda) *is* log-space damping
  for Gaussians, so the engine's damp parameter acts uniformly across
  families.
- The Kalman test tier (`tests/test_kalman.py`) mirrors the HMM tier
  exactly: smoother parity, tree-identity FE, missing data, vmap batching.

### 2c. The exponential-family arena (cff-8s6a)

The third domain family (`ef: <family name>`) generalizes the pattern:
scalar EF edges carry natural-parameter messages in `ef_eta [n_efslots, 2]`,
where — as in the other two arenas — products add and vacuous is zeros.
DERIVATIONS.md D2 states the contract; `cffg/ef.py` is the NumPy table
(compile time, tests), `cffg/efrules.py` the jnp mirror (jitted step/BFE).
Adding a distribution is a table entry, not an engine change. Family names
come in two kinds: *message* families (gamma, beta) whose edges carry
beliefs, and *observation* families (poisson, bernoulli) whose edges are
conditioned DATA ports only — their unobserved predictives leave the
exponential family, so `poisson_obs`/`bernoulli_obs` compile exactly like
observed Gaussian nodes (constant conjugate increment + closed-form energy;
missing data drops the node). Conjugate BP is exact on EF subgraphs, so
beads (mean-field, P-substitution) on EF nodes are compile errors, and the
tree identity BFE = −log evidence is pinned to 1e-12 in
`tests/test_ef_engine.py`. Improper beliefs (e.g. the vacuous Gamma point)
make every ledger term +inf — loud by construction, never NaN. The
re-expression of Cat/Dir/Gaussian onto the same abstraction is a separate
zero-regression refactor ticket.

### 2d. Hybrid discrete-continuous nodes (cff-pksr)

`gaussian_mixture(_linear)` factors straddle two arenas: Gaussian out/in
edges, a discrete selector edge (`MixGroupPlan` carries slot tables into
both). Inference is assumed-density: mixture messages on Gaussian edges are
collapsed by moment matching with the direction-dependent weights derived
in DERIVATIONS.md D3 (backward collapse must rebuild the per-component
scales the scale-free message convention drops); selector messages and node
energies are exact for the held beliefs. The reported BFE is a documented
lower bound (LIMITATIONS S1d). The switching-KF test tier holds the engine
to an *independent* numpy implementation of the same rules (tight) and to
exhaustive regime-sequence enumeration (measured gap with headroom — never
pretending assumed-density equals exact). Observed mixture outputs are a
compile error in v1; emissions route through plain `gaussian_linear` nodes.

### 2e. Executable factor registration (`cffg.kernels`, cff-ri8m)

A `FactorKernel` declares everything the compiler, VM, ledger, and
serialization need to run one factor type: the structural `FactorType`,
per-role edge-family pins, parameter schema and packing, message update,
node energy, observation conditioning (tag plus validator), applicable
constraint modes, and optional readouts. The compiler and engine dispatch
from the registry — a kernel-backed factor type has no type-string branch in
either module, and `tests/test_kernels.py` proves it by registering a new
conjugate observation factor (exponential draws of a gamma rate) from test
code, running it against a hand-computed evidence oracle, and asserting the
central sources never name the type.

Registration is loud and demanding by design. `register_kernel` refuses a
kernel without a `HostileInputs` declaration (the numerical-robustness
checklist: sparse support, improper messages, float32, uncomfortable scale),
a namespaced impl id, or a MAJOR.MINOR.PATCH version. Serialized templates
record `(impl, version)` for external kernels, so loading a template whose
implementation is absent fails at load time and names what to install;
a registered kernel with the wrong identity or MAJOR version is refused too.

v1 binding scope: kernels attach to the EF arena, where the message
representation is uniform (natural parameters, products add, vacuous is
zeros) — `gamma`, `beta`, `poisson_obs`, and `bernoulli_obs` are now
kernels, and `equality` remains structural infrastructure. The discrete,
Gaussian, and hybrid families still run on central paths; they migrate onto
the contract under cff-l6i5 and cff-x0sv rather than in one rewrite.

Compatibility policy (AC6, pre-1.0): the Python registration API
(`register_kernel`, the `FactorKernel` field set) is internal-unstable and
may change in minor releases while built-ins migrate. The serialized surface
— factor type names, the `kernels` manifest keys, and the loud-failure
guarantee — is stable from the first release that writes it. External
kernels must use their own impl-id namespace (`cffg.` is reserved) and bump
MAJOR whenever message/energy semantics or the parameter schema change.

### 3. Message rule library (`cffg.rules`)

A registry keyed by `(factor_type, constraint_config, out_edge_role)`:

- **Standard rules**: sum-product (BP) and VMP for `Cat`, `Dir`, `TransitionMixture`
  (the TM node from Part I Appendix C), equality nodes, δ.
- **GFE rules** for the P-substituted goal-observation composite (Part II Fig. 5):
  Dirichlet message to goal statistics, softmax/Newton backward state message
  (Corollary 1 form, 20-ish Newton steps as in the paper), importance-sampled
  log-message to A. All in log-space; every stochastic rule takes an explicit PRNG
  key.
- **GFE-TM rules** (cff-fk2u, beyond the papers): P-substitution on the
  transition node gives dynamics curiosity — the same Corollary-1 fixed point
  over the joint (state, action) simplex, with the downstream message in the
  goal prior's role. Its substituted interior edge emits the predictive
  cavity (so the two facing messages reconstruct q_w), while its ledger uses
  the node-local objective without the downstream Lagrange coupling. Formal
  derivation + plain-English companion in docs/DERIVATIONS.md D1.
- Each rule ships with a **local free-energy contribution** function (average
  energy + entropy terms with correct edge-entropy overcounting), so global
  BFE/GFE is always exactly the sum of registered local terms — the "true BFE, no
  shortcuts" requirement is enforced structurally.

### 4. Inference VM (`cffg.infer`)

- Schedules (cff-d56d, `cffg.schedule`): the engine builds a schedule from
  the compiled arena instead of always flooding. The factor graph over the
  schedulable arenas (discrete BP, Gaussian, EF conjugate-BP) is
  partitioned into islands, and each model gets the strongest schedule its
  topology admits:
  - two_pass (everything schedulable, all islands forests): factors fire by
    decreasing BFS depth (rooted at a diameter endpoint per component),
    then in reverse. Each directed message is updated O(1) times, so a
    chain costs O(E) instead of flooding's O(E × diameter). One execution
    reaches the BP fixed point; the warm status-bearing run is a single
    fused XLA dispatch (propagation + certification sweep + support flags +
    FE ledger). The certification sweep re-derives every message and its
    elementwise mixed absolute/relative change gates `converged` at a
    dtype-derived threshold. No unrelated block or graph depth can widen
    the certificate, so a stale message can only surface as a non-converged
    status, never as silently wrong marginals.
  - colored (pure BP with cycles): forest islands still run their exact
    pass once; cyclic islands iterate greedy-colored Gauss–Seidel sweeps
    (no two adjacent factors share a color) under the ordinary convergence
    loop — same fixed points as flooding, information travels further per
    sweep.
  - mixed (iterative structures present — vmp beads, GFE composites, hard
    constraints, hybrid mixtures, gaussian_prec): each outer iteration
    updates the iterative phase synchronously (with its damping), then
    refreshes the BP islands before the loop evaluates the global residual
    and FE. Forest islands are re-equilibrated exactly (a full two-pass);
    cyclic islands advance one colored sweep per outer iteration and
    equilibrate across iterations, which the residual and FE stationarity
    criteria measure. Budgets, tolerances, and statuses keep flooding-loop
    semantics.
  - flood: retained as an explicit option, and the fallback for δ form
    constraints (whose per-sweep hardening is defined against flooding) and
    models with nothing schedulable. `Schedule.reason` names what forced it.
  Periodic runs of forest rounds compile to a single `lax.scan` whose carry
  holds only the message blocks that cross steps (registerized; chains
  classify automatically), so compile cost is O(1) in chain length and the
  warm kernel has no per-step dispatch. The schedule is a pure function of
  the arena (which is a pure function of the template), so it reconstructs
  deterministically from a serialized bundle; `Schedule.describe()` is the
  stable textual form, and its fingerprint (arena structure hash + content)
  lets the engine reject stale or doctored prebuilt schedules.
- The runtime records a `RunResult` with converged / budget-exhausted /
  diverged / infeasible status, iteration count, FE and message-delta
  traces, and final residual. Sum/GFE mode stops only when FE is stationary
  and messages are coherent; max mode uses its message fixed-point residual
  and validates the decoded objective. Non-finite arenas/residuals are
  divergence, while proven empty exact support is infeasible with FE = +∞.
  On a two-pass schedule there is no sweep budget: one execution either
  certifies (converged, also with `tol=None`) or reports honestly that it
  did not.
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
  c). Transition learning (`learn_b`, cff-cd1i) follows the same pattern with
  per-action Dirichlets, integrated against the *node belief* at the
  transition factor (`Engine.node_beliefs`, the smoothed pairwise marginal) —
  node beliefs are the general sufficient-statistic source the learnables API
  (cff-x0sv) will build on. Checkpoint = serialize current parameter blobs +
  template ref; can happen at any tick.
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
   The adversarial costly-cue tier then leaves the paper fixture: a binary
   Bayes decision independently checks relevance against a random distractor,
   post-observation satiation, sensor-quality and sensing-cost monotonicity,
   preference calibration, realized information gain, return/regret, and
   truthful convergence status (`tests/test_behavioral_epistemics.py`).

Property tests throughout: FE never counted twice (edge entropy correction),
messages normalized, VMP monotonically decreases BFE on tree graphs, BP exact on
trees.

## Evaluation at scale: what consumer harnesses taught us (2026-07-09)

One round of library-consumer evaluation (three independent harnesses in
sibling experiment folders, engine consumed as an editable dependency,
nothing committed here) — the durable engine-level findings:

**Confirmed at scale.**
- *Exactness survives real graphs*: a two-level 784-leaf plate classifier's
  decisions matched the closed-form posterior on 100% of items; a hybrid
  switching filter matched its independent numpy reference to 3e-12 *in
  situ*, not just in unit fixtures.
- *The compile boundary is the killer feature*: compile-once/swap-params
  turns trial loops from ~90 s to ~2 s; streaming filters hold literally
  flat memory over 10⁵ steps at ~455 steps/s (all growth is one-time JIT
  warmup). This is what "parameters are inputs" was for.
- *Epistemic value is real and cheap*: information-gain-driven observation
  selection reached ~84% of full-observation accuracy from a quarter of the
  input, +27 points over random selection at the same budget — computed
  from engine beliefs with a page of numpy (cff-gul8 promotes it inward).
- *Conjugate continual learning behaves*: online hard-EM recovered
  switching dynamics to 0.009 Frobenius error; the no-transition-curiosity
  Dirichlet count-learning arm recovered planning performance to 0.80 success
  over 200 episodes. GFE-TM behavioral calibration is tracked separately in
  LIMITATIONS S1c and cff-2gev.
- *Long-horizon GFE planning beat a QMDP baseline* (0.79 vs 0.64 success)
  through H=20 — with a measured degradation at H=25 (LIMITATIONS S1g,
  cff-5edo).

**Bottlenecks found (each a ticket, in priority order).**
- Sweeps, not FLOPs: wide fan-ins expand to *linear* equality chains, so
  BP diameter — hence sweeps-to-convergence — is O(N) in plate size; this,
  not compile time (1.2 s for 784 leaves) and not per-sweep cost, capped
  perception throughput (cff-9xmq: balanced trees).
- Hybrid nodes are read-opaque: no soft sufficient statistics, no
  posterior-predictive readout — consumers fall back to hard-EM and
  hand-assembled predictives (cff-lquu).
- Long rollouts re-solve cold every step and degrade past ~20 slices
  (cff-5edo); fan-out and observe-an-interior-latent need one-call helpers
  (cff-mjku); per-member parameter writes beyond discrete groups need the
  generic learnables API (note on cff-x0sv).

**Method note.** The evaluation discipline that made this productive:
every harness declares numeric pass/fail standards *before* running,
misses require a one-line diagnosis phrased in engine terms (those lines
became the tickets above verbatim), and oracle baselines are implemented
independently in numpy so the engine is never graded on its own homework.
Consumer-harness conventions live in README ("Consume the engine from an
experiment harness").

## Resource estimator

Because shapes are static post-compile, memory is computable analytically from
arena sizes. The estimator reports exact named byte counts for every persistent
State arena (`messages`, Gaussian eta/Lambda/observations/masks, and EF
messages/observations/masks) plus parameters; the wider XLA/calibration audit
remains tracked by cff-j7mm. `jax.jit(...).lower(...).compile().cost_analysis()`
can provide FLOPs/bytes per MP sweep, with a calibration run supplying measured
sweep time.

## Diagram rendering (`cffg.viz`)

Templates render in the papers' own visual algebra (Part I §5) through a
four-stage pipeline, each stage importable on its own:

1. **`diagram.derive`** — semantic model: the default variational spec made
   explicit (joint bead per node, normalisation bead per internal edge), then
   perturbed by constraints/ports/blobs exactly as the notation prescribes
   (mean-field splits, P-substitution squares, δ symbols, pointmass
   terminals, glue-port ellipses, GFE composite boxes mirroring the
   compiler's detection).
2. **`compress.present`** — Part I p.19's four compression passes as
   *presentation directives* (level 0 = Fig 14, level 4 = Fig 19); passes
   change what is drawn, never what the diagram means. A node with a
   pointmass edge keeps its beads at every level (the f_d case).
3. **`layout.layout`** — deterministic rectilinear grid: glue-anchored rail
   left→right (time), selector branches above, obs/goal stacks below,
   terminals on the first free side. Placement is a pure function of names
   and topology, so edits move only the rows/columns they touch — the
   stability property a template editor needs.
4. **`scene` / `svg`** — the layout emits a typed scene graph whose
   primitives carry template refs (`factor:`/`var:`/`port:`/`blob:`/`gfe:`)
   for GUI hit-testing; `svg.to_svg` serialises it dependency-free. PNG goes
   through the optional `viz` extra (cairosvg). Golden SVGs pin the pipeline
   for the canonical models.

Message-schedule overlays (Figs 23/24/27 arrows) are deferred (cff-jude);
the scene graph anchors overlays to wires by ref when that lands.

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
  `jax.scipy.special` (digamma, logsumexp) covers the math. Reconsidered at
  the continuous milestone as promised (cff-8s6a, 2026-07-09) and the answer
  held: the EF abstraction needs exactly two special functions (digamma,
  gammaln), both in `jax.scipy.special`; everything else is one-line
  natural-parameter algebra (`cffg/ef.py`, `cffg/efrules.py`). A framework
  dependency would buy nothing and cost the closed-ledger discipline —
  distrax/tfp log-probs don't decompose into our node/edge Bethe terms.
  scipy was adopted as a *dev* dependency only (property-test oracle).
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
- **M8** — plotting facility: paper-faithful CFFG diagrams (landed, cff-ql2n).

### Factor-library roadmap (epic cff-ahps)

The long game is a general factor-graph engine: all useful probability
distributions (continuous, discrete, hybrid), constraint-satisfaction factors
for optimization under uncertainty, and neural-network factors with backprop.
Ordered by dependency, one ticket each — derivations happen at top level and
land as docstrings; delegation guidance lives in the epic:

- **M9 — Gaussian core** (cff-4e5z, landed): domain families, information-form
  arena, linear-Gaussian BP, Kalman parity (§2b above).
- **M10 — Gaussian VMP + conjugate learning** (cff-7c5o): NIW/Gamma priors,
  the continuous analog of Dirichlet count rollover.
- **M11 — Hybrid discrete–continuous** (cff-pksr): conditional-Gaussian
  factors, moment-matched mixture collapse, switching-KF tier. The epistemic
  path: GFE stays on discrete regime variables while Gaussians carry
  perception (what the glucose agent needs).
- **M12 — Nonlinear factors** (cff-o7hs): Jacobian linearization and
  sigma-point moment matching; activates the reserved MOMENT_MATCHING bead
  as EP.
- **M13 — Exponential-family abstraction** (cff-8s6a): natural parameters +
  log-partitions as the message contract; Gamma/Beta/Poisson/Wishart as
  table entries. Cat/Dir/Gaussian re-expressed with zero test regressions.
- **M14 — Constraint/optimization factors** (cff-e8zk): min-sum semiring mode
  on the existing arenas + combinatorial hard-constraint messages
  (alldifferent, cardinality, linear inequalities); the ledger API must
  distinguish FE (sum-product) from objective value (min-sum).
- **M15 — Neural factors** (cff-1o72): NN nodes via Equinox, FE-gradient
  training through the (implicitly differentiated) fixed point, amortized
  message initialization.
- **Research track** (cff-dfpi): continuous/hybrid GFE epistemics — open in
  the literature; hybrid regime-variable epistemics is the working bet.
