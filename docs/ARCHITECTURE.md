# CFFG Engine — Architecture

Status: living as-built record, updated 2026-08-17. Unless a section says
otherwise, every statement before **Target design** describes behavior in the
current repository. Tickets, not the target section, are the source of truth for
planned work.

Sources: Koudahl et al. Part I (arXiv:2306.08014), van de Laar et al. Part II,
and the project brief in Obsidian `agents/CFFG/0.0.md`.

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
factor-type × constraint combination needs a derived update rule. The
implementation therefore uses an **extensible rule registry**, not a universal
solver.

## Implemented layers

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
│                    groups, plans, and index tables      │
├─────────────────────────────────────────────────────────┤
│ 1. Core IR         typed CFFG graph + constraints +     │
│                    ports; serialization; edit actions   │
└─────────────────────────────────────────────────────────┘
```

### Change-path index

Start here before opening the large compiler and engine modules. This is a
contributor checklist: the authority owns a decision, while the derived surfaces
transport or execute it. A ticket handoff names the applicable row, walks each
surface, and marks non-applicable items explicitly; that keeps repository scans
and review context proportional to the change.

| Ticket shape | Authority | Derived surfaces that must agree | Canonical focused tests | Durable documentation |
| --- | --- | --- | --- | --- |
| New IR field | `src/cffg/core/ir.py` (`Template` or its spec dataclass) | `src/cffg/core/validate.py`; `src/cffg/core/serialize.py` (`to_dict`, `from_dict`, `structure_hash`); `src/cffg/core/plates.py` scoping; `src/cffg/core/compose.py` and `src/cffg/core/edits.py` when editable; `src/cffg/compiler.py` and `Arena` when executable; bundle and agent state when persisted | `tests/test_ir.py`, `tests/test_serialize.py`, `tests/test_plates.py`; add `tests/test_edit_codec.py`, `tests/test_edits.py`, `tests/test_agent_spec.py`, or `tests/test_bundle.py` only for the affected boundary | Contract in this document; public construction and migration in `README.md`; unsupported cases in `LIMITATIONS.md` |
| New factor or rule family | `src/cffg/kernels.py:FactorKernel`, its `src/cffg/core/ir.py:FactorType`, and the family rule module; formal equations are authoritative in code docstrings and `DERIVATIONS.md` | Compiler grouping, slot addressing, parameter packing, `Arena`, and `member_of`; synchronous `_make_step`; scheduled `_batch_stage_fn` / `_scan_stage_fn`; `_make_bfe`; schedule classification, readouts, serialization manifest, estimator, and public exports as applicable | `tests/test_kernels.py`; the family rule and engine-oracle tests; `tests/test_schedule.py`; sparse-support, float32, and uncomfortable-scale cases required by the robustness contract | Derivation in `DERIVATIONS.md`; integration contract here; approximations in `LIMITATIONS.md`; public modeling API in `README.md` |
| Runtime numerical or status invariant | `AGENTS.md` numerical contract and `src/cffg/engine.py` (`RunStatus`, `RunResult`, `Engine.run`) | Flooding, colored/mixed, and exact schedules; host `_has_zero_support` and JIT `_make_support_check`; `_make_step`, `_make_exact_unit`, `_make_propagate`, and `_make_bfe`; inner-solver status; `RollingAgent`, `Colony`, estimator, benchmark, and dashboard consumers | `tests/test_runtime_status.py`, `tests/test_schedule.py`, `tests/test_colony.py`, plus the changed rule family's hostile-input tests | Runtime contract here and in `README.md`; any accepted failure mode or approximation in `LIMITATIONS.md` |
| Learnable or checkpoint feature | `src/cffg/core/ir.py` (`LearnableSpec`, `AgentSpec`, `BeliefBinding`) and `src/cffg/learn.py` (`Family`, `Learnables`) | Validation, JSON/hash, plate scoping, compiler `_resolve_learnables`, `Arena.learn_plans`, `member_of`, engine readouts and parameter writes, agent cadence, checkpoint encode/decode and legacy migration, and bundles | `tests/test_learnables.py`, `tests/test_agent_spec.py`, `tests/test_agent.py`, and `tests/test_bundle.py`; add the contributing factor family's statistic test | Conjugate contract here; statistics beyond the papers in `DERIVATIONS.md`; public cadence/checkpoint API in `README.md`; remaining family gaps in `LIMITATIONS.md` |

Duplicated execution surfaces are deliberate review pairs: host/JIT,
flooding/scheduled, single-instance/`vmap`, and current/legacy checkpoint
decoding. A change that touches one side names the other in its ticket handoff,
even when the second side is checked and found not applicable.

### 1. Core IR and serialization (`cffg.core`)

A pure-Python (dataclass) representation, independent of JAX:

- `VariableSpec` supports discrete cardinalities, Gaussian dimensions, and
  scalar exponential-family tags. `FactorSpec` holds a type plus parameter
  references, and `ConstraintSpec` holds normalization, marginalization,
  form/δ, moment-matching, and P-substitution beads. GFE composites are derived
  compiler plans detected from factors and constraints; they are not a separate
  serialized IR node.
- **Templates with ports.** A template is a CFFG fragment whose boundary edges are
  declared **ports** with roles: `data` (sensor clamp), `goal` (goal prior),
  `prior` (parameter prior), `glue` (connects to the next unrolled slice). A model
  is valid iff every port is terminated by data, a prior/goal, or a glue bond.
  This is the single mechanism behind unrolling, rollouts, "no dangling edges,"
  and composing templates — one invariant instead of four features.
- **Serialized agent bindings (cff-nhxf).** A complete rolling model may carry
  one `AgentSpec`: the planning action readout, perception action/observation
  DATA ports, and one or more persistent `BeliefBinding`s. Each belief names
  its stable state blocks, categorical prior factor, filtering posterior
  variable, boundary factorization, and optional episode-scale learnable.
  These names are model structure, round-trip through JSON/`.cffgz`, and
  participate in the structure hash; the compiled arena remains derived.
- **Serialization**: versioned JSON is the canonical structure and constraint
  schema. Array data (A matrices, Dirichlet counts, goal statistics) stays
  outside JSON and is referenced by relative `.npy` or `.safetensors` paths.
  `.npz` is a ZIP container and cannot memory-map its individual arrays through
  ordinary `np.load`, so it is reserved for `RollingAgent` runtime checkpoints,
  whose many small state arrays benefit from one container but do not need
  memory mapping.
  The structural content hash ignores blob paths and groups compatible Colony
  instances.
- **Init templates (cff-wx28)**: a `BlobRef` may carry a canonical-initializer
  spec (`cffg.inits`: uniform/one-hot/sticky/banded/shift/seeded-random plus
  Gaussian eye/SPD), so warm-start parameters need no external file; any
  probability template takes a `strength` knob that turns it into Dirichlet
  counts (confidence-weighted priors for learn_a/learn_b). Resolution order:
  explicit arrays > existing blob file > init spec — a persisted blob overrides
  its warm start on reload. Init specs, like paths, are erased from the
  structure hash (parameter data, not structure).
- **Portable bundle**: the deployable unit is a ZIP (`.cffgz`) containing
  `model.json` (structure + constraints), `blobs/` (parameter arrays), and
  `manifest.json` (schema version, content hashes, provenance). Unzipping and
  pointing the engine at `model.json` is a full model restore. Bundle parameter
  members are canonical `.npy` files; ordinary external references may also use
  safetensors. Plate fragments remain nested specs in `model.json`; the manifest
  inventories every grounded tied/per-copy array, and per-copy blob paths use an
  `{index}` marker resolved during grounding. Array-key-derived subpaths prevent
  collisions without flattening the plate spec. Original flat schema-1 manifests
  remain loadable. No third-party interchange format — this schema is canonical.
- **Edit actions**: the only public mutation API is transactional bundles
  (`add_slice`, `bind_data`, `set_constraint`, `compose(template, port_map)`) that
  validate the port invariant before committing. This client-neutral API is also
  the incremental serialization of model-edit history.

### 2. Compiler (`cffg.compiler`)

Compilation has two phases, mirroring PGMax:

1. **Grounding and validation**: plate fragments expand, the complete Template
   validates, arrays resolve, and edges and constraints are classified. Template
   composition, horizon construction, and past/future constraint selection have
   already happened in model builders; observations remain runtime inputs.
2. **Arena layout**: group factors and edges by (type, shape) into dense arrays;
   messages live in one flat log-space tensor per direction with gather/scatter
   index maps. One synchronous sweep and the BFE ledger are jitted. Exact
   two-pass schedules fuse propagation, certification, support checks, and the
   ledger; periodic scheduled stages use `jax.lax.scan`. The general
   flooding/colored/mixed convergence loop remains Python host control flow.
   Structure edits rebuild the arena and compiled functions; observations and
   parameters are state inputs. Static shapes throughout.

Grounding derives variable degree and port incidence in linear passes over the
factor endpoints and ports. Validation owns the port invariant; arena layout
derives the same transient indexes for conditioned-output checks and dangling
edge binding. Neither pass serializes those indexes or rescans the complete
template for each variable. The incidence work is therefore `O(V + I + P)` in
variables, factor endpoints, and ports.

The repeated rSLDS slice has no private IR or rule path. Ordinary arena grouping
lowers its `recurrence_link`, `gaussian_mixture_linear`, `gaussian_linear`, and
equality members into a constant number of shape-batched operation bodies as
the horizon grows. The historical cold-cost defect was quadratic host incidence
lookup before that lowering, not one JAX operation body per slice. Removing the
rescans leaves the grouped plans, schedule, message rules, ledger, and runtime
statuses unchanged.

Executable reuse stops at the structural boundary. Template structure selects
the arena and auto schedule; schedule content is fingerprinted against that
arena. Horizon, factor dimensions/cardinalities, and schedule therefore select
different static programs, while JAX specializes those programs by input shape
and dtype. Observations and every named parameter family live in `State` and
remain dynamic executable inputs, so same-shape values reuse the engine without
changing serialized model meaning. The rSLDS cold benchmark compares the
pre-index and indexed commits in fresh processes and requires exact equality of
support and statuses plus dtype-scaled parity of messages, marginals, Gaussian
moments, and Bethe free energy.

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
make every ledger term +inf — loud by construction, never NaN.

Categorical, Dirichlet, and Gaussian now share this algebra through
`NATURAL_FAMILIES` / `JNATURAL_FAMILIES`: products route through natural
addition, moments come from log-partition gradients, and family methods own
the corresponding entropy/energy algebra. Their shape-specialized storage is
deliberately unchanged: categorical log vectors and Dirichlet
concentration/count segments remain in the discrete arena, while Gaussian
`(eta, Lambda)` pairs remain in the Gaussian arena. `rules.py`, `learn.py`,
and `gauss.py` are adapters onto the family entries, so structure hashes,
schedules, and serialized layouts do not move.

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

The public learning boundary reads the same D3 local belief, not separately
reported endpoint marginals. `Engine.gaussian_node_stats` exposes each soft
selector responsibility and responsibility-weighted component moments; for a
linear mixture these are augmented `(Syy, Syx, Sxx)` statistics ready for one
MNIW update. Exact selector zeros therefore produce exact zero weighted
statistics. The separate `Engine.gaussian_mixture_predictives` boundary reads
the factor-to-`out` cavity: selector plus Gaussian input for a linear mixture,
but never the message facing `out`. It reports component moments and their
moment-matched collapse. Both boundaries are derived reads of `State`, store
nothing in the arena, and return `proper=False` with shape-preserving NaNs when
the required local Gaussian belief is not normalized.

A P-substitution bead on the hybrid node's `out` role changes that boundary:
when the output faces a fixed Gaussian goal, the compiler absorbs both factors
into `HybridGfePlan`. `cffg.hybrid_gfe` solves the D5 fixed point over the
selector with the shared gauge-reduced/backtracked simplex Newton core, writes
the moment-matched predictive on the substituted Gaussian edge, and leaves the
Gaussian state edge on the existing D3 BP/collapse rule. Its direct `G-hat`
ledger term lower-bounds the exact substituted objective. The O(K²)
Hershey–Olsen sandwich and H(rho) cap are computed only by
`Engine.hybrid_gfe_diagnostics`, not during ordinary inference or scoring.

`recurrence_link(out, in, drive)` is the recurrent hybrid primitive used on
the rSLDS path. For each previous regime it stores `K - 1` source-specific
stick logits, with `weights[K,K-1,d]` and `offsets[K,K-1]`, rather than a
dense transition tensor over a discretized continuous state. A dedicated
`RecurrencePlan` carries the two categorical slot tables and Gaussian slot
table into both arenas. The compiler also derives connected ordinary
categorical components around recurrence factors and records whether each is
acyclic. This metadata authorizes an exact structured-chain claim only for an
acyclic component; a loop retains ordinary local Bethe semantics.

`cffg.recurrence` applies D9's hybrid rule. The current Gaussian moments give
one bounded expected-log pair potential through the coordinate-optimal
Pólya–Gamma/Jaakkola–Jordan auxiliary. Log-space sum-product sends messages
through both categorical roles and owns one source/destination pair belief.
That same pair supplies the at-risk and signed-success masses for the Gaussian
drive message. The recurrence node energy uses the pair entropy, the expected
bounded potential, and the Gaussian entropy; ordinary categorical and
Gaussian edge corrections retain their existing ledger ownership. With a
fixed drive, the categorical result is exact on an acyclic component. With an
uncertain drive, the JJ potential remains a lower bound and the node energy
remains a conservative upper bound. Exact categorical zeros remain exact,
improper Gaussian beliefs are loud, and one unfused categorical pass costs
`O(P K²)` work and transient pair memory for `P` factors without allocating a
`K³` transition tensor. Flooding batches that pass synchronously. The mixed
schedule runs the same recurrence coordinate before conditionally refreshing
ordinary BP islands; recurrence fusion is an optional execution optimization,
not a second pair or energy owner. The sweep and BFE deterministically
rematerialize the pair from current cavities and drive moments; no pair array
is persisted or exposed as a public readout before cff-ifbr.

Its `gaussian_node_stats` record is intentionally pre-PG: the two categorical
beliefs and exact supports plus the drive mean, directly inverted covariance,
and second moment. The direct covariance prevents recurrence learning from
recovering a small covariance by subtracting two large, nearly equal moments.
`Engine.recurrence_weight_stats` is the derived layer: it checks factor-local
probability, covariance, parameter, logit-second, and finite-output predicates;
forms stable PG coordinates from the mean and covariance; and emits additive
Gaussian natural statistics for augmented `[R; r]`. The inference and BFE
owners are structured in this release, while this public learning readout
still uses the historical product of endpoint singletons; cff-ifbr owns its
pair-derived replacement. Its diagnostic scalars
identify the failed predicate and its dtype-scaled bound. The generic
`pg_weights/recurrence_link` learning pairing pools those statistics into
`conj.PGWeights` outside JIT and rolls its posterior mean into the next trial.
Its arrays retain source-regime, stick, and augmented drive-coordinate axes;
role-bearing addresses write separate weights and offsets.

`cffg.models.rslds` is the end-to-end composition boundary. Its serialized
template uses one `recurrence_link(z[t+1], z[t], x[t])`, a
`gaussian_mixture_linear` dynamics node selected by `z[t+1]`, and a shared
`gaussian_linear` emission per slice. Four serialized `LearnableSpec`s own the
initial-regime Dirichlet, component-MNIW dynamics, PG recurrence block, and
tied emission MNIW. Their statistic suppliers come from explicit `sources` or
shared blob references. Factor spelling and sorted factor order have no
learning meaning; the factor-renaming regression changes every name and
retains all four posterior values. The continuous initial-state mean and
covariance are deliberately not learnable: they remain fixed outputs of the
initializer.

Two host cadences deliberately share that assembly through `Learnables`.
`RSLDSLearning` remains a compatibility facade for the historical
initial-regime, dynamics, and recurrence boundary; its distinct-trial
`update`, checkpoint, and installation operations delegate to a restricted
runtime and leave the historical fixed emission unchanged. `fit_rslds` owns a
single observed sequence: `PCAARHMMInitializer` fixes an observation-derived
PCA coordinate system and an initial regime order; each outer coordinate
constructs all four posteriors from immutable priors, calls
`Learnables.replace`, and writes them atomically. Thus an outer iteration
cannot count the same sequence as a new trial or expose a partly replaced
coordinate.

Both cadences install point rollovers through role-bearing addresses into a
fresh observation-bound state. The compiled engine and every runtime shape
stay fixed. A finite inner `budget_exhausted` result may trigger a bounded
restart from that same fresh state. Each retry doubles the update fraction
while it remains a convex natural-parameter blend; this changes the route but
not the fixed point, support, schedule, tolerance, or FE certificate. A
non-finite, infeasible, or diverged attempt never retries. After the last
parameter write on an exhausted cadence, the same bounded schedule runs a
terminal hidden-state solve, so the returned state and point parameters agree.
On convergence, the second qualifying map instead returns the coordinate it
just evaluated: its hidden-state beliefs, installed runtime parameters, and
learned posteriors already agree, and the map's unchecked proposal is not
installed. The cadence rejects an engine with functional `mu3` groups before
running: it has no explicit key and particle-reuse contract with which to make
two stochastic attempts comparable.

`RSLDSFitResult.phases` is the evidence boundary for this host schedule. It
records each attempt's damping, public and recurrence status, iteration count,
FE and message-delta tails, Gaussian-belief propriety, finite-state check, and
parameter scales before and after replacement. Every phase explicitly names
its evaluated, proposed, installed, certified, and returned coordinates; it
also retains proposed scales when a converged phase discards an unchecked
proposal. Inner Bethe free energy remains the engine convergence signal. The
outer loop's status authority is the maximum scale-normalized residual over the
normalized initial-regime potential and the dynamics, recurrence, and emission
values proposed by one complete runtime map. It requires two consecutive maps
at or below the configured tolerance for `converged`. The first installs its
proposal; the second certifies and returns its evaluated coordinate. A rebound
resets the consecutive-hit count. Exhaustion terminal-solves the last installed
coordinate, while inner and non-finite failures propagate their status without
installing a malformed proposal. Each map also retains the historical mixed
posterior/reporting residual and a transition-probability residual on fixed
latent probes derived from the observations and their initializer. Those two
views diagnose representation or functional disagreement but do not select
status. None of the three views makes a scalar outer-objective claim.

The fit boundary remains deliberately small and model-owned:

| Responsibility | Owner | Classification and evidence |
| --- | --- | --- |
| Serialized topology, tied blobs, statistic sources | `cffg.models.rslds` | rSLDS-specific: it names recurrence, mixture dynamics, emissions, and data ports. |
| PCA/ARHMM initialization and continuous initial-state prior | `cffg.models.rslds_init` | rSLDS-specific: it fixes latent coordinates and regime labels from observation pairs. |
| Posterior accumulation/replacement, rollover, atomic writes, checkpoints | `cffg.learn.Learnables` | Reusable: ordinary Gaussian declared-learning fixtures and rSLDS exercise the same public operations. |
| Observation-port binding and result construction | `cffg.models.rslds_fit` | rSLDS-specific: the public result separates recurrence status, inner FE, and outer residual. |
| Attempt evidence | `cffg.models.rslds_fit` | rSLDS-specific: it inspects recurrence status and Gaussian/mixture belief propriety through an explicit wrapper. |
| Named coordinate residual and certification decision | `cffg.models.coordinate_fit` | Reusable evidence and identity primitives: they compare model-owned named blocks, apply the consecutive-hit rule, and keep an unchecked proposal distinct from its evaluated parent. |
| Runtime and transition-function coordinate views | `cffg.models.rslds_coordinate` | rSLDS-specific: it projects learned values into installed semantics and evaluates a truth-free observation-derived transition probe. |
| Status-gated run, identical-state retry, runtime-map certificate, terminal solve | `cffg.models.rslds_fit` | The model-owned loop supplies rSLDS parameter views and returns the coordinate selected by the shared decision primitive. The remaining retry and terminal-solve policy stays here until a non-switching production fitter proves the same interface and parity. |

No shared fitting framework is introduced on that last row merely from one
client. Ordinary non-switching Gaussian models already prove the shared
declared-learning operations directly; they do not yet prove rSLDS phase or
retry policy as a reusable abstraction.

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
zeros) — `gamma`, `beta`, `poisson_obs`, and `bernoulli_obs` are kernels,
and `equality` remains structural infrastructure. The discrete and Gaussian
families share the natural-family math but still dispatch through central
compiler/engine paths; their migration onto executable registration is
tracked by cff-209l. Hybrid families also remain central.

Compatibility policy (AC6, pre-1.0): the Python registration API
(`register_kernel`, the `FactorKernel` field set) is internal-unstable and
may change in minor releases while built-ins migrate. The serialized surface
— factor type names, the `kernels` manifest keys, and the loud-failure
guarantee — is stable from the first release that writes it. External
kernels must use their own impl-id namespace (`cffg.` is reserved) and bump
MAJOR whenever message/energy semantics or the parameter schema change.

### 2f. In-graph Dirichlet parameter edges (cff-fr5c)

`FactorSpec.param_edges` is executable for `categorical`, `matrix_cat`, and
`transition_mixture` likelihoods under an explicit mean-field constraint.
The parameter edge faces one `dirichlet` prior factor; its schema-1
`VariableSpec.cardinality` is the flattened table size, while the likelihood's
structural cardinalities recover the canonical concentration shape `[K]`,
`[K_out,K_in]`, or `[U,K_out,K_in]`. A missing/improper prior, a shape mismatch,
or an unconstrained latent use is a compile error. Exact BP is intentionally
not approximated: its likelihood-to-table message is a mixture, not
Dirichlet-closed (DERIVATIONS D6a).

The derived `DirichletGroupPlan` reuses flat message storage but keeps
parameter slots out of categorical normalization, support, decoding, and the
ordinary edge-entropy ledger. The two parameter slots pack positive prior
concentrations and nonnegative fractional counts, not `alpha-1`; summing them
recovers the posterior without losing sparse float32 concentrations to
cancellation at one. Structural messages consume `E[log theta]`. Runtime
counts remain derived state, so JSON and `.cffgz` reconstruct the arena solely
from template constraints, `param_edges`, and prior blobs.
`Engine.dirichlet_marginals` exposes concentrations, means, expected logs, and
propriety.

The VFE implementation combines the prior node, likelihood node, and
parameter-edge entropy correction algebraically before evaluating hostile
sparse inputs. At the coordinate update this is the stable beta-function
ratio from D6a; a literal decomposition remains tested on small models. This
conjugate path is the proper base posterior for `mu3(A)`, not a representation
of `mu3` itself.

### 2g. Learned goal parameter edges (cff-gjjk)

A P-substituted goal-observation composite may replace its fixed categorical
goal blob with a `statistic` parameter edge facing a proper Dirichlet prior.
The mean-field bead on the goal and
`goal_uncertainty=dirichlet` on the P-substitution make that coordinate
explicit in serialized structure. `GfePlan` owns the parameter update because
the categorical goal node and substituted observation edge are absorbed into
one composite; it is not also compiled as an ordinary `DirichletGroupPlan`.

Each future member emits `n_t=Abar_t zbar_t`. Members whose priors read the
same blob form one tie group and use `alpha=alpha0+sum_t n_t`; their state
message consumes `E[log c]`. The complete stable ledger adds the shared
Dirichlet beta-function ratio once, not once per member. Runtime rollover is
explicit: `goal_prior_updates` reads a converged trial, and
`with_goal_priors` accepts it only on a fresh state. `RollingAgent` stages
planning output and commits it once at `reset_episode`, with current/pending
hyperparameters checkpointed separately.

### 3. Message rule library (`cffg.rules`)

Discrete BP and VMP use factor-type update and energy tables; `FactorKernel`
packages the same pair for registered EF factors; specialized GFE composites
have explicit update and ledger functions. The compiled plan selects among
these surfaces:

- **Standard rules**: sum-product (BP) and VMP for `Cat`, `Dir`, `TransitionMixture`
  (the TM node from Part I Appendix C), equality nodes, δ.
- **GFE rules** for the P-substituted goal-observation composite (Part II Fig. 5):
  conjugate Dirichlet message to learned goal statistics,
  softmax/Newton backward state message
  (Corollary 1 form, with a gauge-projected globalized Newton solve), and the
  optional importance-sampled functional log-message to A. The serialized
  `parameter_message=mu3_is_v1` constraint compiles to evaluator identity
  `cffg.mu3_is.v1`; the proper Dirichlet base posterior comes from the
  perception/VMP path. All weights are evaluated in log-space and sampling
  takes an explicit PRNG key.
- **GFE-TM rules** (cff-fk2u, beyond the papers): P-substitution on the
  transition node gives dynamics curiosity — the same Corollary-1 fixed point
  over the joint (state, action) simplex, with the downstream message in the
  goal prior's role. Its substituted interior edge emits the predictive
  cavity (so the two facing messages reconstruct q_w), while its ledger uses
  the node-local objective without the downstream Lagrange coupling. Formal
  derivation + plain-English companion in docs/DERIVATIONS.md D1.
- **Hybrid GFE rules** (D5): P-substitution on a
  `gaussian_mixture(_linear)` output facing a Gaussian goal solves the
  regime-selector fixed point, preserves D3 continuous messages, and reports
  the collapse-entropy optimistic bound plus its exact-channel sandwich.
- **Recurrent hybrid rules** (D9): source-specific stick-breaking transitions
  couple a structured source/destination pair to a Gaussian drive.
  Pólya–Gamma/JJ potentials feed log-space sum-product and pair-derived
  Gaussian information messages under flooding or the mixed schedule. The
  matching pair-region ledger records its conservative VFE bound direction;
  compiler component metadata limits exact-chain claims to acyclic ordinary
  categorical components.
- Every update path has a matching **local free-energy contribution** (average
  energy plus the appropriate entropy terms). `_make_bfe` adds those node terms
  and the arena-derived edge-entropy corrections, so the global BFE/GFE follows
  the constrained Bethe ledger rather than a scoring shortcut.

#### 3b. Declared learnables and parameter addresses (`cffg.learn`, cff-x0sv, cff-6tek)

Parameter learning is a declaration, not agent code. A model names its
conjugate posteriors in `Template.learnables`; the compiler resolves each one
against the arena; `cffg.learn.Learnables` holds the hyperparameters and moves
them. Four rules define the contract:

1. **The unit of learning is a blob set, not a factor.** Blobs are how the IR
   already ties parameters — `build_rollout` names one `Bs` on every slice's
   transition, plates name one blob across every copy — so counts from every
   instance sharing a blob accumulate into one posterior. Parameter tying is
   what makes learning transfer across states, so the learnable has to be
   addressed at the thing that is tied. A learnable spans several blobs when
   one parameter is read in several forms (`A`/`Ac`) or when the family is
   joint over several (MNIW over a `gaussian_linear`'s matrix, offset, and
   covariance).

   The compiled form is `Arena.member_of[factor_name] -> MemberAddress`.
   Each frozen member address carries the serialized factor type and one
   frozen `ParameterAddress` per blob-backed role: `(role, blob, key, index,
   transform)`. The transform is `identity`, `probability_to_log`, or
   `covariance_to_precision`. This is a derived runtime index, never serialized.
   It covers the discrete factors, Gaussian priors and linear conditionals,
   Gaussian mixtures, recurrence links, and `gaussian_prec`. Plate expansion
   needs no special case because every grounded factor receives an address.
   `cffg.set_params` validates values in serialized form, fans one blob/role
   write out to every tied runtime row, and updates each stacked array once.
   Synthetic GFE members retain internal learning targets but are not public
   named-write entry points.

2. **Factor type fixes the sufficient statistic; family fixes the update.**
   The registry `cffg.learn.FAMILIES` maps a family name to a `Family`:
   `check` (compile-time admissibility), `stats` (engine readout → statistic),
   `fold` (statistic into hyperparameters), `rollover` (hyperparameters →
   what the rules consume), `raw` (→ what a counts target consumes), `mean`
   (→ the reporting readout). `Dirichlet` serves the discrete vocabulary:
   `categorical` over `[K]`, `matrix_cat` over `[K_out, K_in]`,
   `transition_mixture` over `[U, K', K]`. `NIW` serves only
   `niw/gaussian`; ordinary `MNIW` serves `mniw/gaussian_linear`, including the
   conditioned runtime group used by observed outputs. The rSLDS declares its
   shared emission this way and pools every tied time slice.
   `component_mniw/gaussian_mixture_linear` adds one MNIW posterior per
   selector state, and `pg_weights/recurrence_link` adds one Gaussian natural-
   parameter posterior per source regime and stick. Precision and EF factors
   have named addresses but no generic conjugate adapter yet.

   Hyperparameters, statistics and rollover outputs are keyed dicts, and a
   family names which engine readout it draws from (`cffg.learn.READOUTS`),
   because a conjugate posterior is not always one array read off one
   readout. An MNIW over a `gaussian_linear` carries four hyperparameters,
   accumulates three statistics from `Engine.gaussian_node_stats`, and rolls
   over into three blobs at different shapes. Component MNIW uses the same
   statistic names with leading selector axes and occurrence counts equal to
   soft responsibilities. PG weights instead draw additive `linear` and
   `quadratic` records from `Engine.recurrence_weight_stats`. A `LearnTarget`
   names both the blob it writes and the parameter role that blob fills, which
   routes every multi-parameter write. `Dirichlet` is the degenerate case of
   that contract: one hyperparameter, one statistic, one role.

3. **Sources are perception, targets are everywhere.** Statistics come from
   the readout named by the family: discrete families use
   `Engine.node_beliefs`, while NIW/MNIW and component MNIW use
   `Engine.gaussian_node_stats`; PG weights use
   `Engine.recurrence_weight_stats`. Every eligible tied factor contributes
   one local record; the family pools those records and performs one posterior
   update. A masked conditioned Gaussian-linear output contributes zero
   evidence. An exactly unsupported mixture component likewise contributes
   exact zeros and retains its prior coordinate. GFE composites are write
   targets but never sources: a planning
   graph with parameter uncertainty therefore has zero sources for `A` and
   writes the raw counts α that its novelty term differentiates; the perception
   graph supplies the statistics and writes
   `E[log θ] = ψ(α) − ψ(α₀)`.

   Reading is not evidence, so a learnable may restrict which instances
   supply statistics: `LearnableSpec.sources` is `None` for every reader, a
   tuple for exactly those, and an empty tuple for none. Habits are the case
   that needs all three. The prior applies at every step of the horizon, so
   every planning slice reads it, and none of them is evidence — the planning
   window declares an empty restriction and learns nothing from what it
   merely considered. The perception slice branches its control edge through
   an equality node so the habit prior sits beside the clamped action, which
   makes the node belief there a one-hot on the executed control.

4. **Exactness is documented per family, because the FE ledger depends on it.**
   Dirichlet on a BP group is exact conjugacy: the node belief at a discrete
   factor on a tree is the exact posterior over that factor's variables. On a
   VMP (mean-field) group it is the standard mean-field conjugate update, not
   an exact one — the belief is a product of edge marginals, so the update
   integrates against a factorized measure. Rollover always writes
   E[log θ], the mean-field-consistent point (Part II §5), never log E[θ].

   NIW/MNIW and component MNIW integrate the converged Gaussian node belief
   against the conjugate sufficient statistics in DERIVATIONS D4. Their
   inference rollover writes inverse expected precision, `Psi / nu`, not the
   posterior-mean covariance used by the reporting API. PG weights integrate
   the current legacy endpoint-product Pólya–Gamma statistics and roll their
   Gaussian posterior mean into point weights and offsets; cff-ifbr will bind
   this readout to the same factor-local D9 pair derivation. Every posterior
   remains outside JIT; inference in the next run consumes point parameters
   rather than joint parameter uncertainty.

Cadence is deliberately not part of `LearnableSpec`. `Learnables` keeps a
read-only validated copy of every initial prior. `accumulate(engine, result)`
starts from the current posterior and adds a distinct trial;
`replace(engine, result)` starts from that immutable prior and recomputes one
sequence coordinate. Both require a status-bearing, converged `RunResult`,
accept an `only` name filter, pool tied sources, validate all selected
posteriors in temporary storage, and commit atomically. `write` is likewise
atomic across selected role-bearing targets. Versioned checkpoints encode
each `(learnable, hyperparameter)` pair independently and round-trip the NIW,
MNIW, component-MNIW, and PG arrays. Legacy spellings remain explicit
migrations; a multi-array posterior must be complete rather than silently
combining restored and prior hyperparameters. The
model/runtime interface still declares which boundary learnable is
episode-scale because that choice determines rollover semantics, not merely
update frequency.

#### 3c. Rolling boundary contract (`AgentSpec`, cff-nhxf)

`RollingAgent` consumes only the serialized interface; graph-local conventions
such as `prior`, `u1`, `act1`, `obs1`, `zm1`, `d`, and `e` have no runtime
meaning. Planning and perception templates may use different local names as
long as their bindings agree on logical belief names, state blocks, domains,
factorization, and episode learnables. Core validation resolves every named
variable, factor, port, constraint, blob owner, and learnable before
compilation. Cross-window validation then checks action and rollover domains.

A `BeliefBinding` carries either:

- one categorical joint over all of its named blocks, preserving correlations
  through filtering, checkpointing, and next-window prior injection; or
- one marginal under `factorization=mean_field`, with the binding naming the
  actual `MEAN_FIELD` node constraints in that Template.

This makes boundary approximation part of the CFFG variational family. A
runtime cannot silently replace a correlated `q(z₁,…,zₙ)` by a product of
marginals. Independent blocks may remain separate joint bindings; coupled
blocks use one joint binding or explicit mean-field bindings. Multi-boundary
checkpoints store every belief by its binding name. The legacy `belief` key and
`RollingAgent.belief` view remain exact only for a one-block model.

#### 3d. Factored rollout lowering (`cffg.models.factored`, cff-b47g)

`build_factored_rollout` is a topology-generic front end over the existing IR,
not a new graph vocabulary. `StateBlockSpec` declares each categorical prior,
transition table, selector coupling, and optional initial/transition
learnable. `CouplingSpec` declares an ordered Cartesian product of block
names; `ObservationSpec` reads one block or one such product. Product
coordinates follow NumPy/C order, with the rightmost declared component
fastest. A product-selected transition's public tensor shape is
`[P₁,...,Pₙ,U,K′,K]`; its stored derived blob flattens only
`[P₁,...,Pₙ,U]` into the ordinary selector axis.

Lowering uses no `FactorType` or compiler special case:

1. each binary product is a deterministic, exact-zero
   `transition_mixture` implementing `out = left * K_right + right`;
2. products of more components form a left-deep chain of those nodes; and
3. every logical net with more than two endpoints becomes the minimum
   left-deep chain of ternary equality factors, keeping every materialized
   FFG edge at degree two.

One blob per block is shared by every time member, so blob identity provides
parameter tying and `LearnableSpec.sources` names only past/data-bearing
members. A filter statistic can therefore update the posterior once and the
next planning window writes that update to every tied slice without manual
accumulation.

The complete host Template, never a fragment or plate, owns `AgentSpec`.
Several block marginals are an explicit mean-field rolling boundary. Each
binding cites a neutral boundary marker: a normalized `matrix_cat` of shape
`[1,K]` facing a singleton categorical. That two-node branch integrates to
one, contributes zero net FE, and changes no marginal; its degenerate VMP bead
is the concrete Template anchor for the serialized boundary projection. It
does not factorize extra within-window beliefs. At rollover, the runtime
therefore carries the declared product of block marginals rather than silently
discarding a joint correlation. Exact joint rollover remains the separate
`BoundaryFactorization.JOINT` representation from §3c.

### 4. Inference VM (`cffg.engine`)

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
    constraints, hybrid mixtures, recurrence links, gaussian_prec): each
    undamped outer map updates the iterative phase synchronously, then
    refreshes the BP islands.
    Forest messages whose conditional solution is unique are eliminated from
    the outer chart and reconstructed by an exact full two-pass; cyclic BP
    messages remain coordinates and advance one colored sweep per map.
    Plans containing GFE composites solve the resulting fixed-point residual
    with the D7 fixed-support, gauge-reduced trust-region method. A bounded
    damped sequence first proposes a common-case root, but can return only
    after the same undamped residual, exact-support, inner-solver, feasibility,
    and FE certificates pass; otherwise the trust controller takes over. Mu3
    skips this phase so its total budget remains available for outer-root and
    particle-moment block alternation.
    Derivative kernels are cached by exact chart support, dtype, continuous
    metric, and hardening phase, while parameters and observations remain
    dynamic inputs. Other mixed plans retain the ordinary sweep loop and its
    optional damping.
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
  traces, and final residual. Mixed-GFE results additionally expose the
  accepted outer residual/merit, merit-gradient norm, trust radius,
  accepted/rejected counts, outer-map/trial/Krylov operation counts, solver
  compilation count, and a machine-readable termination reason. The merit is
  numerical step-acceptance machinery and never enters `_make_bfe` or policy
  scoring. GFE convergence requires a root of the complete
  undamped outer map, converged local solves, unchanged exact support, and a
  finite stationary FE ledger; a nonzero stationary merit is exhaustion.
  Max mode uses its message fixed-point residual and validates the decoded
  objective. Non-finite arenas/residuals are divergence, while proven empty
  exact support is infeasible with FE = +∞.
  "Proven empty" is decided in log space by exact comparison with the
  reserved `NEG` sentinel, never by exponentiating or applying a magnitude
  threshold. `exp` underflows below about -745, while a threshold merely
  moves the same false-zero boundary: a proper sparse Dirichlet can produce
  any finite negative E[log θ], including values below `NEG`. Edge support
  intersects the two facing messages' support masks before doing log-space
  arithmetic, so one actual `NEG` term remains distinguishable from two
  legitimate small terms. Dirichlet rollover reserves the sentinel bit
  pattern too: if a dtype cast makes a proper E[log θ] exactly equal `NEG`,
  that one value moves one ULP toward zero rather than becoming false
  zero-support.
  On a two-pass schedule there is no sweep budget: one execution either
  certifies (converged, also with `tol=None`) or reports honestly that it
  did not.
- The D7 chart packs supported categorical messages as log ratios, Gaussian
  messages in a fixed initial whitening, and other minimal EF messages in a
  fixed Fisher metric. Exact zeros are absent coordinates. Its residual scale
  is dtype-derived and frozen for one trust solve; a support or hardening
  change rebuilds the chart, while a finite nonroot termination may begin a
  new solve with a newly frozen scale. Hard δ support is selected once per
  hard phase rather than changing inside a trust model; multiple δ variables
  activate in deterministic sequential phases so later decisions condition
  on earlier commitments. The compiled matrix-free JVP/VJP and Steihaug
  kernels accept per-instance coordinates, scales, and radii under `vmap`;
  the host controller owns active-set and status transitions until cff-wbhu
  moves that carry into a compiled scan.
  A bounded shadow Anderson/Picard sequence may supply candidates, but a
  nonroot shadow point still needs the trust model's Cauchy and reduction
  safeguards. The only bypass is a shadow state that independently passes
  the complete raw-root, support, inner-status, and FE certificate.
- Ordinary and analytic learned-parameter GFE nodes are deterministic. A
  `mu3_is_v1` node fixes its sampled A matrices for the whole inference
  attempt, making every sweep and its FE trace deterministic conditional on
  the explicit key. Shared-A future slices add their functional scores before
  one normalization; the weighted `E[A]` and `E[h(A)]` feed the Fig. 5
  composite without moment-matching back to a Dirichlet. `RunResult` exposes
  ESS, log-normalizer, and converged/budget-exhausted/diverged stochastic
  status. `Engine.score_mu3_epochs` refreshes independent particle sets,
  accepts only converged epochs, and applies burn-in solely to the final
  policy-score average.
- Inference *modes* are just constraint presets on the same grounded model:
  filtering/smoothing (data on past), planning (goals on future, P-substitution
  on), what-if (clamp hypothetical δs), backward inference (goals only, no data).

### 5. Agent runtime (`cffg.agent`)

- **Receding-horizon loop**: at each tick, bind new observation (future edge's
  constraint flips from goal to data δ), re-anchor each declared planning
  prior on its latest filtering posterior, re-infer, and read the declared
  control marginal (or its MAP under a δ-constraint). Two compiled windows
  cover an unbounded episode without graph growth or recompilation.
- **Continual learning**: driven entirely by declared learnables (§3b). The
  agent contributes cadence, while `AgentSpec` supplies all addressing.
  Observation, transition and habit posteriors update every tick from the
  perception graph's node beliefs. Each boundary's optional episode learnable
  accumulates on the first tick and re-anchors that boundary at
  `reset_episode()`.
  `act()` is read-only: planning an action is not evidence that it was taken.
  `learn_a` / `learn_b` remain as the older spelling of `learn=("A",)` /
  `learn=("B",)`. Checkpoints carry every learnable's hyperparameters, the
  named beliefs and episode cadence, so a resumed agent does not re-fire the
  episode-start statistic. Legacy single `belief` and `counts` / `b_counts`
  keys are restored/migrated exactly or refused, never silently dropped.
- Sensor/actuator adapters are plain Python callbacks mapping env observations to
  one-hot data bindings and control marginals to env actions.

### 5b. Colony execution (many small graphs, one machine)

`Colony` owns an instance table of `(structure_hash, State)` pairs and caches one
`Engine` per structure hash. Each `State` is an Equinox pytree containing
messages, parameters, and conditioned observations; structure remains in the
cached engine. Ordinary iterative instances with one structure are stacked and
one scheduled iteration is evaluated under `filter_vmap`, while different
structures run as separate groups. Mixed-GFE instances reuse the same compiled
trust kernels but currently run one host controller per instance so each keeps
its own trust radius, active support, and termination reason. Inter-instance
coupling happens through host-side data binding and marginal reads, never by
merging graphs.

The convergence loop and per-instance status bookkeeping are currently Python
host control flow around the vmapped iteration. The measured 32-instance CPU
speedup is about 1.5× over sequential execution because dispatch dominates.
Stochastic `mu3` evaluators are rejected by `Colony`: explicit PRNG preparation,
particle state, and stochastic statuses exist on `Engine`, but Colony does not
yet carry them per instance. Per-instance structure changes also fragment
batching because each distinct hash owns a compiled engine.

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
- Hybrid nodes originally lacked soft sufficient statistics and predictive
  readouts; cff-lquu resolved that consumer boundary.
- Long rollouts re-solve cold every step and degrade past ~20 slices
  (cff-5edo); fan-out and observe-an-interior-latent need one-call helpers
  (cff-mjku). Role-bearing non-discrete writes are now the `set_params`
  contract in §3b.

**Method note.** The evaluation discipline that made this productive:
every harness declares numeric pass/fail standards *before* running,
misses require a one-line diagnosis phrased in engine terms (those lines
became the tickets above verbatim), and oracle baselines are implemented
independently in numpy so the engine is never graded on its own homework.
Consumer-harness conventions live in README ("Consume the engine from an
experiment harness").

## Canonical rSLDS consumer validation (updated 2026-08-20)

A fresh consumer used CFFG commit `6c01812` as a pinned Git dependency and
reconstructed the current stick-breaking Linderman NASCAR generator without
editing this repository. At the canonical `T=10,000`, `K=4`, latent `d=2`,
observed `dy=10` scale, fixed-generating-parameter inference converged in 248
sweeps: latent-state RMSE `0.001264`, MAP regime accuracy `99.95%`, mean
true-regime probability `0.999106`, and regime NLL `0.001223`. This validates
the assembled recurrence, switching-dynamics, emission, mixed-schedule, and
ledger path at scale. The separate experiment and provenance artifacts live in
`../experiments/cffg-nascar-validation` at commit `0543683`.

That run did **not** validate the paper-style learned model or an rSLDS
advantage over an SLDS, because it supplied generating emission, dynamics, and
recurrence parameters. cff-l7y4 added the separate observed-only path described
in §2d and evaluated it in the same sibling experiment. Statistical authority
stayed phase-separated: fitting received observations, configuration, priors,
and keys; generating states and parameters entered only the later affine/label
alignment and diagnostics. Each rSLDS/SLDS arm compiled one engine, used the
same 8,000/2,000 split and two-coordinate budget, and compared K=4 structure
parity plus the formula-selected K=5 closest parameter budget.

At the preregistered `T=10,000`, seed-0 final, every inner and post-write final
inference converged; every fit truthfully retained outer
`budget_exhausted`. The rSLDS versus same-K SLDS held-out mean log-likelihood
margin was `+0.313296`, passing the registered `+0.005` claim. Its aligned
latent RMSE was `0.001583` versus `0.001559`, and aligned regime accuracy was
`86.325%` versus `88.8875%`; both registered recovery claims missed and no
recovery advantage is asserted. The K=5 control scored `31.762280` held-out
mean log likelihood and `99.6375%` regime accuracy; no comparison claim against
it was registered. Only the rSLDS rollout passed the boundedness gate, with
periodicity `0.517638`, but that was not a preregistered advantage claim.

The learned-result artifact pins package commit `4b82b07`, experiment-code
commit `1350e72`, final artifact commit `f4e7b0f`, and SHA-256
`99d02c92315357e1d778146c7b7ca74cd20aa2390f4ad165ed17515fba29548e`.
Whole-command time was 2,063 seconds with 5.16 GB peak RSS. Cold lowering is
measured separately by the fresh-process cff-r6bl benchmark. On the same Apple
M2 Max CPU/x64 environment, three workers per path gave `T=10,000` median setup
of `741.361` seconds before incidence indexing and `24.060` seconds after, a
`30.81x` speedup. Median converged inference was `27.342` versus `27.020`
seconds. Messages, exact support, categorical and Gaussian readouts, BFE, and
statuses were identical at `T=100`, `1,000`, and `10,000`; both paths retained
seven shape-batched operation bodies. Indexed peak RSS was `1.948` GiB.

The evidence lives in `../experiments/cffg-nascar-validation` at commit
`1b1b714`; `results.json` SHA-256 is
`5432c42ba6795850898e878ebdfad39ef82f8cc7dbf83ddc6f98c8ed3d4651db`
and the environment/profile/raw-artifact manifest SHA-256 is
`515bd5e2dc05741953e225912f62c8d7b0136a3d1d6b9855938b117bcd7f033b`.

### Classic architecture and dual-family task boundary

The final parity audit separates probability-model architecture from fitted
task performance. For the source-homogeneous, shared-emission NASCAR subclass,
the version-2 mapping record and independent oracles cover the complete joint:

| Joint term | CFFG owner | Independent adapter owner | Executable evidence |
| --- | --- | --- | --- |
| `p(z1)` | categorical prior factor | initial `pi_0` | exact categorical mapping, including literal zero support |
| `p(x1)` | tied Gaussian prior factors | one Gaussian repeated across regimes | parameter mapping and independent Gaussian-density oracle |
| `p(z[t+1] | z[t], x[t])` | `recurrence_link` | source-indexed decision-list transition | structured sum-product and finite-enumeration oracles |
| `p(x[t+1] | x[t], z[t+1])` | destination-conditioned Gaussian mixture | destination-indexed adapter dynamics | independent quadrature and destination-dynamics oracles |
| `p(y[t] | x[t])` | tied `gaussian_linear` emission | shared full-covariance emission | parameter mapping and independent information-form likelihood oracle |

Forney equality nodes only split graph degree and contribute no probability
term. Static interface tests keep the fit boundary truth-clean: both learners
receive observations and declared priors before fitting, while generating
states, regimes, and parameters enter only post-fit alignment and scoring.
This proves the bounded probabilistic architecture, not identity of code or
inference. CFFG uses variational Pólya--Gamma/Jaakkola--Jordan updates and
Gaussian moment collapse; the adapter uses Pólya--Gamma Gibbs sampling.

The experiment keeps two generators and claims separate. The classic family
ties recurrence rows, initial Gaussian laws, and emissions across regimes. It
uses 10,000 training observations plus a 2,500-observation held-out extension
and cannot identify distinct source rows. The local heterogeneous family keeps
the NASCAR dynamics and emission but rotates each source's gates; its
1,280/320 split and common-state evaluation grid test source-function recovery
outside the original demonstration.

The immutable final execution completed every one of the three classic and
three heterogeneous coordinates in 51,254.918 seconds. Both family decisions
are `not_ready`. All classic CFFG tiers converged, but all paper-adapter tiers
failed the registered mixing gate, so task parity was not evaluated. In the
heterogeneous family, seed 2102 failed prefit source/gate coverage, seed 2103
exhausted both CFFG outer budgets, and every fitted adapter tier failed mixing.
Descriptive seed-2101 source-function errors also exceeded the registered
0.150 ceilings. Neither family failure alters the other, and no joint claim is
registered.

The complete local evidence is experiment commit `6c0437a`, with results
SHA-256 `eaf4bdccb6b2d9ecf35f59a91fa1012873bba697c8fe499781b28c1f787697c4`
and manifest SHA-256
`b71c41ad46cd2e210a9e204743d8fd58f1a76a69305afae8601c7e307b64a56a`.
cff-l5ag owns adapter mixing, cff-q1au owns robust heterogeneous coverage,
cff-qulu owns outer convergence, cff-0oqo owns source-function recovery, and
cff-3v7q owns classic periodicity robustness. These experiment repairs do not
change compiler, VM, schedule, kernel, ledger, serialization, or factor
ownership.

## Resource estimator

Because shapes are static post-compile, memory is computable analytically from
arena sizes. The estimator reports exact named byte counts for every persistent
State arena (`messages`, Gaussian eta/Lambda/observations/masks, and EF
messages/observations/masks) plus parameters. It does not currently provide a
trustworthy XLA cost analysis; cff-j7mm records that limitation.

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

Message-schedule overlays (Figs 23/24/27 arrows) are not implemented; the
current renderer stops at the typed scene graph.

## Dependency policy

The library's direct runtime dependencies are:

- **`numpy`** for host-side compilation, serialization, and readouts;
- **`jax`** for numerical rules and compiled execution;
- **`equinox`** for State pytrees plus filtered JIT and `vmap`; and
- **`safetensors`** for safe external parameter-blob loading.

Pytest and SciPy are development dependencies; SciPy is an independent
distribution/property oracle. CairoSVG is an optional visualization dependency.
Dynamax and simglucose are isolated to benchmark/development groups. Portable
`.cffgz` bundles use `.npy` members even though the compiler also accepts
external safetensors.

The engine does not directly depend on Flax/NNX, NumPyro, BlackJAX,
Distrax/TFP, PGMax, inferactively-pymdp, Optax, or JAXopt. In particular,
Distrax/TFP remain deliberately unadopted: the EF implementation needs only
`jax.scipy.special` primitives, and third-party log-probability APIs do not
provide the node/edge decomposition required by the Bethe ledger. PGMax and
pymdp are reference material only; no source package is imported or vendored.
`jaxtyping`, Chex, and Hypothesis are likewise not direct project dependencies.

One structural rule: **`cffg.core` (IR + serialization) imports neither JAX nor
Equinox.** Templates must be loadable, validatable, and editable on a machine
with no accelerator stack; JAX enters at the compiler boundary.

## Target design

Everything above this heading describes the repository as built. This section
preserves intended directions and the original roadmap; it is not an
implementation claim. The `tk` dependency graph and individual tickets are the
source of truth for readiness, scope, and status.

### Near-term architectural targets

- **Colony runtime (cff-wbhu, cff-rqqc).** Put the general convergence loop and
  per-instance status carry inside one compiled scan, and give stochastic
  evaluators explicit per-instance PRNG streams and particle state.
- **Resource analysis (cff-j7mm).** Restore correct XLA lowering and calibrated
  FLOP/byte reporting on top of the existing exact arena-byte estimate.
- **Visualization clients (cff-jude).** Use the edit API as the model-editor
  boundary and add message-schedule overlays to the scene graph's wire
  references.

### Hierarchical and sophisticated inference

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

### Long-term positions from the brief

- **Interchange format (brief item 8)**: decided — no ONNX. ONNX encodes directed
  computational graphs of NN operators and cannot represent factor-graph
  structure, variational constraints, or P-substitution. The canonical format is
  our own schema, and the deployable model is the self-contained `.cffgz` ZIP
  bundle described above.
- **Existing probabilistic libraries**: borrow layout and oracle ideas without
  adding PGMax, Distrax, or TFP as engine dependencies. FE accounting and message
  rules remain native because the node/edge ledger decomposition is the engine's
  defining contract.
- **Scheduling**: preserve the current schedule abstraction while removing host
  dispatch from general iteration and fusing common chain slices. Hybrid GFE
  selector messages use the existing mixed schedule; arbitrary
  Gaussian-only/nonlinear GFE messages remain research rather than scheduling
  work.

### Original milestone plan

This is the sequence from the project brief, retained for design context rather
than current status:

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

### Extensible-kernel baseline and optional factor-library roadmap (epic cff-ahps)

The long game is a general factor-graph engine with a deliberately extensible
vocabulary. Epic cff-ahps closes on a finite integrated baseline for
representative discrete, Gaussian, exponential-family, and hybrid paths; it
does not close by claiming all useful distributions or a universal solver.
Additional nonlinear, neural, plate, and topology work remains optional and is
selected from measured demand. Existing and possible extensions are listed
below in dependency order; each still needs its own derivation and executable
contract:

- **M9 — Gaussian core** (cff-4e5z, landed): domain families, information-form
  arena, linear-Gaussian BP, Kalman parity (§2b above).
- **M10 — Gaussian VMP + conjugate learning** (cff-7c5o): NIW/Gamma priors,
  the continuous analog of Dirichlet count rollover.
- **M11 — Hybrid discrete–continuous** (cff-pksr): conditional-Gaussian
  factors, moment-matched mixture collapse, switching-KF tier. The epistemic
  path: GFE stays on discrete regime variables while Gaussians carry
  perception (what the glucose agent needs).
- **Recurrent stick-breaking link** (cff-k9ws, landed): consecutive regimes
  are coupled to a Gaussian drive with source-specific logistic sticks and a
  Pólya–Gamma/JJ VMP update, supplying the transition primitive needed by the
  rSLDS graph without materializing a state-dependent transition tensor.
- **M12 — Nonlinear factors** (cff-o7hs): Jacobian linearization and
  sigma-point moment matching; activates the reserved MOMENT_MATCHING bead
  as EP.
- **M13 — Exponential-family abstraction** (cff-8s6a, cff-l6i5, landed):
  natural parameters + log-partitions as the message contract;
  Gamma/Beta/Poisson/Wishart as table entries. Cat/Dir/Gaussian re-expressed
  with zero test regressions.
- **M14 — Constraint/optimization factors** (cff-e8zk): min-sum semiring mode
  on the existing arenas + combinatorial hard-constraint messages
  (alldifferent, cardinality, linear inequalities); the ledger API must
  distinguish FE (sum-product) from objective value (min-sum).
- **M15 — Neural factors** (cff-1o72): NN nodes via Equinox, FE-gradient
  training through the (implicitly differentiated) fixed point, amortized
  message initialization.
- **Hybrid GFE epistemics** (cff-dfpi/cff-izjd, landed): continuous curiosity
  is channelled through discrete regime variables with a declared
  collapse-entropy bound; arbitrary Gaussian-only/nonlinear GFE remains open.
