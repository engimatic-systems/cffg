# CFFG Engine — Architecture

Status: living as-built record, updated 2026-08-31. Unless a section says
otherwise, every statement before **Target design** describes behavior in the
current repository. Tickets, not the target section, are the source of truth for
planned work.

Sources: Koudahl et al. Part I (arXiv:2306.08014), van de Laar et al. Part II,
and the project brief in Obsidian `agents/CFFG/0.0.md`.

## What the papers give us (and what they don't)

The papers provide a general derivational language and complete updates for
specific submodels:

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

### Paper constraint inventory

The execution class below is part of the release contract. "Exact" means the
named registered construction implements the paper's local constraint without
an added mathematical approximation; it does not make loopy Bethe inference
globally exact. A paper construction with no template syntax is unsupported,
while reserved syntax that round-trips but cannot execute is syntax only.

| Paper construction and source | Template/compiler representation | Execution class | Rule, ledger, direct authority, or guard |
| --- | --- | --- | --- |
| Normalization, Part I §2.1 Eqs. (9a–b), Fig. 4; Part II §2.3 Eq. (2a) | Implicit invariant, not a `ConstraintSpec`; compiler plans create normalized beliefs | Exact for every registered family | `_norm_pad` and natural-family normalizers; node energies in `ledger.make_belief_energy`; `test_fully_observed_marginals_and_bfe` and `test_bfe_equals_negative_loglikelihood` |
| Marginalization, Part I §2.1 Eq. (9c), Fig. 4; Part II §2.3 Eq. (2b) | Implicit paired edge slots, gather/scatter maps, fixed-point edge products, and interior-edge entropy corrections | Exact density consistency at a fixed point for ordinary registered BP/VMP rules | BP/VMP reductions plus `ledger.make_belief_energy`; HMM forward-backward and tree `-log Z` tests. The Gaussian-mixture projection below is a named exception, not exact density consistency. |
| Gaussian-mixture assumed-density projection, repository extension derived in `DERIVATIONS.md` D3 | `gaussian_mixture(_linear)` rules collapse each Gaussian-edge mixture message to the Gaussian with the same mean and covariance | Named assumed-density approximation; only the selected moments, not the full marginal density, agree | `mixg.py`; node terms are exact for held beliefs, while `ledger.make_belief_energy` uses the collapsed Gaussian edge entropy and reports the bounded ledger described in LIMITATIONS S1d; `test_mixg.py` and switching-KF enumeration tests |
| Joint node factorization, Part I §5.1 Figs. 5 and 7 | Absence of a factorization bead selects `GroupPlan.mode="bp"` | Exact registered BP on trees; named Bethe approximation on loopy graphs | `BP_UPDATES`, `NODE_ENERGIES`; exhaustive tree and HMM authorities in `test_engine_tree.py` and `test_engine_hmm.py` |
| Naive mean field, Part I §5.1 Eq. (26), Figs. 5 and 7; Part II §4.1 Eq. (7) | `ConstraintKind.MEAN_FIELD` targets one factor and selects a VMP or Dirichlet-VMP plan | Named mean-field/VMP approximation | `VMP_UPDATES`, `VMP_ENERGIES`, Dirichlet VMP rules, and `ledger.make_belief_energy`; `test_vmp_exact_in_conjugate_case`, `test_vmp_differs_from_bp_with_two_free_edges`; unsupported factor/family pairs are compile errors |
| Structured mean field, Part I §5.1 Eq. (27), Fig. 6; Part II §2.4 Fig. 1 | No role-partition syntax. `BoundaryFactorization` is rolling-state metadata, not this node constraint. | Unsupported | No rule or ledger. The enum cannot express the construction, so it cannot be mistaken for executable syntax. |
| Edge form constraint, Part I §5.2 Eq. (28), Fig. 8 | No generic variational-family bead. `VariableSpec` chooses a model domain/arena and is not a form constraint on `q`. | Unsupported in general; data and optimized δ are the exact special cases below | Unknown constraint kinds fail deserialization; no generic projection or ledger is advertised. |
| Node form constraint, Part I §5.2 Eq. (29), Fig. 9 | No node-belief family syntax. `FactorType` declares model factor `f`, not a constrained form for `q_a`. | Unsupported | No rule or ledger; the construction is absent from `ConstraintKind`. |
| Known data constraint, terminating form, Part I §5.3 Eq. (30), Fig. 10; Part II §3.3.1 Fig. 3 | `PortRole.DATA` on a degree-one variable; `Engine.bind_data` writes categorical one-hot evidence or conditioned Gaussian/EF observations | Exact at registered terminating placements | Port validation requires the exposed variable to have one factor endpoint. Ordinary node energy has no exterior-edge entropy correction; observed/missing tree, Kalman, and EF evidence tests are direct authorities. Unsupported continuous placements are compile errors. |
| Known data constraint, non-terminating form, Part I §5.3 Fig. 10 | No representation: degree-two variables cannot carry ports | Unsupported | `_check_degrees_and_ports` rejects a port on a fully connected variable. Supporting the paper's information-blocking bead on an interior edge requires separate IR and execution semantics. |
| Fixed model clamp, Part II §2.4 Fig. 1 | `FactorSpec.param_blobs` binds a fixed value to a factor parameter role | Exact for registered factor roles; this is part of the model, not observed data | Parameter packing and each registered rule consume the blob; `test_blob_bound_node_loses_default` and `test_wire_terminals` check the distinct black-square diagram terminal. |
| Optimized δ/point mass, Part I §5.3 Eq. (30), Fig. 11 and Fig. 29 application | `ConstraintKind.DELTA` on a discrete edge; compiler records `delta_pairs`; the hardening phase projects both slots to the marginal argmax | Exact discrete point-mass coordinate | Ordinary ledger evaluated on the point mass; `test_map_policy_variant`. Dirichlet-parameter, Gaussian, and EF edges are rejected with named compile errors. |
| Moment matching, Part I §5.4 Eq. (31), Fig. 12 | `MOMENT_MATCHING` names a factor and records `role` plus `projection` in its details. `gaussian_first_second` selects E[X] and E[XXᵀ]; each endpoint is independent. | Exact Gaussian prior/linear rules accept Gaussian moments; nonlinear EP remains pending | `core/moment_matching.py` validates and explicitly migrates declarations; registered Gaussian capabilities gate execution. Figure 12 diagrams retain one E edge bead and dash only selected half-edges. Factor-only legacy payloads round-trip but cannot compile or draw. `test_moment_matching_schema.py` covers both endpoints, transport, admission, and diagrams. |
| P-substitution, Part I §4 Eqs. (20–25), §5.5 Eq. (32), Fig. 13; Part II §3.3.2 Eqs. (4–5), §4.1 Eqs. (6–8) | `ConstraintKind.P_SUBSTITUTION` plus `substituted_role`; compiler derives a discrete, transition-mixture, or hybrid GFE plan only for registered shapes | Exact local Fig. 5 fixed-A composite; named tentative-decision/analytic-moment approximation for uncertain A without functional `mu3`; separately named TM/hybrid extensions; unsupported otherwise | Specialized GFE update/energy functions own the substituted edge term in `ledger.make_belief_energy`. Compiler rejects unsupported roles, factors, goals, uncertainty modes, and evaluator names. Figure 5 authorities are inventoried below. |
| Trial-time planning bead → data bead, Part II §6.3 Fig. 7 (static contrast in §3.3 Fig. 3) | `RollingAgent` uses immutable planning and filtering templates plus a carried boundary belief; it does not mutate one growing trial graph | Exact window-transformation oracle for the fixed-A binary discrete joint-boundary case; named two-window realization outside that boundary | `test_literal_snapshots_record_the_constraint_change_without_rewriting_variables` checks the immutable before/after declarations. `test_two_window_agent_matches_two_step_enumerator_and_local_ledgers` checks both action posteriors, the filter posterior, boundary carry, local GFE, the derived boundary offset, and the two-structure bound against independent enumeration. Exact-zero and float32 cases are separate authorities in the same file. |

Normalization and marginalization are therefore always-on compiler/runtime
invariants, not optional beads in `ConstraintSpec`. General structured forms
and arbitrary facing-node calculations remain paper notation until a concrete
factor/constraint combination has a registered rule and objective owner.

### Part II Figure 5 message inventory

The table follows Figure 5 and Appendix C in source order. The fixed-A rows are
exact local categorical algebra. For uncertain A, Appendix C.1 separately uses
a tentative-decision approximation, replacing `q(A)` by a point mass at its
mean when forming the predictive `q(x)`. Learned goals remain conjugate. The
functional `mu3` evaluator then uses importance sampling for the nonconjugate
message; that sampling approximation is distinct from the tentative decision,
with exact finite-sample ledger identities and truthful stochastic statuses.

| Paper message and source → destination | Mathematical form and execution | Production rule and ledger owner | Independent oracle and hostile-input authority |
| --- | --- | --- | --- |
| `mu1(c)`, Fig. 5 and App. C.3: predicted `x` → goal parameter `c` | Closed-form `Dir(c | Abar zbar + 1)`; executable when the goal uncertainty is declared Dirichlet | `GoalParameterPlan` pools original GFE predictions across shape groups; `goal_parameters.shared_goal_energy` charges one `goal_dirichlet_energy` per serialized owner | `test_mu1_shared_goal_posterior_and_complete_fe_match_direct_algebra` checks the same-shape oracle; `test_shared_goals.py` covers cross-shape posterior, off-coordinate energy, tied trust-region coordinates, and 32 sparse float32 contributors |
| Direct `mu2(z)`, Fig. 5 and App. C.4: goal-observation composite → state `z` | Closed-form categorical `softmax(rho(zbar))` for held moments; recursive direct iteration is not scheduled independently because Part II §4.5 warns it can diverge | `_gfe_rho`; `gfe_update[_stats]` emits normalized `rho` after the stationary solve; `gfe_energy[_stats]` owns `-zbar·rho-H[zbar]` | `test_raw_mu2_and_local_energy_match_independent_formula` checks the raw outgoing message, predictive message, and isolated energy before edge normalization can compensate |
| Indirect `mu2(z)`, Fig. 5, App. C.5 Eq. (25): same composite → state with incoming `Cat(z|d)` | Nonlinear stationary coordinate: solve `zbar=softmax(rho(zbar)+log d)`, then emit `log zbar-log d` (the production `rho` form is gauge-equivalent) | Gauge-projected globalized `_gfe_solve_result`, `gfe_stationary_stats`, and `gfe_update[_stats]`; failures have converged/budget-exhausted/diverged inner status | Simplex enumeration and objective parity in `test_newton_matches_brute_force_argmin` and `test_bfe_equals_local_objective_minimum`; vacuous input, literal zero support, float32, and 40–80-state cases in `test_gfe.py` |
| `mu3(A)`, Fig. 5, App. C.6 Eq. (26): future `(x,z,c)` composite → observation matrix `A` | Functional log-density `zbar·xi(A)`, whose predictive term follows Appendix C.1's tentative-decision approximation; it is not Dirichlet-closed, so `cffg.mu3_is.v1` separately uses fixed-particle self-normalized importance sampling | `mu3_log_message`, `mu3_importance_update`, `Engine.prepare_mu3`, and `_update_mu3`; slice structural terms use `gfe_energy_stats`, and one shared KL leaves the parameter contribution `-log Z` | Expanded-form and binary quadrature oracles in `test_mu3_derivation.py`; multi-slice ledger in `test_multiple_slices_add_functional_weights_and_parameter_ledger_once`; exact support, float32, scale, and Engine divergence authorities named in the runtime section below |

Figure 5's average energy is `U_x = -zbar·rho`. Production's composite
ledger adds the state negative entropy and surrounding edge/factor terms.
For uncertain A, the analytic agent path feeds closed-form Dirichlet moments
into `rho` but does not apply the functional density. Only an explicitly
declared `cffg.mu3_is.v1` evaluator carries the Figure 5 `mu3(A)` update.
`RollingAgent` executes it only when its serialized planning template compiles
to that evaluator; `learn_a=True` alone retains the analytic-moment
approximation.

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
| Derived component coordinate | Serialized explicit region/constraint semantics and approximation identity; D12 equations in `docs/DERIVATIONS.md` (production protocol pending) | New IR field validation/JSON/hash/scoping/edits/composition; topology and specialized/registered ownership; compiler grouping, slots, parameter packing, resolution, assembly, `Arena` and `member_of`; component/boundary indexes and schedule stages; synchronous and scheduled local/batch/scan execution; lifecycle and host/JIT support/status checks; ledger region/separator/adapter ownership; readouts and `Learnables.replace`; learnable/executable/checkpoint/CFFGZ semantic identity and legacy rejection/migration; estimator and public exports | `tests/test_hybrid_separator_derivation.py` mathematical oracle; affected IR/serialize/plates/edits/topology/resolution/assembly, kernel/ledger/runtime/schedule, learnable/readout/checkpoint/bundle tests from the rows below; public composition fixtures and host/JIT, synchronous/scheduled, single/vmap, current/legacy parity | D12 derivation and anchors; ownership here; construction/migration in README; approximation and unsupported compositions in LIMITATIONS |
| New IR field | `src/cffg/core/ir.py` (`Template` or its spec dataclass) | `src/cffg/core/validate.py`; `src/cffg/core/serialize.py` (`to_dict`, `from_dict`, `structure_hash`); `src/cffg/core/plates.py` scoping; `src/cffg/core/compose.py` and `src/cffg/core/edits.py` when editable; `src/cffg/topology.py` declaration snapshots/layout, `src/cffg/specialized/` semantic claims/drafts, `src/cffg/registered/` same-arena drafts, `src/cffg/resolution/` arrays/addresses, `src/cffg/assembly.py` arena assembly, and `src/cffg/compiled.py` values when executable; bundle and agent state when persisted | `tests/test_ir.py`, `tests/test_serialize.py`, `tests/test_plates.py`, `tests/test_topology.py`, `tests/test_registered.py`, `tests/test_resolution.py`, `tests/test_assembly.py`; add `tests/test_edit_codec.py`, `tests/test_edits.py`, `tests/test_agent_spec.py`, or `tests/test_bundle.py` only for the affected boundary | Contract in this document; public construction and migration in `README.md`; unsupported cases in `LIMITATIONS.md` |
| New factor or rule family | `src/cffg/kernels.py:FactorKernel`, `kernel_bindings.py`, its `src/cffg/core/ir.py:FactorType`, and the family rule module; formal equations are authoritative in code docstrings and `DERIVATIONS.md` | `topology.py` declared role/slot facts; `specialized/` named family discovery/completion; `registered/` binding capabilities and same-arena structural lowering; common declarations in `array_requirements.py`; `registered/packing.py` and `specialized/packing.py` family adapters; `resolution/` parameter/observation resolution; `assembly.py`, `Arena`, and `member_of`; typed local updates in `execution/` and `ExecutionDelta`; registered mathematics/readouts/energy in `kernel_runtime.py`; objective assembly in `ledger.py` and specialized local terms in `ledger_terms.py`; `execution/synchronous.py:make_sweep`; `execution/transport.py` and `schedule_program.py` batch/registerized/flat scans; `lifecycle.py` observation binding and `mu3_lifecycle.py` functional state; `ledger.make_belief_energy`; support checks, assignment scoring, `readouts/` posterior/statistic adapters, schedule classification, serialization manifest, estimator, and public exports as applicable | `tests/test_kernels.py`, `tests/test_execution.py`, `tests/test_registered.py`, `tests/test_resolution.py`, `tests/test_assembly.py`, `tests/test_discrete_kernels.py`, `tests/test_gaussian_kernels.py`, `tests/test_gaussian_kernel_schedules.py`; `tests/test_ledger.py`, `tests/test_readout_owners.py`; family rule and engine-oracle tests; `tests/test_schedule.py`, `tests/test_schedule_program.py`; sparse-support, float32, and uncomfortable-scale cases | Derivation in `DERIVATIONS.md`; integration contract here; approximations in `LIMITATIONS.md`; public modeling API in `README.md` |
| Runtime numerical or status invariant | `AGENTS.md` numerical contract and `src/cffg/runtime_types.py` (state/result/status values) and `src/cffg/iteration.py` (ordinary policy), `exact_strategy.py` and `outer_strategy.py` (distinct run control), and `run_results.py` (public results), plus `ledger.py` and `ledger_terms.py` (objectives) and `readouts/` (posteriors/statistics) | Flooding, colored/mixed, and exact schedules; host `runtime_checks.StateChecks` and `schedule_program.make_support_check`; `execution.synchronous.make_sweep`, `ScheduleProgram`, `make_exact_unit`, `make_propagate`, and `ledger.make_belief_energy`; `StateLifecycle`/`Mu3Lifecycle` initialization, preparation and status forwarding; inner-solver status; `RollingAgent`, `Colony`, estimator, benchmark, and dashboard consumers | `tests/test_runtime_status.py`, `tests/test_schedule.py`, `tests/test_schedule_program.py`, `tests/test_engine_contracts.py`, `tests/test_lifecycle.py`, `tests/test_iteration.py`, `tests/test_ledger.py`, `tests/test_readout_owners.py`, `tests/test_run_strategies.py`, `tests/test_colony.py`, `tests/test_colony_failures.py`, `tests/test_constraint_data.py`, `tests/test_constraint_reconstruction.py`, plus the changed rule family's hostile-input tests | Runtime contract here and in `README.md`; any accepted failure mode or approximation in `LIMITATIONS.md` |
| Learnable or checkpoint feature | `src/cffg/core/ir.py` (`LearnableSpec`, `AgentSpec`, `BeliefBinding`) and `src/cffg/learn.py` (`Family`, `Learnables`) | Validation, JSON/hash, plate scoping, specialized parameter/objective ownership, `resolution/addresses.py:resolve_learnables` and declared member/parameter requirements, `Arena.learn_plans`, `member_of`, `StateLifecycle` initial arrays and binding, `Mu3Lifecycle` functional parameter state, `readouts/` adapters and `StateLifecycle.with_goal_priors`, agent cadence, checkpoint encode/decode and legacy migration, and bundles | `tests/test_learnables.py`, `tests/test_readout_owners.py`, `tests/test_lifecycle.py`, `tests/test_constraint_reconstruction.py`, `tests/test_resolution.py`, `tests/test_assembly.py`, `tests/test_agent_spec.py`, `tests/test_agent.py`, and `tests/test_bundle.py`; add the contributing factor family's statistic test | Conjugate contract here; statistics beyond the papers in `DERIVATIONS.md`; public cadence/checkpoint API in `README.md`; remaining family gaps in `LIMITATIONS.md` |

Duplicated execution surfaces are deliberate review pairs: host/JIT,
flooding/scheduled, single-instance/`vmap`, and current/legacy checkpoint
decoding. A change that touches one side names the other in its ticket handoff,
even when the second side is checked and found not applicable.

### 1. Core IR and serialization (`cffg.core`)

A pure-Python (dataclass) representation, independent of JAX:

- `VariableSpec` supports discrete cardinalities, Gaussian dimensions, and
  scalar exponential-family tags. `FactorSpec` holds a type plus parameter
  references. Default normalization and marginalization are implicit;
  `ConstraintSpec` holds discrete δ, naive mean-field, endpoint moment-matching,
  and P-substitution beads. GFE composites are derived
  compiler plans detected from factors and constraints; they are not a separate
  serialized IR node. `moment_matching(factor, role, projection)` identifies
  one structural endpoint; Gaussian projection selects first and second
  moments. Role names remain local when templates are scoped, composed or
  expanded into plates. `migrate_moment_matching` requires the caller to
  supply the missing role and projection for a legacy factor-only payload;
  compilation and diagram generation continue to reject that legacy syntax.
  Gaussian bindings declare their supported `moment_projections`. For the
  analytic Gaussian prior and linear conditional, first/second-moment
  consistency is equivalent to full Gaussian density consistency, so their
  existing messages and node energies remain exact. Nonlinear projected
  messages and their objective still require their own numerical contract.
- **P-substitution declaration consistency.** `core/constraint_details.py`
  checks every declaration and every raw detail key before dictionary conversion
  can discard a conflicting value. Core validation preserves existing target/
  role diagnostics and then appends one sorted conflict report per target as
  `ModelError`. Serialization uses the same targeted check and raises
  `ValueError` before detail conversion or sorting; it does not perform general
  model validation. Bundle structure-hash checks also reach that guard.
  Identical repeated declarations and detail keys remain accepted and are not
  removed from the template. Equality means identical canonical JSON metadata,
  ignoring mapping-key order but preserving numeric representations such as
  `1` versus `1.0`; a missing key differs from an explicit value. Unknown fields
  participate so another declaration cannot silently discard future meaning.
  The schema and valid-model hashes are unchanged. Tests live in
  `tests/test_constraint_details.py`, alongside core/serialization/compiler and
  current bundle and current/legacy agent checks. This does not change other constraint families
  or the JSON parser's general duplicate-object-key behavior.
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
  memory mapping. The agent envelope records its version, exact field manifest,
  planning/filtering structure and static-parameter hashes, runtime
  planning/action configuration and construction-time JAX execution precision,
  named belief shapes, cadence, NumPy RNG, learnables, and learned-goal state.
  Restore rejects duplicate physical members and validates all fields into
  temporary values before committing; legacy unversioned analytic checkpoints
  retain an explicit migration path.
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
- **Portable bundle**: schema `"2"` is a ZIP (`.cffgz`) containing `model.json`,
  `blobs/`, `learning/`, and `manifest.json`. Template JSON keeps its independent
  schema version. The manifest owns the complete point inventory, NPY hashes,
  exact array shape/dtype/value digests, and the family-independent learning
  inventory. Each learning member uses
  `learning/<percent-encoded-owner>/<prior|posterior>/<percent-encoded-key>.npy`;
  owner and key are encoded independently, including scoped slash characters.
  Blob members use reversible per-segment encoding under `blobs/`, a separate
  namespace. Plate fragments stay compact in `model.json`; grounding resolves
  the `{index}` marker in per-copy blob paths and enumerates every concrete
  learnable and point array. Compiler plans reconstruct logical ownership;
  no arena address or executable callback is serialized.
  `load_bundle` returns a `LoadedBundle` with template, point arrays, manifest,
  and reconstructed `Learnables`. The learner owns read-only immutable priors
  and current posteriors; the caller owns the supplied point arrays. An empty
  declared-learning inventory is explicit. Loading verifies unique safe ZIP
  members, exhaustive inventories, all hashes, compiler shapes and both family
  states before returning. Saving copies state under the learner's update lock,
  validates the exact archive bytes, then fsyncs a sibling and replaces the
  destination atomically. Direct edits to caller-owned mappings require caller
  synchronization. This is a committed learning boundary, not executing state.
  A fit result retains the original template and an independent complete
  `Learnables` snapshot of its returned coordinate. Its `save_bundle` method
  combines fixed initialization arrays with fitted point arrays. On resume,
  `Learnables.write` installs the posterior's inference parameters in a fresh
  State; these packed parameters remain derived values, not ZIP blob keys.
  Reporting arrays and inference rollover values have distinct owners in the
  family API. `test_bundle_resume.py` compares both plus the next posterior
  and installed state against uninterrupted learning for all registered
  families, accumulation/replacement, float32, tied sources, sparse statistics,
  and 64 scoped plate owners. `test_bundle.py` regenerates the representative
  rSLDS archive with the complete 22-array learning inventory.
  The supported inference path remains `load_bundle` → `compile_template` →
  `Engine`; extracted point blobs also work with `root=extracted_directory`.
  During alpha only the current bundle schema is accepted. Older versions
  fail explicitly; no point estimate supplies missing posterior provenance.
  A stable compatibility and migration contract remains future work. No
  third-party interchange format is implied.
- **Edit actions**: `check_edits` validates a sequence of typed actions and
  `apply_edits` commits it atomically to a copy. `AddVariable`, `AddFactor`,
  `RemoveFactor`, `RemoveVariable`, `ReplaceFactor`, `AddConstraint`,
  `RemoveConstraint`, `DeclarePort`, `RemovePort`, `AddBlob`,
  `RebindBlobPath`, and `SetPlateDims` are the executable action vocabulary;
  `bundle_to_json` / `bundle_from_json` serialize the sequence. `compose`
  remains the separate complete-template composition operation. No public
  `add_slice` or template-level `bind_data` function is advertised.
- **Supported facade**: top-level `cffg` re-exports only `CorridorEnv`, `Engine`,
  `RollingAgent`, `RunStatus`, `build_gridworld`, `compile_template`,
  `load_bundle`, `save_bundle`, and the existing `set_params` compatibility
  helper. Their focused modules remain canonical owners. Before 1.0, rule
  tables, compiled arenas/group plans, kernel registration objects, and private
  helpers are internal-unstable; serialized schemas retain their explicit
  version contracts. The facade resolves exports on first access, so importing
  `cffg.core` does not import JAX or Equinox. Fresh-process tests forbid those
  runtime imports while validating and round-tripping a template.

### 2. Compiler (`cffg.compiler`)

`cffg.compiled` owns compiled plan/address records, `Arena`, and the unchanged
log-space sentinel. Compiler orchestration constructs these values; runtime,
scheduling, learning, and parameter consumers import them from that lower-level
owner. The old `cffg.compiler` imports remain compatibility re-exports. The
compiled-value owner imports neither compiler orchestration nor runtime code;
its dependencies are NumPy and the host-side kernel declarations.

Published arenas and plans are immutable shared blueprints. Their constructors
snapshot mutable input arrays, mappings, and sequences without changing the
caller's objects. Stored mappings are read-only, sequences are tuples, and
numeric fields use the internal `FrozenArray` value. Field names, numeric
dtypes, shapes, element order, and serialized identities are unchanged; the
container representation intentionally differs from the characterization
baseline. Kernel binding declarations follow the same publication policy.

`FrozenArray` owns private numeric storage. Its shape and dtype cannot be
reassigned, and its NumPy conversions return detached working arrays; requesting
a zero-copy NumPy export is rejected. This separation is necessary because
NumPy's `ufunc.at` can bypass the read-only flag, including through NumPy views
of CPU JAX arrays. JAX conversion explicitly copies as well. Reusing a published
value in another plan, arena, or `dataclasses.replace` reuses its stored payload;
it does not republish a copy. Indexing selects from private storage before any
conversion, so reading a slot does not copy the complete graph's prefix table.
The value owner does not import JAX at module load; conversion imports it lazily.

Runtime consumers explicitly convert index values at the JAX boundary. Numerical
message updates and free-energy evaluation remain JAX computations; compiler
construction and host address calculations remain NumPy operations. There is no
per-iteration arena freezing. Host array exports have a copy cost, and initial
publication adds allocation and construction work; immutability itself is not an
inference speedup. The benefit is that captured JIT metadata, schedules,
readouts, and parallel instances cannot accidentally observe different versions
of a shared plan.

Mutable compiler builders remain private. Structure changes require a new
template/compilation; observations, messages, and learned parameters still update
through per-instance `State`. `Arena.log_params` contains initial parameter
values, not a prohibition on learning. Kernel handles preserve registered
identity, including when copied; arbitrary callback closure state is trusted,
not recursively frozen. This ownership contract is not a security sandbox.
`dataclasses.asdict` supports inspection but retains read-only mapping/array
leaves; it is not a model or checkpoint serializer. Templates plus data remain
the reconstruction authority. `tests/test_compiled.py` covers publication,
alias isolation, copy behavior, import direction, and distinct vmapped states.

`cffg.topology.build_topology` accepts the expanded, core-validated template
and returns one immutable `TopologyLayout`. It snapshots variable, factor-role,
constraint, and port declarations; records ordered endpoints, degrees,
ports-by-variable, raw edges, own/opposite slots and DATA candidates; and
records discrete/Gaussian segment lengths. Factor role mappings use the
read-only `FactorDeclaration` snapshot, not mutable core `FactorSpec` mappings.
The three logical slot domains
retain independent numbering. Endpoint order sorts factor/role names, while
declared message-role order and numeric hard-constraint role order remain
separate conventions in lowering.

Topology does not classify registered or specialized factors, infer a parameter
edge's table shape/axis, decide legal observation conditioning, or assign
observation rows. Parameter carriers already declare their flat cardinality;
semantic discovery produces new tagged edge records without changing topology.
Observation rows follow lowered group/member order, not DATA-candidate order.
Resolution converts these lengths into int64 prefixes at the existing late gate,
after array validation and before learning validation, so out-of-range declarations
retain their original error precedence. It also resolves observation prefixes;
assembly consumes the prefixes and specialized entropy exclusions. The
module's `prefix_offsets` helper calculates offsets from supplied segment lengths;
it does not choose observation assignments or interpret factors.

`compile_template` still owns semiring argument checking, plate expansion, and
core `validate`. Core malformed-incidence failures remain `ModelError`.
Topology returns executable-port diagnostics in declaration order, and compiler
orchestration appends them after factor-admission errors at the existing
`CompileError` gate. Conditioning errors retain their later gate after array
resolution and Dirichlet discovery. Moving structural work does not change the
error owner or precedence. `tests/test_topology.py` directly protects these
boundaries and the immutable declaration snapshot.

#### Specialized structural lowering

`src/cffg/specialized/` owns factor semantics that do not fit the ordinary
same-arena registered-kernel contract. It is an explicit set of lowerers, not a
fallback for unrecognized factor types. Its two operations are:

1. `discover_specialized(Template, TopologyLayout, semiring, RegistrySnapshot)` returns immutable
   `SpecializedClaims`: parameter-edge shape/axis tags, composite membership,
   absorbed factors, equality routes, specialized conditioning restrictions,
   and deferred diagnostics. It never reads a blob. Raw constraint details are
   used transiently for the existing diagnostic text; mutable template objects
   are not retained.
2. `complete_specialized(claims, OrdinaryMembership)` returns immutable
   `SpecializedPlanDrafts`. Ordinary categorical BP membership, including
   structural equalities, completes recurrence components without classifying
   their factors again. Registered lowering supplies this membership after
   consuming the specialized claims.

The specialized owners are:

| Responsibility | Owner |
| --- | --- |
| Admission traversal, parameter-edge tags, composite claims, domain routing, specialized initializer shapes | `specialized/discovery.py`; registered capability checks delegate to `registered/capabilities.py` at their original per-factor positions |
| Dirichlet parameter paths, categorical and transition GFE, hybrid GFE, functional μ3 metadata | Named `lower_*` functions in `specialized/composites.py` |
| Mixture, recurrence, and Gaussian precision plans | Named `lower_*` functions in `specialized/mixed.py`; recurrence incidence classification consumes immutable variable/endpoint mappings in `recurrence_topology.py` |
| Hard-constraint drafts and family-owned k/table validation | `specialized/constraints.py` |
| Equality plans, edge DELTA meaning, final semiring restrictions | `specialized/structural.py` |
| Global learned-goal parameter plan | Completion delegates to `goal_parameters.build_goal_parameter_plans`; runtime count pooling and energy remain in `goal_parameters.py` |
| Family array validation and representation changes | Named packers in `specialized/packing.py`; hard-constraint payloads remain in `specialized/constraints.py` |
| Cross-family completion and original-factor/objective ownership map | `specialized/__init__.py`; value contracts in `specialized/contracts.py` |

Drafts reuse existing immutable compiled records when every field is structural.
They are the same objects passed into `Arena`, without republishing their array
payloads. `ArrayRequirement` declares ordered factor/role/blob inputs, initializer
shapes, a named family recipe, and parameter-key or constraint-payload
destinations. Recipes are identifiers, not serialized callables. Compatibility
packers consume these declared inputs and destinations. A `ConstraintDraft`
deliberately has no k/table value until its family validator resolves the payload.
Generic array resolution does not choose a factor's mathematical meaning.
Specialized composite shapes come from their family declarations, not an ordinary
kernel's shape callback. Replacing a built-in kernel must not cause its callback
to run earlier; registered shape providers retain their initializer/packing gates.

`CompositeOwnership` connects original factors and constraints to their plan,
absorbed members, ordinary-entropy exclusions, parameter edges, and shared
parameter/objective owners. Fixed categorical goals remain explicit in this map
even though the existing `GfePlan.goal_members` compatibility field is empty.
μ3 and global learned-goal owners are overlays, not duplicate factor execution.
Transition GFE absorbs no facing factor and retains its output edge entropy;
categorical/hybrid observation GFE excludes only the substituted output.
Dirichlet carriers are excluded through their parameter tags, separately from
GFE output exclusions. Specialized claimed factors and ordinary registered
factors form a disjoint, exhaustive partition.

Discovery/completion happen before blob resolution, but error publication keeps
the existing checkpoints: admission, parameter layout, conditioning and prior
arrays, composite declarations/arrays, remaining-factor grouping, family arrays,
hard constraints, semiring, DELTA, and assembly. Invalid concentrations still
produce the original orphan-prior diagnostic. Explicit constraint diagnostics
become `CompileError`; incidental array-library exceptions retain their original
class. Numerical packer mathematics now lives beside its family owners;
`resolution/` sequences source loading, validation, observation/member addresses,
and learning. Final assembly lives in `assembly.py`. No numerical update, approximation,
saved-model schema, supported combination, or public compiled-plan field changes.

`tests/test_specialized.py` protects these boundaries, ownership distinctions,
immutable snapshots, unresolved constraint payloads, registered `any`-domain
routing, diagnostic timing, and 32 specialized shape groups under float32 and
serialized reconstruction. Existing family tests remain the mathematical
authorities. The explicit corrected behavior baseline is `ad028f0`, following
the global learned-goal and conflicting-declaration fixes.

#### Registered structural lowering

`registered/lowering.py:lower_registered` accepts the expanded `Template`,
`TopologyLayout`, `SpecializedClaims`, semiring, and a `RegistrySnapshot`.
It returns immutable `RegisteredPlanDrafts`: ordered discrete, Gaussian, and
EF group drafts, ordinary BP membership, discrete DATA slot intent, and deferred
conditioning/grouping diagnostics. This stage reads declarations, never blobs.
`registered/capabilities.py` captures the binding map once per compilation and
owns registered constraint-mode and conditioned-output admission checks. Discovery
and registered lowering share that snapshot; trusted callbacks retain identity,
not frozen closure state, and are not serialized into templates.

Discrete drafts reuse complete `GroupPlan` records. Gaussian and EF drafts carry
roles, dimensions/family, member order, gather/scatter arrays, and the frozen
kernel handle containing message/energy/readout and implementation identity.
Observation intent records an observed role and member-ordered source references,
not row IDs. Unobserved is distinct from observed-but-unresolved; an invalid
conditioning source retains a missing port until its existing diagnostic or
row-resolution gate. It cannot publish a row under a missing port key.
`resolution/observations.py` assigns the independent Gaussian and EF row counters and
publishes final plans without copying already immutable address arrays.

Specialized claims are authoritative exclusions. Equality is not reinterpreted:
opaque equality references stay interleaved in original group-key order, and
specialized completion supplies the plans. Discrete equality members contribute
to BP membership for recurrence completion. Numeric shape keys, not displayed
group-name strings, determine ordering. Existing `family="any"` declared-domain
routing is retained for non-equality registered extensions.

`array_requirements.py` is the shared declaration contract. Registered inputs
record factor/parameter-role/blob identity and an unevaluated `ShapeRequirement`
(trusted `ParameterSpec` plus shared immutable role sizes); destinations record
the existing packed parameter paths. `SourceRequirement` records source roles
for every referenced factor, including absorbed members, in original traversal
order; covariance checks and historical initializer fallbacks remain distinct
from canonical packing shapes. `MemberRequirement` and `ParameterRuntime`
declare row/write eligibility, transforms, and learning form/layout without
interpreting runtime key suffixes in the learning resolver. Specialized drafts
also retain prior claims, deferred diagnostics, edge tags and initializer fallback
facts needed after discovery. `resolution/addresses.py` materializes the declared
addresses and binds learning sources/targets; `resolution/resolve.py` owns
the integrated numeric/address checkpoints. `compiler_errors.py` supplies their
shared `CompileError`, reexported by the public compiler facade.

External shape callbacks still execute only
at initializer or packing checkpoints. Registered packers retain their complete
resolved-blob mapping and original callback/key-validation behavior, including
the legacy EF adapter. Requirements do not narrow this callback input contract.
`registered/packing.py` owns the adapters that invoke these callbacks and the
existing abstract-shape checks in `kernel_runtime.py`.

`tests/test_registered.py` checks draft reuse, opaque equality routes, observation
intent, snapshot isolation, shared registry input, callback timing/input mappings,
ordered admission and late conditioning errors, and 32 ordered groups with
float32 source arrays and JSON reconstruction. Existing registered-factor
enumeration and Gaussian covariance oracles remain the mathematical authorities.

#### Array and learning resolution

`resolution/resolve.py:resolve_compile_data` accepts the expanded `Template`,
array overrides/root, `TopologyLayout`, and complete registered/specialized
drafts. It returns one immutable `ResolvedCompileData`, not an `Arena` or a
schedule. This value contains parameter arrays, final ordinary plans,
observation/member addresses, writable-member facts, validated hard constraints,
prefix arrays, learning plans, and the unchanged functional μ3/global-goal plans.
The compiler facade reuses its published maps, arrays, and plans by identity.

`resolution/sources.py` owns override → existing file → initializer precedence.
It consumes the ordered source catalog, invokes declared shape providers only
when an initializer needs them, and checks covariance in the source dtype before
float64 host packing. Canonical composite packing shapes do not replace the
historical first-role initializer fallback. Registered packers see every resolved
model blob; specialized packers consume their declared inputs. Family owners
retain probability-to-log conversion, concentration interpretation, covariance
inversion, and constraint-payload meaning; the resolver chooses none of these
from factor names or runtime key suffixes.

Private builders remain mutable through the ordered validation pass. Existing
checkpoints run from source loading through conditioning/prior checks, composite
packing, registered and mixed-domain groups, constraints, semiring/DELTA checks,
observation lengths, and global-goal validation. Prefix conversion precedes
learning validation; immutable numeric publication follows it. This preserves
both overflow-versus-learning failures and the legacy EF callback's late
object-array rejection. `resolution/addresses.py` consumes declared write and
statistic-source eligibility plus parameter transforms and learning layouts.
No learning cadence, inference rule, or serialized schema changes here.

Final edge tagging, interior entropy-ledger tables, and arena construction
live in `assembly.py`. Assembly consumes the resolved prefixes without
calculating them again. `tests/test_resolution.py` checks complete data/address
parity across the compiler characterization models, failure ordering, and reuse
without mutable aliases; existing family and learning tests remain the numerical
authorities.

#### Final arena assembly

`assembly.py:assemble_arena` accepts topology, registered and specialized drafts,
resolved compile data, and the chosen semiring. It joins the declared parameter
edge tags with raw topology, groups Gaussian/EF interior-edge pairs for the
entropy ledger, and constructs the immutable `Arena`. It needs no `Template`,
source files, initializer, numerical packer, or registry lookup. Published
parameters, prefixes, and plans are reused, not copied. Discrete interior groups
and schedule fingerprints remain derived by their existing consumers.

`assembly_checks.py` checks the cross-stage joins before publication. Every
original factor must have exactly one execution owner, either ordinary or
specialized, including absorbed goals and priors. Equality references bind to
specialized plans without counting those factors twice. Ownership is joined by
original member identity, not a globally unique plan name. Declared entropy
exclusions, parameter-edge tags, and attached constraints must agree with their
owners. Resolved plan addresses and functional/shared-goal owners must retain
their published references; prefix lengths must match their logical slot counts.

Parameter-message/objective owners are separate from factor execution. Shared
goal contributor rows and carrier slots must agree with the completed GFE plans,
and each original contribution appears once; several groups may reference the
same shared owner. Functional μ3 rows must match their declared members. These
checks inspect completed declarations, never choose a factor's rule, rederive
its energy, or compare numeric parameter values to infer sharing. Mismatches
raise `CompileError`. Indexes are built once per assembly; no whole-template
scan is repeated for each plan or member.

`compiler.py:compile_template` checks its public semiring argument, expands
plates, runs core validation, captures one registry, and sequences topology,
specialized discovery/admission, registered lowering, specialized completion,
data resolution, and assembly. `specialized.validate_admission` publishes the
existing admission/layout errors at their original checkpoint. Lower-level
owners import compiled values and `CompileError` directly; none imports this
public facade. Its compiled-value exports and used private compatibility
helpers remain available.

`tests/test_assembly.py` checks direct assembly after caller templates/arrays are
discarded, rejects accidental source/packing/prefix callbacks, verifies storage
reuse, and confirms the facade delegates final construction once. Plan names may
repeat across execution-family namespaces; they are not globally unique IDs.
Negative cases remove or duplicate owners and corrupt exclusions, parameter
references, shared-goal carriers, μ3 rows, prefixes, and resolved group membership.

The compiler characterization baseline is `bf6bbe9527b5e1eea9172a30b8c7c214a08ed9ee`,
the completed discrete/Gaussian registration foundation. Compiler extraction
preserves its public plans, parameter addresses, slot order, diagnostics, and
serialized identities. The baseline's ownership is concentrated in these blocks
of `src/cffg/compiler.py` (line numbers refer to that commit):

| Responsibility | Baseline owner | Interpretation to consolidate during extraction |
| --- | --- | --- |
| Public entry | `compile_template`, 646–665 | Semiring argument, plate expansion, and core validation stay at the public boundary. |
| Topology and slot layout | `compile_template`, 666–681, 801–886, 1898–1936 | Degree, incidence, endpoint order, and prefix offsets describe declared structure; conditioning and parameter-edge semantic tags currently enter the same pass. |
| Registered lowering | Remaining-factor classification, 1417–1547 and 1703–1778; `_stack_params`, `_stack_gauss_params` | Registry roles, shapes, conditioning, packing, and same-arena classification recur in admission checks, grouping, and array-shape inference. |
| Specialized lowering | Constraint admission and parameter tags, 677–799; composite discovery/construction, 894–1415; mixed/recurrence/precision plans, 1548–1701; hard constraints, 1780–1854 | Composite claims must precede ordinary grouping. Goal and prior absorption, inferred parameter-edge meaning, and objective ownership cannot be independently rediscovered by each stage. |
| Learning addresses | `_set_member_address`, `_resolve_learnables` | Group construction records parameter roles and transforms; learning resolution interprets those addresses and rejects unsupported source/target pairings. |
| Array resolution | `_resolve_arrays`, `_blob_shape`, `_load_blob`, `_stack_*` | Override/file/init precedence belongs to resolution. Registered shape declarations and specialized shape rules must agree with their lowering owners. |
| Final assembly | `compile_template`, 1856–1972 | Semiring restrictions, delta checks, entropy-edge membership, and array offsets currently finish beside `Arena` construction; assembly must eventually consume validated results, not repeat semantic decisions. |

Schedule construction is a consumer in `cffg.schedule`, not another compiler.
Flooding, scheduled stages, and batched execution use the same arena addresses.
Model JSON and bundles reconstruct the declared template before compilation.
Current and supported legacy agent checkpoints restore runtime data against an
already compiled, builder-defined model. There is no separate host/JIT compiler
pair.
`tests/test_compiler_contracts.py` pins these addresses and identities with
small existing model builders, exact diagnostics at their owning boundary,
float32/float64 source comparisons, and 32 distinct shape groups. The existing
schedule, kernel, learning, AgentSpec, and checkpoint tests retain the runtime
and compatibility proofs; characterization does not replace them.

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
not a second pair or energy owner. The sweep, BFE, and public readout
deterministically rematerialize the pair from current cavities and drive
moments; no pair array is independently persisted.

`Engine.recurrence_pair_beliefs` exposes the normalized factor-local pair with
source rows and destination columns, both marginals, literal support, direct
Gaussian drive moments, and propriety. `gaussian_node_stats` includes the same
record. The direct covariance prevents recurrence learning from recovering a
small covariance by subtracting two large, nearly equal moments.
`Engine.recurrence_weight_stats` derives its at-risk, signed-success, linear,
and quadratic rows from that pair: it checks factor-local probability,
covariance, parameter, logit-second, and finite-output predicates and emits
additive Gaussian natural statistics for augmented `[R; r]`. The Gaussian
message and coefficient learning therefore share D9.12's masses without a
singleton-product fallback. Its diagnostic scalars identify the failed
predicate and its dtype-scaled bound. The generic
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
runtime and leave the historical fixed emission unchanged. A freshly
constructed facade labels its recurrence block as an unlearned prior;
`update`, fitted posteriors, and checkpoint restore carry the current D9
structured-pair statistic identity. `fit_rslds` owns a
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

D11's singleton-boundary candidate fails to convey externally inferred
regime/state association. D12 accepts the explicitly approximate
`hybrid-regime-ec-v1` mathematical contract: alternating recurrence/dynamics
regions share hybrid regime-conditioned moments, with an equivalent grouped
Gaussian evaluator and explicit identity adapters for ordinary ports. Each
region owns its labeled entropy and factor energies; hybrid and ordinary
separator corrections follow actual declared incidences. Signed quotient
messages require proper aggregate and tilted beliefs. The temporal
counterexample in D12.7 limits the exactness claim even at zero EC residual.

Production remains D9. `experiments/hybrid_separator.py` is an unbound NumPy
derivation anchor, not a second runtime route. The Derived component coordinate
row inventories the implementation: cff-k6g5 owns serialized declarations and
identity; cff-ffsg owns local JAX mathematics; cff-7sw5 owns bounded compiler
plans; cff-f3jz owns execution, joint solver status, and the ledger; cff-czdq owns
learning/readout/persistence semantics; cff-kul2 owns public composition and
execution parity. cff-ulbz supplies the explicitly selected D11.14 EC binding
where exterior mixture ports remain; internal affine switching dynamics uses
D12.3's elimination and does not need a D3 port. Existing D3 collapse-first
messages are not assumed stationary for EC. No production selection, API,
learned-statistic meaning, or checkpoint version changed in this derivation.

### 2e. Executable factor registration (`cffg.kernels`)

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

Each kernel contains exactly one typed arena binding in `kernel_bindings.py`.
`EFBinding` preserves the original scalar natural-parameter contract and its
legacy constructor. `DiscreteBinding` owns ordinary `categorical`,
`matrix_cat`, and `transition_mixture` execution, including mean-field rules.
Its ordered roles get cardinalities from the template. BP consumes padded
opposite-side cavities; VMP consumes normalized `(q, log_q)` per role from
both facing messages. Mode-specific message, node-energy and optional
readout callbacks use the same inputs and named parameter snapshot.

`GaussianBinding` owns Gaussian priors and ordinary/conditioned linear
factors. Callbacks receive local information-form blocks, named parameters,
ordered dimensions, and the optional observed-role index/value/mask. Every
structural role is retained, including the cavity predictive on a DATA
role. `GaussianConditioning` declares the legal role and its host validator;
the internal `gaussian_linear_obs` label is only a lowering of the serialized
`gaussian_linear` identity. External conditioning may use any declared role,
not necessarily the first. For missing data, callbacks marginalize that role:
the linear likelihood retains its negative input-belief entropy term, while
a joint Gaussian can retain a nonconstant marginal factor.

Gaussian `statistics` callbacks feed `gaussian_node_stats`; optional `readout`
callbacks feed `kernel_readouts`. Host sufficient-statistic evaluation keeps
its existing operation order and learning schemas, including cross-covariance,
propriety and missing-row masks. It is separate from JAX message/energy
callbacks; registration does not imply a new conjugate-learning family.

`FactorSpec.function` is an optional, immutable `FunctionSpec`: a namespaced
mean implementation, required semantic version, JSON configuration, and
`linearization` or `sigma_point` method. It contains no arrays or executable
objects. Core validation, editing, JSON, hashing, scoping, and plates preserve
it without importing JAX. `mean_functions.py` resolves its host registration;
the installed version must share its major version and meet its minimum
version. Missing implementations fail on loading. Numeric parameters and
positive noise covariance retain ordinary blob ownership and bundle storage.

`GaussianBinding.configure` specializes a binding from this declaration and
incident dimensions before source-shape inference. Registry snapshots retain
the resulting binding by factor name. Grouping includes the canonical function
declaration; members with equal configuration share a group while their
`theta` and noise precision `W` retain separate leading rows. Configuration
changes therefore cannot alias parameter addresses. Named writes use existing
parameter ownership and covariance validation. `nonlinear.py` owns both local
approximations, tilted moments, replacement sites, and normalized node energy
(DERIVATIONS D13); `nonlinear_kernels.py` owns configured shape/packing and
batched callbacks. The nonlinear factor's structural roles are `out,in` and
its numeric parameter roles are `parameters,cov`. No conjugate-learning family
is implied by the statistic callback.

`ParameterSpec` supplies serialized shape inference and optional writable
packed-row metadata. The packer declares every output key; the compiler
checks keys, shapes and member axes. Existing built-ins keep their bare
group-name parameter addresses and transition-table axis order. External
keys normally occupy `group:key` addresses. Nonlinear/joint packs can omit
direct-write addresses and reconstruct through compilation. Internal plan
aliases and duplicate parameter addresses are rejected before runtime values
can be overwritten. Declared covariance parameters use the named-write
validator in their source dtype before the compiler widens them for packing.
The declared
log-potential accessor owns ordinary discrete support checks, BP joint
beliefs and decoded assignment scoring. Max-product does not report BFE.

`runtime_types.py` owns pure State, result and termination-code declarations.
`engine.py` preserves its existing public type exports, and `outer.py`
re-exports the same `OuterTerminationReason` enum. This leaf imports no engine
or numerical solver. Field schemas, enum values and the top-level facade's
export list are unchanged; checkpoint data remains arrays and explicit metadata,
not serialized Python classes.

`iteration_types.py` defines the internal `IterationOutcome`, a typed reason
and fixed-shape `StepFacts`/`TerminationFacts` array records. An unchecked
finite/support predicate is distinct from a false predicate. Outcome traces
are immutable host arrays that retain Python-float precision even when JAX
x64 is disabled; the public boundary converts them to Python tuples.

`iteration.py` owns ordinary soft/hard iteration and the shared incremental
`IterationProgress` assessment. It consumes prepared State, `ScheduleProgram`,
explicit policy/controls and narrow check/objective/functional-update callbacks;
it does not inspect factor types or import Engine. Engine supplies current
step/prime handles and callbacks at run entry. Colony retains its vmapped
transport and per-instance active masks, but uses the same assessment owner.
Initial rejection precedes priming, stopped batch members remain frozen, and
FE comparison history resets between soft and hard phases.

`run_results.py` is the public status/result authority for all three run
strategies. Its `finalize_iteration`, `finalize_exact` and `finalize_outer`
functions validate their distinct terminal evidence and construct `RunResult`.
Ordinary initial rejection retains `NOT_RUN`, including in explicit
legacy-Picard GFE calls; valid zero-budget and later Picard results retain
their `LEGACY_PICARD` tag. The host convergence loop is not newly jitted.
`test_iteration.py` protects the owner boundary, immutable/precision-preserving
traces, transformable facts and shared finalization; existing family and
failure tests remain the numerical authorities.

`exact_strategy.py:ExactRunStrategy` consumes only the fused exact schedule
unit, objective mode, finite-state predicate and objective callback. It returns
`ExactOutcome`: a logical execution count, immutable host traces and certificate
facts. Exact support rejection after propagation counts one execution even
though its delta trace is empty. A structural certificate can prove convergence
with `tol=None`; outer control intentionally retains its different meaning.

`outer_strategy.py:OuterTrustRegionStrategy` owns soft/hard phase budgets,
functional-coordinate cadence and accumulated solve diagnostics. It receives
`OuterStructure` (edges, shared-goal plans and segment offsets, without arena
parameters), hardening rows, eliminated slots, current schedule handles and
check/ledger/functional callbacks. It returns `OuterOutcome`, not a public
status. The numerical chart and trust-region algorithms remain in `outer.py`.

`outer_kernels.py:OuterKernelProvider` owns a per-engine lazy derivative cache.
Its keys retain chart support, dtype and reference metrics plus hard-message
index dtypes, shapes and bytes. Chart bases, observations, masks, current
parameters and hard values are passed separately for every solve. The provider
does not capture an Engine; its schedule handles are refreshed through the
facade's existing instrumentation aliases. Cache creation timing and reported
compilation counts are unchanged. These trusted process-local callables are
derived runtime artifacts, never serialized model or checkpoint data.

`ledger.py:ObjectiveLedger` owns the two distinct objective operations.
`free_energy` retains the host checks in their original order: reject
max-product mode, require prepared functional state, reject proven empty
support with infinite FE, then evaluate the raw belief kernel. Engine supplies
its current check/readiness/evaluator handles, preserving instrumentation.
`assignment_energy` receives an already decoded discrete assignment and current
parameters; it neither decodes beliefs nor performs message updates.

`ledger.py:make_belief_energy` supplies the same raw BFE/GFE computation to
fused exact schedules, ordinary and outer convergence checks, and direct JAX
evaluation. `engine._make_bfe` remains a compatibility alias. The raw kernel
does not replace the host support/readiness checks; exact schedules retain
their fused support classification. `ledger_terms.py` gives each existing
specialized local-energy family an explicit adapter. Registered same-arena
terms still use their declared energy callbacks; legacy EF callback lookup
retains its trace-time behavior.

Local additions retain their order. The local GFE adapter recomputes predictive
statistics at the evaluated state, adds one term per shared learned-goal owner,
and adds one KL contribution per fixed functional parameter group. Interior
entropy corrections and P-substituted exclusions keep their compiled ownership
and multiplicity. Discrete delta hardening changes those beliefs; this move
does not add a separate delta energy. Mixture and recurrence approximation
labels, current-instance parameter/observation inputs, and objective formulas
are unchanged. Neither ledger module imports run-control or public agent code.

`readouts/` owns posterior and statistic adapters over current State and typed
read-only structural views. These protocols expose only the needed compiled
plans and addressing methods; Arena implements them without per-call copies.
The shared transport helpers accept the same message, Gaussian and observation
layout protocols, so readouts do not duplicate segment-index arithmetic.

| Readout surface | Owner |
| --- | --- |
| Categorical edge marginals, BP/VMP node beliefs and decoding | `readouts/discrete.py` |
| Dirichlet parameter/goal beliefs and next-trial prior readout | `readouts/parameters.py` |
| Gaussian marginals, node statistics and mixture predictives | `readouts/gaussian.py` |
| Recurrence pair beliefs and coefficient-learning statistics | `readouts/recurrence.py`, using the existing `cffg.recurrence` algebra |
| Exponential-family edge marginals | `readouts/exponential_family.py` |
| Hybrid GFE diagnostic channels | `readouts/hybrid.py`, using the existing D5 diagnostic rules |
| Optional declared kernel readouts | `readouts/registered.py` |
| Public readout value records and named readout dependencies | `readouts/types.py` |

Engine retains its public signatures, docstrings and value-type import aliases.
Its readout methods are delegates, with current public readout callbacks passed
where one readout depends on another. Goal readouts still use
`dirichlet_marginals`; Gaussian/recurrence learning statistics still use
`recurrence_pair_beliefs`; decoding still uses `marginals`. This preserves call
order and instrumentation without capturing an Engine inside a readout owner.
Existing ordinary/legacy EF callback lookup timing, field order, dtype, support
and improper-belief outputs are unchanged. No readout vectorization or caching
is introduced; `cff-3kdk` remains that separate performance work.

`with_goal_priors` is a state replacement, not a readout. Its implementation now
lives in `StateLifecycle`, retaining the fresh-trial count guard, prior-shape
and positivity checks, dtype conversion and atomic parameter replacement.
Functional diagnostics and epoch coordination remain in `Mu3Lifecycle`; they
are not duplicated by the readout extraction. Engine now only constructs and
connects owners, validates public controls, selects a run strategy and forwards
results. It implements no family rule, schedule algorithm, solver loop or
local objective/readout calculation. Compatibility exports for existing
transport helpers remain available alongside `_make_bfe`.

`lifecycle.py:StateLifecycle` owns initial array construction and DATA-port
binding over the immutable arena. It converts defaults at each call, so the
active JAX precision setting still determines newly initialized dtypes.
Binding returns a replacement State only after every observation succeeds;
`None` keeps the existing categorical-uniform or continuous-mask behavior.
Gaussian validators come from compiled bindings, while EF validator lookup
retains its existing bind-time behavior.

`mu3_lifecycle.py:Mu3Lifecycle` separately owns functional parameter defaults,
particle preparation, readiness checks, moment installation and diagnostics.
It receives immutable functional plans and latent-message indices, not the
arena. Its numerical State view contains only messages and parameters, and
current particle-budget/ESS settings are supplied at each call. Ordinary
models have no functional lifecycle instance. Epoch scoring carries complete
States opaquely through supplied preparation and run callbacks; the run
controller remains the authority for convergence. `Engine` preserves the
public methods as delegates. PRNG split order, parameter keys, observation
masks and checkpoint schemas are unchanged. `test_lifecycle.py` protects these
boundaries, precision/settings timing, atomic binding and epoch failure forwarding.

`execution/registered.py` owns the typed local-update boundary for registered
discrete, Gaussian and exponential-family plans. Flooding, batch stages,
registerized scans and fallback scans all use these owners, including the
exponential-family equality and observed-message calls. Inputs are local
blocks and selected numeric parameters/observations, never a whole arena or
Engine. Transport retains its existing gather, prefetched carry and scatter.
`ExecutionDelta` is a JAX-compatible immutable record of outgoing blocks and
optional numerical solver facts; absent facts are not a convergence claim.
Exact support stays encoded in the outgoing blocks for downstream checks.
These registered updates have no local iterative solver diagnostics.

`execution/synchronous.py:make_sweep` owns the global synchronous transport.
It gathers every local input from the pre-sweep arrays, collects cross-domain
writes, pools shared-goal predictions once, and applies the existing damping,
log-zero flooring and delta hardening. The full and restricted mixed-phase
sweeps use the same constructor. Local factor calculations and parameter
selection remain with their existing execution owners; current constraint
values still come from `State.params`. `Engine` injects this factory into the
schedule program and contains no synchronous transport implementation.

`schedule.py` derives the deterministic declarative schedule. The independent
`schedule_program.py` owner validates the selected schedule and builds a
frozen `ScheduleProgram` record: flooding/iterative updates, optional priming and
forest reconstruction, directed propagation, and exact certification. The
program consumes the typed `ScheduledExecutors` collection plus the existing
`execution.synchronous.make_sweep` factory and free-energy callback. It owns ordering, damping
composition and transport, not parameter meanings, factor rules, ledger
formulas, outer elimination policy or public termination decisions.
`schedule_contracts.py` names those callable inputs and outputs. They are
trusted host handles, never serialized model or runtime-State data.

`execution/transport.py` supplies shared shape-local gathers and scatters.
Registerized scans still prefetch inputs, carry only cross-step local blocks,
and scatter once after the scan; flat scans remain the fallback. EF parameter
prefetch uses the same selector as flooding. The execution support owner
evaluates factor support from local inputs; the schedule support check combines
it with edge support. Exact certification calls the supplied objective only
where the ledger is defined and returns residual/support facts for run control
to classify. Schedule kinds, fingerprints, numerical formulas and checkpoint
representations are unchanged. The boundary is exercised in
`test_schedule_program.py`; existing chain oracles, trace-reuse and local-carry
checks remain in `test_schedule.py` and the registered-kernel schedule tests.

`execution/cross_domain.py` owns mixture, structured-recurrence and precision
updates; `execution/composites.py` owns Dirichlet parameter edges and fixed,
transition and hybrid GFE composites. Named output records distinguish
categorical, Gaussian, parameter-count and predictive blocks. Parameter
selection lives in `execution/parameters.py`; categorical cavities remain
cavities, while recurrence-drive and precision VMP beliefs are assembled by
their local owners from the supplied edge sides.

The constraint owner accepts explicit numerical threshold/table arguments
selected from `State.params`. Family validation publishes their initial integer
arrays in `Arena.log_params`; `ConstraintPlan` retains only structural metadata
and a family-prefixed `parameter_key`. Updates, free energy, host feasibility
and decoded-assignment scoring all read the current instance's values. No
runtime consumer falls back to another instance's compiled defaults.

`execution/functional.py` separates the existing stationary and fixed-particle
importance calculations. Readiness, base-posterior checks, low-ESS policy and
moment installation belong to `Mu3Lifecycle`; `Engine` installs the returned
parameters in the complete State. Shared-goal contributors are
still pooled once after all local GFE updates and before damping. Objective
terms and their once-only parameter accounting are unchanged. The exact JAX
log-zero predicate lives in `execution/support.py`. Host finiteness and exact
edge/factor support checks live in `runtime_checks.py:StateChecks` and
`state_is_finite`. The checker keeps immutable edge-cardinality buckets and
reads each instance's current parameters; it never uses compiled constraint
defaults in place of runtime values. Public termination classification remains
outside these numerical checks.

`test_execution.py` checks registered local outputs with distinct runtime
inputs under jitted batching and the synchronous owner's dependency boundary.
`test_execution_specialized.py` checks the
explicit owners, preservation of numerical failure facts, and runtime
threshold/table arguments. Existing family tests remain the mathematical
authorities; these tests protect the execution boundary, not a new inference
approximation.

The owners delegate existing discrete/Gaussian mathematics to
`kernel_runtime.py`, which also retains the node-ledger/readout calls. It never
regathers the global arena; scan carry and persistent flat layout remain unchanged.
Outgoing-message, node-energy and potential shapes are checked by abstract
JAX evaluation during `compile_template`, after host grouping and packing;
no callback receives concrete values in this check. Structural registration
and serialization do not import numerical rules. External implementation
fixtures and independent enumeration live in `test_discrete_kernels.py`.
The Gaussian extension fixture uses three unequal roles and a conditioned
middle role; covariance-conditioning/KL oracles and custom conditioned scan
tests live in `test_gaussian_kernels.py` and `test_gaussian_kernel_schedules.py`.

Structural equality, Dirichlet parameter-edge plans, hard constraints and
GFE composites retain their separate plans. Multi-arena hybrid registration
remains outside the registration foundation, as do new nonlinear Gaussian
rules. The compiler and execution extractions preserve those explicit owners.

Compatibility policy (AC6, pre-1.0): the Python registration API
(`register_kernel`, the `FactorKernel` field set) is internal-unstable and
may change in minor releases while built-ins migrate. The serialized surface
— factor type names, the `kernels` manifest keys, and the loud-failure
guarantee — is stable from the first release that writes it. External
kernels must use their own impl-id namespace (`cffg.` is reserved) and bump
MAJOR whenever message/energy semantics or the parameter schema change.

#### Bounded factor extension paths

The public compiler and Engine are orchestration boundaries, not places to add
factor-type branches. A new ordinary factor using an existing discrete, Gaussian
or EF representation registers its declaration and callbacks in its own module.
Registration is a supported extension *path*, not a stable pre-1.0 Python API and
not permission to substitute a different variational rule for the same storage.

The matrix below names semantic owners. Shared source loading, assembly and
transport consume their declarations; those consumers do not rediscover which
factor a plan represents. A specialized plan is an explicit representation of
cross-domain, composite or structural semantics, never a fallback for an unknown
type. Unknown types fail admission.

| Extension class | Classification and structural lowering | Local execution | Objective and public readout |
| --- | --- | --- | --- |
| Ordinary discrete | `registered/capabilities.py` admits the declared modes; `registered/lowering.py` builds groups from `DiscreteBinding` | `execution/registered.execute_discrete` → `kernel_runtime.discrete_message` → selected BP/VMP callback | `kernel_runtime.discrete_energy` and the declared potential, assembled by `ledger`; `readouts/discrete.py` and `readouts/registered.py` |
| Ordinary Gaussian | Same registered owners, using `GaussianBinding` and its declared conditioned role | `execution/registered.execute_gaussian` → `kernel_runtime.gaussian_message` → binding callback | `kernel_runtime.gaussian_energy`, assembled by `ledger_terms.gaussian_energy`; binding statistics/readout through `readouts/gaussian.py` and `readouts/registered.py` |
| Scalar EF | Same registered owners, using `EFBinding` and edge-family pins | `execution/registered.execute_ef` → registered message callback; the legacy EF lookup timing is retained | Registered energy through `ledger_terms.ef_energy`; `readouts/exponential_family.py` and `readouts/registered.py` |
| Multi-arena `recurrence_link` | `specialized/discovery.py` claims the factor; `specialized/mixed.lower_recurrence` completes it with ordinary BP membership | `execution/cross_domain.execute_recurrence` → existing structured-recurrence rule | `ledger_terms.recurrence_energy`; `readouts/recurrence.py` uses the existing pair/statistic mathematics |
| Hybrid P-substitution composite | `specialized/discovery.py` claims mixture + facing goal; `specialized/composites.lower_hybrid_gfe` builds the composite | `execution/composites.execute_hybrid_gfe` → existing prior/linear hybrid-GFE rule | `ledger_terms.hybrid_gfe_energy`; `readouts/hybrid.py` reports the existing diagnostics |
| Hard constraint (`exactly_k` representative) | `specialized/discovery.py` admits the structural family; `specialized/constraints.lower_constraints` owns its draft and `resolve_constraint` validates its payload | `execution/constraints.execute_constraint` selects the existing combinatorial message rule | `ledger_terms.add_constraint_terms` selects the BFE term; `ledger.constraint_satisfied` scores a decoded assignment. Ordinary discrete posterior/decode needs no separate constraint readout. |

For specialized rows, classification and lowering are two explicit stages with
one owner each. `specialized/__init__.py` joins them into `CompositeOwnership`;
`assembly_checks.py` verifies this declaration instead of repeating classification.
Runtime parameter selection belongs to `execution/parameters.py`; flat and mixed
sweeps transport its values through `execution/synchronous.py`. Registered
directed schedules use the same local executors through `ScheduledExecutors`.

| Extension class | Accepted inference meaning; what the representation does **not** enable |
| --- | --- |
| Discrete | BP cavities with declared sum/max semirings; optional mean-field VMP consumes both-sided beliefs and supports sum mode only. Max uses the declared potential for assignment scoring, not BFE. No automatic P-substitution or moment matching. |
| Gaussian | Ordinary information-form BP, optionally conditioning one declared DATA role; sum mode. Declared Gaussian first/second-moment endpoints are exact for prior/linear bindings. No Gaussian mean-field, arbitrary nonlinear projection, or P-substitution supplied by registration. |
| EF | Declared conjugate natural-parameter messages and observation contract; shipped adapters use sum-mode BP. Capability admission uses the binding's declared constraints; adding a declaration alone cannot supply a missing rule or projection. |
| Recurrence | Existing categorical source/destination pair region and Gaussian drive belief, in sum mode; the D9 structured recurrence approximation and supported topology boundaries remain in force. No P-substitution on the recurrence factor. |
| Hybrid P-substitution | Existing fixed-goal exterior mixture prior/linear composite with an explicitly substituted output, in sum mode. Interior, arbitrary nonlinear and Gaussian-only variants are unsupported. |
| Hard constraint | Existing finite-domain support factor with sum/max reductions; no registered VMP/GFE callback is inferred. Delta hardening is a separate form-constraint mechanism, not this factor's energy. |

The following are review surfaces, not a demand to edit every listed consumer.
For an extension staying within an existing contract, verify the consumer and
reuse it. Change it only when the new declaration actually requires new behavior.

| Surface | Same-arena registration | Recurrence / hybrid composite / hard constraint |
| --- | --- | --- |
| Declarations and validation | External `FactorType` + binding; registry capability/role validation. No new entry in central type-name maps. | Explicit specialized discovery/lowering; change core declarations only for genuinely new serialized structure. |
| Initializer shape | `ParameterSpec` receives role sizes; `registered/requirements.py` declares source shapes. EF retains its existing scalar-prior fallback. | Family-owned shape intent in `specialized/discovery.py`; payload recipes in the lowerer. Hard-constraint threshold/table validation remains with `specialized/constraints.py`. |
| Arrays, files and precedence | `resolution/sources.py`: explicit array override → existing file → initializer; registered packer sees the complete resolved mapping. | Same source owner and precedence; `specialized/packing.py` or constraint validator interprets the family payload. No resolver factor-name branch. |
| Named parameter writes | `ParameterSpec` key/transform declarations → `registered/requirements.py` → `resolution/addresses.py` → `params.py`. A nonlinear pack without a direct-write declaration must recompile raw blobs. | Family declarations supply the same address resolver. Constraint numerical payloads live in each `State.params`; specialized learning-only carriers are not ordinary writable factor rows. |
| Learning addresses and statistics | Existing learning-family compatibility still applies; binding statistics do not create a new learning family. EF conjugate adapters are distinct from declared `LearnableSpec` families. | Recurrence uses its existing Pólya–Gamma weight statistics/addresses. The fixed-goal hybrid representative and hard-constraint representative have no learned parameter owner; new learning behavior needs its own derivation and address contract. |
| Objectives and ownership | One registered factor term, plus existing edge-entropy correction. Assignment potential is a distinct max capability. | Recurrence owns its pair-region term; hybrid absorbs its goal and excludes the substituted output entropy; constraint contributes once. The ownership map below identifies shared-parameter overlays. |
| Flooding and scheduling | Shared registered executors; `schedule.py` classifies topology and `schedule_program.py` transports local blocks. Keep direct/flat/registerized paths on one callback. | Explicit synchronous owners also serve the iterative phase of mixed schedules. These representatives are not automatically directed BP scan nodes. |
| Colony | Deterministic distinct-input instances share compiled structure; current arrays and active masks remain per instance. Exercise invalid/stopped-neighbor isolation. | The same deterministic batching contract applies. Functional mu3 is a separate overlay that Colony still rejects; per-instance stochastic keys/status batching remains future work. |
| Persistent bytes and traces | Exact flat slots and declared parameter arrays; estimator counts State leaves. Existing discrete/Gaussian scan tests check bounded local carry, one final scatter and unchanged-shape trace reuse. | Multi-arena shapes remain explicit; constraint values are dynamic State leaves. No speedup or corrected XLA cost-analysis claim follows from this matrix; resource-estimator work remains separate. |
| Manifest and reconstruction | `kernels.manifest` stores external `(impl, version)` identities; trusted Python callbacks are installed locally, never serialized. JSON/CFFGZ reconstruct through sources and packers. | Shipped specialized plans reconstruct from the template and data, not a serialized callback or arena. A new serialized feature requires the IR/codec change path; there is no general multi-arena plugin ABI. |
| Checkpoint consumers | Existing runtime-State schemas and `_agent_checkpoint.py` remain the boundary. No schema change is required just for an additional factor under an existing binding; new state/learning fields do require one. | Existing recurrence/hybrid/constraint restoration and supported legacy decoders remain required evidence. `_constraint_checkpoint.py` owns the narrow legacy constraint-data reconstruction; general learning-resume support remains a separate roadmap. |
| Hostile inputs | Required declaration and executable sparse-support, improper-input, float32 and uncomfortable-scale cases. Representation alone never establishes robustness. | Preserve the existing family tests and loud failure statuses. Source ownership checks are numerical N/A: they add no message rule, solver, floor or inference loop. |

Composite ownership remains tied to original graph identities, not equal numeric
values or group names. These are the preservation authorities, not new scientific
validation:

| Plan/overlay | Original factors, exclusions and shared terms | Existing executable authority |
| --- | --- | --- |
| Recurrence | Executes the recurrence factor once; no facing prior/goal absorption and no GFE output exclusion. The pair-region objective remains D9. | `test_structured_recurrence_engine.py`, including fixed-drive chain, schedule parity, support and topology boundaries |
| Hybrid GFE | Executes the emission composite, absorbs its facing Gaussian goal, and excludes the substituted output from ordinary edge entropy. No second mixture plan is published. | `test_specialized.test_hybrid_and_dirichlet_exclusions_keep_their_distinct_meaning`; existing `test_hybrid_gfe.py` rule/engine authorities (D5) |
| Fixed categorical GFE | Absorbs the categorical goal and excludes only the substituted output; transition GFE instead keeps its facing factor and output entropy. | `test_specialized.test_fixed_goals_and_mu3_overlays_have_distinct_ownership` and `test_transition_gfe_retains_facing_factors_and_output_entropy` |
| Dirichlet parameter edge | Absorbs its prior; the carrier is a parameter edge, not a categorical entropy contribution. | `test_compiler_contracts.test_parameter_edges_have_count_slots_not_categorical_entropy` and parameter-edge analytic tests |
| Shared learned goal | One owner per serialized prior blob across shape groups; every contributor appears once. Equal arrays under different blob names stay independent. | `test_shared_goals.py` (D6c), `test_assembly.py` negative ownership cases |
| Functional mu3 | Overlay on GFE factors, not another factor execution. Tied future slices pool functional weights and contribute one shared KL term per parameter group. | `test_mu3.test_multiple_slices_add_functional_weights_and_parameter_ledger_once` (D6b); `test_ledger.py` off-coordinate KL case |
| Hard constraint | One execution and one local constraint energy; current threshold/table values feed updates, feasibility and assignment scoring. | `test_constraint_data.py`, `test_constraint_reconstruction.py`, existing combinatorial enumeration tests |

`test_extension_ownership.py` protects the cross-module boundaries: external
fixture names never appear in production sources, ordinary built-ins do not gain
a second type-selected numerical adapter, specialized rule/energy calls keep their
declared owners, and extracted owners neither import the facades nor form a cycle.
These are targeted regression checks, not a proof against every possible future
implementation. Existing external factor oracles, compiler/engine characterization,
Colony failure-isolation tests, and current/legacy reconstruction tests remain the
behavioral authority. The gate adds no new factor family, inference method,
performance threshold, paper-faithfulness claim or release-readiness promise.

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
explicit in serialized structure. `GfePlan` owns each local predictive update
because the categorical goal node and substituted observation edge are absorbed
into one composite; it is not also compiled as an ordinary
`DirichletGroupPlan`. An immutable `GoalParameterPlan` owns the shared goal
coordinate independently of those shape-batched composites.

Each future member emits `n_t=Abar_t zbar_t`. Members whose priors read the
same blob name form one owner across all compatible shape groups and use
`alpha=alpha0+sum_t n_t`; their state message consumes `E[log c]`. Equal numeric
priors with different blob names remain independent. The owner records original
group rows, carrier edges, and prior/count slots; existing `GfePlan.goal_*`
metadata remains a per-group compatibility view, not the pooling authority.
`goal_parameters.py` derives these owners without moving addresses or adding
runtime state. The sweep computes all predictions from pre-sweep counts, pools
each original member once, and copies the result to all carrier slots before
damping. The outer solver charts each shared message side once and broadcasts
trial coordinates to its aliases; aliases are included in the chart cache key.

The complete stable ledger charges the shared parameter term once, not once
per member or shape group. Away from a fixed point it retains the D6c.5
expected-log correction using freshly recomputed predictions, not stored
predictive messages. Global owners enter the schedule fingerprint. No template,
bundle, or checkpoint schema changes are needed: recompilation reconstructs the
owners from existing references. Runtime rollover is
explicit: `goal_prior_updates` reads a converged trial, and
`with_goal_priors` accepts it only on a fresh state. `RollingAgent` stages
planning output and commits it once at `reset_episode`, with current/pending
hyperparameters checkpointed separately.

### 3. Message rule library (`cffg.rules`)

Discrete BP/VMP and EF factors use `FactorKernel` bindings. The discrete
built-in adapters reuse the numerical update and energy functions in this
module; the engine does not dispatch their type-name tables. Specialized
GFE composites have explicit update and ledger functions. The compiled plan
selects among these surfaces:

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
  energy plus the appropriate entropy terms). `ledger.make_belief_energy` adds those node terms
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
   (→ the reporting readout), `semantic_identity`, `parameter_forms`, and
   `persistent_parameters` (checked target-role shapes → named persistent
   array schemas). `Dirichlet` serves the discrete vocabulary:
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
   the factor-local D9 pair's Pólya–Gamma statistics and roll their Gaussian
   posterior mean into point weights and offsets. Every posterior remains
   outside JIT; inference in the next run consumes point parameters rather
   than joint parameter uncertainty.

Cadence is deliberately not part of `LearnableSpec`. `Learnables` keeps a
read-only validated copy of every initial prior. `accumulate(engine, result)`
starts from the current posterior and adds a distinct trial;
`replace(engine, result)` starts from that immutable prior and recomputes one
sequence coordinate. Both require a status-bearing, converged `RunResult`,
accept an `only` name filter, pool tied sources, validate all selected
posteriors in temporary storage, and commit atomically. `write` is likewise
atomic across selected role-bearing targets.

`LearnableSpec.semantic_identity` is serialized under learning schema `1`
and participates in the structure hash. It identifies both the posterior
update and sufficient-statistic meaning. Built-in identities are
`cffg.learning.dirichlet/1.0.0`, `cffg.learning.niw/1.0.0`,
`cffg.learning.mniw/1.0.0`, `cffg.learning.component-mniw/1.0.0`, and
`cffg.recurrence.d9-structured-pair/1.0.0`. New Python declarations select
these defaults. An unknown identity is rejected on JSON loading, validation,
or compilation; numeric prior, posterior, and point values remain outside
the structure hash. Scoping transports the identity unchanged, and compiling
reconstructs its resolved descriptor from the scoped declarations.
Composition currently requires flat operands: call `expand_plates` before
`scoped` or `merge`. Compact operands are rejected with affected learnable
names; `cff-b2fz` owns preserving compact plates through composition.

`cffg.learning_schema.PersistentStateDescriptor`, returned by
`Learnables.persistent_descriptors()`, binds each learnable's name, family,
factor type, semantic identity, statistic sources, and logical parameter
targets (factor, blob, role, consumption form, layout, transform). Each named
immutable-prior and current-posterior array has a hyperparameter key, shape,
canonical NumPy dtype spelling (`dtype.str`), and SHA-256 of C-order bytes.
Array names encode each scoped owner and key independently; empty, dot,
parent, backslash, and control-character scopes are rejected. Group keys and
arena indices are derived addresses and never enter this descriptor.
The family owns required keys and shapes; generic validation checks complete
schemas, duplicate keys, compatible roles/forms, and floating dtypes without
casting. All five registered families implement the contract.
`resolution/addresses.py:resolve_learnables` checks this schema after parameter
addresses are resolved and before arena assembly. It retains the logical reader
name and semantic identity in `LearnTarget` and `LearnPlan`; the compiler facade
does not duplicate learning resolution.

`state_dict()` keeps the independently encoded `v2:` posterior keys and adds
`__cffg_learning_manifest__`, a scalar Unicode JSON inventory with schema
`1.0.0`. Restoration checks semantic identity, ownership, completeness, exact
dtype/value digests, and family admissibility before mutation. The manifest
describes priors, but this flat checkpoint API still restores only posteriors;
immutable priors remain those supplied by the destination caller. Schema-2
CFFGZ archives instead store and restore both complete states through
`Learnables.from_persistent_state`, preserving source dtypes without coercion.
With `load_state_dict(exact=False)`, the manifest may name a subset of the
destination owners; every supplied descriptor and posterior must still be
complete. `exact=True` additionally requires all destination owners. Arrays
without a supplied descriptor are rejected, and untouched owners retain their
current posteriors. Both modes stage validation before committing any changes.

Unversioned declarations and bare/slash/v2 checkpoints for the four unchanged
nonrecurrence families migrate deterministically. Complete multi-array state
is required. Historical agent structure hashes are accepted only for those
unchanged identities. Alpha CFFGZ loading accepts only the current bundle
schema, including its complete learning descriptors and arrays. Older bundle
versions are rejected; checkpoint migration has its own version contract.
Standalone unversioned recurrence JSON remains rejected. Legacy recurrence-only
checkpoint manifests retain their D9 identity/ownership/digest checks; D8 and
unidentified recurrence posteriors cannot migrate. `hyperparameter_snapshot()`
is numeric diagnostic output, not a recurrence checkpoint. The
model/runtime interface still declares which boundary learnable is
episode-scale because that choice determines rollover semantics, not merely
update frequency.

`RollingAgent.checkpoint` namespaces every entry from `Learnables.state_dict`,
including the generic manifest. Its inventory decoder also accepts historical
envelopes without that field, then delegates learning migration and complete
state validation to `Learnables` before committing any agent state.

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

For the fixed-A binary discrete reference in `test_constraint_flip.py`, replacing
the first slice's planning/P-substitution declaration with action and
observation DATA bindings leaves the named trial variables and generative
factors unchanged. Independent simplex enumeration gives the first action
posterior, exact filtering posterior, carried boundary, and second action
posterior. For each rolled one-step plan, the ordinary tree ledger outside the
goal-observation composite reduces to `-Σ_z q(z) log d_pred(z)`; adding the
local GFE term gives the enumerated planning objective. The paper-shaped mixed
history retains the conditioned past slice, so its whole-graph free energy
exceeds the rolled plan by exactly `-log p(x₁ | u₁, d₀)`. The test checks
that derived offset rather than asserting raw scalar equality. This establishes
the window transformation for that registered matrix-categorical,
joint-boundary case. It does not turn the two immutable templates into a
literal mutable graph or establish the transformation for mean-field,
continuous, hybrid, or hierarchical boundaries.

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
  numerical step-acceptance machinery and never enters `ledger.make_belief_energy` or policy
  scoring. GFE convergence requires a root of the complete
  undamped outer map, converged local solves, unchanged exact support, and a
  finite stationary FE ledger; a nonzero stationary merit is exhaustion.
  Max mode uses its message fixed-point residual and validates the decoded
  objective. Non-finite arenas/residuals are divergence, while proven empty
  exact support is infeasible with FE = +∞.
  Ordinary Engine and Colony delta reductions preserve NaN from any message
  representation; a finite change in another block cannot hide it. Their
  failed-run traces therefore retain the non-finite change as well as reporting
  divergence. Finite deltas and successful-run stopping rules are unchanged.
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
  status. Exact-zero sampled mass is inert, goal-excluded positive mass has
  zero weight, and positive sampled mass over a zero marginal predictive is an
  incoherent cavity reported as divergence; the independent D6b formula and
  Engine propagation are pinned in
  `test_mu3_hostile_support_weight_collapse_float32_and_scale` and
  `test_incoherent_predictive_support_diverges_through_engine`.
  `Engine.score_mu3_epochs` refreshes independent particle sets, accepts only
  converged epochs, and applies burn-in solely to the final policy-score
  average.
- Inference *modes* are just constraint presets on the same grounded model:
  filtering/smoothing (data on past), planning (goals on future, P-substitution
  on), what-if (clamp hypothetical δs), backward inference (goals only, no data).

### 5. Agent runtime (`cffg.agent`)

- **Receding-horizon loop**: at each tick, bind new observation (the logical
  constraint changes from planning to observed data), re-anchor each declared
  planning prior on its latest filtering posterior, re-infer, and read the
  declared control marginal (or its MAP under a δ-constraint). The runtime
  realizes that lifecycle with separate immutable planning and filtering
  templates; it does not mutate the paper's literal graph. Two compiled windows
  cover an unbounded episode without graph growth or recompilation. The exact
  two-step oracle in §3c validates this transformation for a fixed-A binary
  discrete joint boundary; other boundary families retain the named
  fixed-memory realization claim only.
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
  episode-start statistic. The versioned envelope also binds the payload to
  both serialized window structure hashes, the non-checkpointed static
  parameter values, and the horizon, solver, damping, action-selection, and JAX
  precision configuration; it inventories every logical and physical field,
  checks normalized beliefs and learned-state shapes, restores the NumPy action
  RNG, and commits only after complete validation. Unknown versions,
  structures, parameterizations, configurations, shapes, fields, and duplicate
  archive members fail specifically. Legacy single `belief` and `counts` /
  `b_counts` spellings are complete and unambiguous or refused, never silently
  dropped.
- **Functional A-message lifecycle**: `mu3=True` asks the planning builder to
  serialize `parameter_message=mu3_is_v1`; the compiled arena declaration,
  not `learn_a` or a factor name, decides whether the runtime is stochastic.
  The filter is built with `mu3=False`. Before an epoch, `Learnables.write`
  installs the current raw A posterior and `Engine.prepare_mu3` consumes the
  explicit key. One particle set remains fixed through the attempt. Repeated
  unchanged `act()` calls reuse the result; observation/reset invalidates it,
  and exact comparison with `base_alpha` prevents stale-posterior reuse.
  `trial` refreshes on every invalidated trial, while `posterior` can rerun a
  context-only change with the same particles. An action requires the ordinary
  and all evaluator statuses to be converged, leaving ESS exhaustion distinct
  from inference-budget exhaustion. The typed planning runtime retains the
  status-bearing result and next-key cursor separately from NumPy action RNG.
  For a prepared functional epoch, that envelope additionally stores the full
  `State`, result trace, evaluator and schedule identity, particle
  configuration, refresh flags, and JAX key representation. Restore validates
  prepared-state array shapes and dtypes, base-posterior identity, solver
  diagnostics, and the derived `mu3` particles, weights, moments, ESS,
  log-normalizer, ledger, and status before atomically mutating the live agent;
  analytic legacy checkpoints remain supported, while a functional legacy
  restore is loud.
  Legacy analytic envelopes that lack fixed-parameter hashes cannot detect a
  change to supplied fixed model data. Their initial runtime parameters still
  come from the supplied template and arrays; strict identity checks belong to
  the current envelope.
  Prepared functional caches with the exact older parameter layout, before
  constraints were stored in State, reconstruct only those fixed arrays from
  the supplied model data. Their other parameter fields, physical inventory,
  shapes, dtypes and cached-result checks remain mandatory. The migration lives
  in `_constraint_checkpoint.py`. New checkpoints use the existing parameter-
  array encoding for these values; no new envelope version or model field is
  introduced.
- Sensor/actuator adapters are plain Python callbacks mapping env observations to
  one-hot data bindings and control marginals to env actions.

### 5b. Colony execution (many small graphs, one machine)

`Colony` owns an instance table of `(structure_hash, State)` pairs and caches one
`Engine` per structure hash. Each `State` is an Equinox pytree containing
messages, parameters, and conditioned observations; structure remains in the
cached engine. Ordinary iterative instances are grouped by structure plus State
pytree layout, array shapes and dtypes before stacking. Each compatible batch
evaluates one scheduled iteration under `filter_vmap`; different structures or
numeric layouts run as separate groups. Different allowed-tuple table lengths
therefore cannot enter one stack, while same-shaped tables with different
contents retain vectorized execution. One cached Engine can have several JAX
shape/dtype specializations; numeric contents do not define its identity.
Mixed-GFE instances reuse the same compiled
trust kernels but currently run one host controller per instance so each keeps
its own trust radius, active support, and termination reason. Inter-instance
coupling happens through host-side data binding and marginal reads, never by
merging graphs.

Initial non-finite and proven empty-support states are classified before
schedule priming. Rejected members retain their input values and report zero
iterations. Non-finite input reports both overall and inner divergence, with
an infinite inner residual, matching standalone `Engine.run`. Healthy peers
continue through the vmapped updates; a fully rejected group skips priming and
updates. `tests/test_colony_failures.py` checks complete result/state parity
across both float precisions, zero/positive budgets and mixed/flood schedules.
Later divergence or infeasibility reports an infinite public residual, even
when earlier updates produced a finite residual. Inner-solver facts and FE
trace entries are retained; a budget stop with a defined ledger keeps its
existing residual. Controlled-failure tests compare every public result field
with standalone execution and verify that healthy batch members continue.
`tests/test_constraint_data.py` additionally checks distinct per-instance limits
and tables, both spawn orders, all four numeric consumers, and actual batching
of mixed float precisions and table lengths. `test_constraint_reconstruction.py`
checks bundles, current and supported legacy agent reconstruction, and strict
fixed-data mismatch rejection.

Ordinary runs share `IterationProgress` assessment and `finalize_iteration`
result construction with Engine; Colony owns batching and active-member
freezing, not a second termination policy. The convergence loop and
per-instance status bookkeeping are currently Python
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

## Canonical rSLDS consumer validation (2026-07-31)

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
- **Recurrent stick-breaking link** (cff-k9ws, landed): consecutive regimes
  are coupled to a Gaussian drive with source-specific logistic sticks and a
  Pólya–Gamma/JJ VMP update, supplying the transition primitive needed by the
  rSLDS graph without materializing a state-dependent transition tensor.
- **M12 — Nonlinear factors** (cff-o7hs): Jacobian linearization and
  sigma-point moment matching; builds on the endpoint- and projection-aware
  `MOMENT_MATCHING` schema and explicit legacy migration.
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
