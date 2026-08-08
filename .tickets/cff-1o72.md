---
id: cff-1o72
status: open
deps: [cff-o7hs]
links: [cff-fxmj, cff-o7hs, cff-ri8m, cff-l6i5, cff-209l]
created: 2026-07-08T16:50:10Z
type: feature
priority: 2
assignee: coopwilliams
parent: cff-ahps
---
# Neural factors: NN nodes with backprop, Laplace messages, amortized recognition

## Background in Plain English

CFFG needs neural factors for nonlinear relationships that are too large or flexible for a table or a hand-written function. This ticket adds a deterministic neural map between Gaussian variables, trains its parameters from a free-energy objective, and lets a trained network propose better starting messages. It builds on cff-o7hs, which owns nonlinear Gaussian messages and their approximate ledger terms. It preserves the project rule that serialized templates remain the source of truth and that `cffg.core` can load and edit them without importing JAX or Equinox.

The neural node does not make an approximate objective exact. Its ledger entry remains a named surrogate, while analytic factors elsewhere in the graph keep their Bethe or Generalised free-energy meaning. Solver gradients describe a stationary solution only after the runtime has certified that solution; an exhausted, diverged, or infeasible run does not silently become training data.

## Outcomes / Acceptance Criteria

1. cff-o7hs is closed with its nonlinear Gaussian message, stationary-site, damping, and approximate-ledger contracts satisfied before this ticket closes.
2. The serialized template schema declares a deterministic neural factor `y = f_theta(x)` between Gaussian edges.
   2.1. The template records a namespaced implementation identity and compatible version, the network architecture data needed to reconstruct the factor, and references to external parameter blobs. It does not store an Equinox module or a large parameter array inline.
   2.2. Core template loading, validation, editing, hashing, and bundle inventory do not import JAX or Equinox. Compilation fails with a specific missing-implementation or incompatible-version error when the required neural kernel is unavailable.
   2.3. A template and its blobs round-trip through JSON and `.cffgz` without changing the factor identity, parameter values, structure hash, or compiled neural-factor behavior.
3. The compiled neural factor evaluates the network and sends Gaussian messages through the nonlinear rule contract delivered by cff-o7hs.
   3.1. Forward values and Jacobians agree with an independent analytic network fixture within stated float64 and float32 tolerances.
   3.2. From an identical initial state, synchronous and scheduled execution produce messages, marginals, run status, and ledger values that agree within the stated tolerance.
   3.3. Each neural node records which cff-o7hs nonlinear constraint mode it uses. Its ledger term is labeled with that mode's named Laplace, linearization, or moment-matching surrogate, and analytic terms elsewhere in the graph retain their analytic labels.
4. The training API optimizes neural parameters against BFE and against GFE only when the complete graph already lies within the documented GFE scope.
   4.1. The training API differentiates a certified stationary solution with respect to the neural parameters through the stationary equations. Gauge freedoms are removed or projected before the implicit derivative is solved.
   4.2. On a deterministic fixture with one scalar input, one scalar output, and no more than 20 neural parameters, parameter gradients agree with independent central finite differences at a tolerance justified by the finite-difference step, objective conditioning, and active dtype.
   4.3. The training API reports and refuses to apply a stationary-solution gradient when inference is budget-exhausted, diverged, infeasible, non-finite, or outside its declared surrogate domain.
   4.4. On a seeded nonlinear-regression fixture with a held-out input grid, free-energy training lowers held-out error below both the untrained network and a linear-Gaussian baseline. The fixture records its generating map, observation noise, split, seeds, stopping rule, and numeric acceptance threshold before the final comparison is run.
   4.5. Saving and restoring the neural parameters and training state preserves the next parameter update and objective value within the checkpoint tolerance; loading a legacy checkpoint either migrates it explicitly or fails with a specific incompatibility error.
5. A trained amortizer proposes initial messages for a neural-factor graph without changing the model or its certified stationary solution.
   5.1. On a paired held-out benchmark, amortized and neutral initialization reach stationary beliefs and ledger values that agree within the benchmark tolerance. A reported 95% confidence interval for the paired sweep-count reduction excludes zero.
   5.2. An amortized proposal never restores a state excluded by exact support and never converts an improper belief into a proper one through a numeric floor.
   5.3. If the amortizer supplies an importance-sampling proposal, the evaluator applies the corresponding importance weights, uses an explicit JAX PRNG key, reports effective sample size, and reproduces particles and results when the key and inputs are replayed.
6. The neural rule satisfies the numerical robustness contract.
   6.1. Sparse discrete support is N/A for a factor whose declared edges are nondegenerate Gaussian variables with support on `R^d`. A singular or deterministic Gaussian limit is either represented by a proved rule or rejected; it is not approximated with an absolute precision clamp.
   6.2. A vacuous Gaussian input follows the cff-o7hs neutral-message rule. An improper covariance, precision, site, or incoming belief produces an explicit infeasible or diverged result and a non-finite surrogate ledger value, never a finite converged result.
   6.3. Float32 execution derives floors, damping guards, positive-definiteness checks, finite-difference steps, and solver tolerances from the active dtype.
   6.4. A benchmark with at least ten times the basic fixture's input dimension, output dimension, parameter count, or neural-factor count completes within a documented runtime and memory envelope or returns a machine-readable non-converged status. The resource estimator includes neural parameters and declared per-sweep work.
7. The implementation covers compiler grouping and parameter packing, Gaussian arena addressing, synchronous steps, scheduled stages, BFE/GFE dispatch within the declared scope, run-status propagation, readouts, serialization manifests, resource estimation, and documented exports. It also covers `member_of`, or records why `member_of` is structurally irrelevant when neural parameters reside in a separately typed pytree.
8. `README.md`, `docs/ARCHITECTURE.md`, `docs/DERIVATIONS.md`, and `docs/LIMITATIONS.md` document construction, training, checkpoint and blob behavior, stationary-gradient requirements, the named surrogate, and the boundary against arbitrary nonlinear GFE.

## Methodological Suggestions

These suggestions are optional. Another method is acceptable if it satisfies the numbered outcomes with equivalent artifacts and executable evidence.

- Use the **New factor or rule family** row in `docs/ARCHITECTURE.md` as the primary change path. Also use the **Learnable or checkpoint feature** row for parameter updates and persistence, and the **Runtime numerical or status invariant** row for stationary-gradient gating.
- Build on the Gaussian `FactorKernel` binding supplied through cff-209l and cff-o7hs. Keep network callables and Equinox modules on the compiler/runtime side of the core-IR boundary. Store portable architecture data in the template and parameter arrays in `.npy` or `.safetensors` blobs.
- Audit compiler grouping, slot addressing, parameter packing, arenas, `member_of`, synchronous `_make_step`, scheduled `_batch_stage_fn` and `_scan_stage_fn`, `_make_bfe`, schedule classification, readouts, serialization manifests, estimator accounting, and package exports. Explicitly check host/JIT, flooding/scheduled, single-instance/`vmap`, and latest-schema/legacy checkpoint pairs; mark a pair structurally irrelevant only after stating why.
- Consider `jax.jacfwd` for the local network Jacobian and a custom JVP/VJP for the stationary solve. Attach the implicit derivative only to a solution whose original stationarity residual meets the runtime certificate, following the gauge-reduced patterns already used by the GFE inner and mixed outer solvers.
- Derive the Laplace or other named neural-node message and ledger surrogate before implementation. Put formal equations and the implicit-gradient argument in rule docstrings and `docs/DERIVATIONS.md`; record the approximation boundary in `docs/LIMITATIONS.md`.
- Use independent oracles: an analytic affine or shallow network for values and Jacobians, central finite differences for parameter gradients, a dense-grid posterior for a nonlinear factor with at most two input dimensions or a Monte Carlo posterior with reported sampling error, and paired held-out models for amortized initialization. Record seeds, tolerances, benchmark hardware where timing matters, and confidence-interval construction.
- Test literal support preservation, vacuous and improper Gaussian inputs, float32, uncomfortable network/factor scale, non-finite parameters, exhausted inference, replayed PRNG keys, serialization failure modes, and host/JIT plus `vmap` agreement.
- Per Cooper's 2026-07-08 delegation decision, keep message and ledger derivations plus the stationary-gradient argument under top-level review. Well-specified Equinox plumbing, training-loop scaffolding, analytic golden fixtures, and narrowly framed literature questions may be delegated and reviewed at top level.
- During implementation, run the focused neural rule, kernel, compiler, schedule, runtime-status, serialization, bundle, learning, estimator, and engine-oracle tests. After the final code change, run the dashboard-backed short suite once under the `AGENTS.md` protocol; run marked slow behavioral or scale tests only when this ticket's final evidence requires them.

## Future work

- cff-ahps uses this ticket to close the neural-factor branch of the universal factor-library roadmap after every other epic prerequisite is satisfied.
- Arbitrary Gaussian-only or nonlinear GFE messages remain excluded. cff-dfpi and cff-izjd selected the bounded discrete-regime epistemic route; expanding nonlinear GFE requires a separate ticket and a new derivation.
- Neural structure search, architecture selection, and automatic network growth are not part of this ticket. No downstream ticket for that work is identified.

## Definitions

- **Amortized recognition:** A trained function that predicts an inference starting point or proposal from model inputs so repeated inference can use less work.
- **Bethe free energy (BFE):** The graph objective formed from factor-local energy and entropy terms plus edge-entropy overcounting corrections.
- **Central finite difference:** A numerical derivative computed from objective values at equal positive and negative parameter perturbations.
- **Effective sample size:** A diagnostic that summarizes how concentrated a set of importance weights is.
- **Equinox:** The JAX-compatible neural-network and pytree library already used by CFFG runtime state.
- **FactorKernel:** The executable registration record that binds a factor's structure, parameters, messages, energy, robustness declarations, and serialized identity.
- **Gauge freedom:** A parameter direction that changes a message representation without changing the represented belief.
- **Generalised free energy (GFE):** The active-inference objective used for graph regions with a supported P-substitution construction.
- **Implicit differentiation:** Computing how a stationary solution changes by differentiating its defining equations instead of differentiating every solver iteration.
- **Importance sampling:** Estimating a target expectation with samples from another distribution and weights that correct for that proposal distribution.
- **JAX PRNG key:** Explicit state that makes JAX random sampling reproducible and traceable.
- **Laplace approximation:** A Gaussian approximation derived from local curvature around a mode.
- **Pytree:** A nested container of arrays that JAX transformations traverse as one parameter object.
- **Stationary solution:** A solver result whose declared stationarity residual satisfies its convergence certificate.
- **Support:** The inputs or states to which a probability model assigns nonzero density or probability.
- **Surrogate:** A named approximate objective term used in place of an unavailable analytic free-energy term.
- **Vacuous message:** A neutral incoming message that contributes no information.

## Notes

**2026-08-08T04:18:34Z**

Retained wiggle-audit candidates and intended meanings after the bounded current scan: n_a = sparse discrete support is structurally irrelevant to the declared nondegenerate Gaussian factor; preserve = a stated semantic or checkpoint-value invariant; register = executable FactorKernel registration; implement = the planned code and integration work owned by this ticket; support = literal probability support or the documented mathematical objective scope, as stated locally; complete = the locally named whole graph or execution termination; deliver = the prerequisite contract supplied by cff-o7hs; remaining = a preserved invariant or an explicitly excluded future capability; approximate = a named Laplace, linearization, or moment-matching method; close = acceptance-based ticket closure; certify = the runtime stationarity-residual certificate; check = a numeric guard or an explicit duplicated-surface review; converge = the machine-readable converged/non-converged status; correct = importance weights mathematically compensate for the proposal distribution; equivalent = equal acceptance artifact and executable-evidence coverage; estimate = the named resource estimator or importance-sampling estimation; evidence = named automated checks and reference artifacts; exact = analytic mathematical meaning or literal zero support; match = the technical phrase moment matching; oracle = an independent analytic, finite-difference, grid, or Monte Carlo reference; reproduce = equal particles and results under replayed keys and inputs; robust = the AGENTS.md numerical robustness contract; test = automated test execution or named test artifacts; validate = core input/schema validation; infer = fixed-parameter state inference; learn = the integrated parameter-training test surface; learnable = the named persistent-parameter change-path category; scale = increased network dimensions, parameter count, or factor count; can = demonstrated core loading/editing capability or possibility in the amortization definition; may = explicit permission to delegate bounded work; explicit_owner = cff-o7hs owns prerequisite nonlinear rules while this ticket owns its numbered neural outcomes; named_ticket_handoff = the concrete cff-o7hs prerequisite owner; at_least = the inclusive ten-times lower bound; every = each cff-ahps prerequisite or each solver iteration, as stated locally; only = a temporal gate, structural irrelevance gate, test-selection condition, or explicit scope boundary. Intelligible-ticket review: complete.

Intelligible-ticket review: complete.
